from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from STSDataset import STSDataset
from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device


class Seq2Seq(nn.Module):
    def __init__(self, model_names, meta_embedding_dim, nonlinear, attention_head_num, attention_dropout_ratio):
        super().__init__()
        self.model_names = model_names
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.total_dim = sum(self.embedding_dims.values())
        self.meta_embedding_dim = meta_embedding_dim # 半分，同じ，倍にしてどうか
        self.projection_matrices = nn.ModuleDict({model: nn.Linear(self.embedding_dims[model], self.meta_embedding_dim, bias=False) for model in self.model_names})
        self.parameter_vector = nn.Linear(self.meta_embedding_dim, 1, bias=False)
        self.nonlinear = nonlinear # None, nn.ReLU() # linear で試してみる
        self.attention = nn.MultiheadAttention(embed_dim=self.meta_embedding_dim, num_heads=attention_head_num, dropout=attention_dropout_ratio)
        self.layer_norm = nn.LayerNorm(normalized_shape=1)


class TrainSeq2seqWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu'):
        self.device = get_device(device)
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.source = {model: GetSentenceBertWordEmbedding(model, device=self.device) if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']) else GetHuggingfaceWordEmbedding(model, device=self.device) for model in self.model_names}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.total_dim = sum(self.embedding_dims.values())

        self.meta_embedding_dim = 1024 # 半分，同じ，倍にしてどうか
        self.nonlinear = nn.ReLU() # None, nn.ReLU() # linear で試してみる
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.attention_head_num = 1
        self.attention_dropout_ratio = 0.3
        self.sentence_pooling_method = 'avg'
        self.learning_ratio = 0.01
        self.gradient_clip = 0.0
        self.weight_decay = 1e-5
        self.lambda_e, self.lambda_d = 0.1, 0.1 # lambda = 0でやってみる （直交性を入れたいから入れている→学習ラウンドごとに長さを１にするとか．W, W.Tの結果を見て，対角の値QR分解をかける）

        self.s2s = Seq2Seq(model_names=self.model_names,
                           meta_embedding_dim=self.meta_embedding_dim,
                           nonlinear=self.nonlinear,
                           attention_head_num=self.attention_head_num,
                           attention_dropout_ratio=self.attention_dropout_ratio).to(self.device)
        self.s2s.train()
        self.attention = self.s2s.attention
        self.projection_matrices = self.s2s.projection_matrices
        self.nonlinear = self.s2s.nonlinear
        self.parameter_vector = self.s2s.parameter_vector
        self.parameters = self.s2s.parameters()

        for k, v in self.s2s.named_parameters():
            print("{}, {}, {}".format(v.requires_grad, v.size(), k))

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.save_model_path = f'../models/seq2seq-{self.tag}.pkl'
        self.information_file = f'../results/seq2seq/info-{self.tag}.txt'
        self.which_prime_output_to_use_in_testing = 'encoder'


    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=True):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores, losses = [], [], []
        padded_sequences, padding_masks = self.modify_batch_embeddings_to_easy_to_compute(batch_embeddings)

        pooled_fe_primes, pooled_fd_primes, fe_primes, fd_primes = [], [], [], []
        fe_prime_outputs, fd_prime_outputs = [], []
        for i in range(2):  # for input sentences, sentence1 and sentence2
            pooled_fe_prime, pooled_fd_prime, fe_prime, fd_prime = self.step({model_name: padded_sequences[model_name][i] for model_name in self.model_names},
                                                                             padding_mask={model_name: padding_masks[model_name][i] for model_name in self.model_names})
            # pooled_fe_primes.append(pooled_fe_prime)
            # pooled_fd_primes.append(pooled_fd_prime)
            # fe_primes.append(fe_prime)
            # fd_primes.append(fd_prime)

            ## calculate loss of original and prime
            ld = 1. - self.cos2(padded_sequences[self.model_names[self.get_decoder_model_idx()]][i], fd_prime)
            le = 1. - self.cos2(padded_sequences[self.model_names[self.get_encoder_model_idx()]][i], fe_prime)

            ld, le = torch.sum(ld), torch.sum(le)

            l_we = self.projection_matrices[self.model_names[self.get_encoder_model_idx()]]
            l_we = torch.mm(l_we.weight, l_we.weight.T) - torch.eye(self.meta_embedding_dim, device=self.device) # sub identity matrix for orthogonalization
            l_we = torch.square(torch.norm(l_we)).to(self.device)

            l_wd = self.projection_matrices[self.model_names[self.get_decoder_model_idx()]]
            l_wd = torch.mm(l_wd.weight, l_wd.weight.T) - torch.eye(self.meta_embedding_dim, device=self.device) # sub identity matrix for orthogonalization
            l_wd = torch.square(torch.norm(l_wd)).to(self.device)

            loss = le + ld + self.lambda_e * l_we + self.lambda_d * l_wd

            running_loss += loss.item()
            losses.append(loss)
            fe_prime_outputs.append(pooled_fe_prime)
            fd_prime_outputs.append(pooled_fd_prime)

        if with_calc_similality:
            if self.which_prime_output_to_use_in_testing == 'encoder':
                sys_score = list(map(self.similarity, fe_prime_outputs[0].tolist(), fe_prime_outputs[1].tolist()))
            else:
                sys_score = list(map(self.similarity, fd_prime_outputs[0].tolist(), fd_prime_outputs[1].tolist()))

            sys_scores.extend(sys_score)
            gs_scores.extend(scores)

        print(running_loss)

        if with_training:
            loss = torch.mean(torch.stack(losses))
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
            self.optimizer.step()

        # torch.cuda.empty_cache()

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        # dim: sentence length, meta embedding dim
        x = [self.projection_matrices[model_name](feature[model_name]) for model_name in self.model_names]

        batch_size = len(x[0])
        sentence_length = x[0].shape[1]
        if self.nonlinear is not None:
            # dim: sentence length, meta embedding dim
            x = self.nonlinear(x[0] + x[1])
            # dim: sentence length
            x = self.parameter_vector(x)
            x_skip = x.transpose(0, 1)
        else:
            # dim: batch size, sentence length
            x = self.parameter_vector(x[0] + x[1])
            x_skip = x.transpose(0, 1)

        # dim: sentence length, batch size, embedding dim
        # x = self.s2s.layer_norm(x_skip)
        # x = x_skip
        x, attention_weight = self.attention(x, x, x)
        # x = x + x_skip # residual connection
        x = x.transpose(0, 1).squeeze(2)

        # dim: sentence length, source embedding dim
        fd_prime = torch.stack([torch.stack([torch.mul(x[b][l], feature[self.model_names[self.get_encoder_model_idx()]][b][l]) for l in range(sentence_length)]) for b in range(batch_size)])
        fe_prime = torch.stack([torch.stack([torch.mul(x[b][l], feature[self.model_names[self.get_decoder_model_idx()]][b][l]) for l in range(sentence_length)]) for b in range(batch_size)])

        if self.sentence_pooling_method == 'avg':
            pooled_fe_prime = torch.mean(fe_prime, dim=1)
            pooled_fd_prime = torch.mean(fd_prime, dim=1)
        elif self.sentence_pooling_method == 'concat':
            pooled_fe_prime = torch.cat([out for out in fe_prime], dim=1)
            pooled_fd_prime = torch.cat([out for out in fd_prime], dim=1)
        elif self.sentence_pooling_method == 'max':
            pooled_fe_prime, _ = torch.max(fe_prime, dim=1)
            pooled_fd_prime, _ = torch.max(fd_prime, dim=1)

        return pooled_fe_prime, pooled_fd_prime, fe_prime, fd_prime

    def get_encoder_model_idx(self):
        return 0

    def get_decoder_model_idx(self):
        return 1

    def get_save_path(self, tag):
        return f'../models/seq2seq-{self.tag}-{tag}.pkl'

    def save_model(self):
        torch.save(self.s2s.state_dict(), self.get_save_path('s2s'))
        # for key, value in self.projection_matrices.items():
        #     torch.save(value, self.get_save_path('projection-' + key))
        # torch.save(self.attention.state_dict(), self.get_save_path('attention'))
        # torch.save(self.parameter_vector, self.get_save_path('param_vector'))
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            self.s2s.load_state_dict(torch.load(self.get_save_path('s2s')))
            self.s2s.to(self.device)
            # for key in self.projection_matrices.keys():
            #     self.projection_matrices[key] = torch.load(self.get_save_path('projection-' + key))
            # self.attention.load_state_dict(torch.load(self.get_save_path('attention')))
            # self.parameter_vector = torch.load(self.get_save_path('param_vector'))

    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'meta_embedding_dim: {self.meta_embedding_dim}\n')
            f.write(f'nonlinear: {str(self.nonlinear)}\n')
            f.write(f'attention_head_num: {self.attention_head_num}\n')
            f.write(f'attention_dropout_ratio: {self.attention_dropout_ratio}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'which_prime_output_to_use_in_testing: {self.which_prime_output_to_use_in_testing}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'lambda_e: {self.lambda_e}\n')
            f.write(f'lambda_d: {self.lambda_d}\n')
            f.write(f'batch_size: {self.batch_size}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/seq2seq-{self.tag}.pkl'
        self.information_file = f'../results/seq2seq/info-{self.tag}.txt'

    def update_hyper_parameters(self, hyper_params):
        self.meta_embedding_dim = hyper_params['meta_embedding_dim'] # 半分，同じ，倍にしてどうか
        self.attention_dropout_ratio = hyper_params['attention_dropout_ratio']
        if hyper_params['activation'] == 'none':
            self.nonlinear = None # None, nn.ReLU() # linear で試してみる
        elif hyper_params['activation'] == 'relu':
            self.nonlinear = nn.ReLU()
        elif hyper_params['activation'] == 'tanh':
            self.nonlinear = nn.Tanh()

        self.s2s = Seq2Seq(model_names=self.model_names,
                           meta_embedding_dim=self.meta_embedding_dim,
                           nonlinear=self.nonlinear,
                           attention_head_num=self.attention_head_num,
                           attention_dropout_ratio=self.attention_dropout_ratio).to(self.device)

        self.attention = self.s2s.attention
        self.projection_matrices = self.s2s.projection_matrices
        self.nonlinear = self.s2s.nonlinear
        self.parameter_vector = self.s2s.parameter_vector
        self.parameters = self.s2s.parameters()

        self.sentence_pooling_method = hyper_params['sentence_pooling_method']
        self.learning_ratio = hyper_params['learning_ratio']
        self.gradient_clip = hyper_params['gradient_clip']
        self.weight_decay = hyper_params['weight_decay']
        self.lambda_e, self.lambda_d = hyper_params['lambda_e'], hyper_params['lambda_d'] # lambda = 0でやってみる （直交性を入れたいから入れている→学習ラウンドごとに長さを１にするとか．W, W.Tの結果を見て，対角の値QR分解をかける）

        super().__init__()

        self.batch_size = hyper_params['batch_size']
        self.datasets['train'].batch_size = self.batch_size
        self.which_prime_output_to_use_in_testing = hyper_params['which_prime_output_to_use_in_testing']


class EvaluateSeq2seqModel(AbstractGetSentenceEmbedding):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = get_device(device)
        self.tag = get_now()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainSeq2seqWithSTSBenchmark(device=device)
        self.model_tag = [f'seq2seq-{self.tag}']
        self.output_file_name = 'seq2seq.txt'
        self.which_prime_output_to_use_in_testing = self.model.which_prime_output_to_use_in_testing
        self.model.s2s.eval()

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):

        with torch.no_grad():
            padded_sequences, padding_masks = self.modify_batch_sentences_for_senteval(batch)
            # dim: sentence length, input dim
            pooled_fe_prime, pooled_fd_prime, fe_prime, fd_prime = self.model.step({model_name: padded_sequences[model_name] for model_name in self.model_names},
                                                                                   padding_mask={model_name: padding_masks[model_name] for model_name in self.model_names})
        if self.which_prime_output_to_use_in_testing == 'encoder':
            outputs = pooled_fe_prime.tolist()
        else:
            outputs = pooled_fd_prime.tolist()
        return np.array(outputs)

    def set_tag(self, tag):
        self.model_tag[0] = f'{self.model_tag[0]}-{tag}'
        self.tag = tag

    def save_summary_writer(self, rets):
        sw = SummaryWriter('runs/Seq2seq')

        hp = {'source': ','.join(self.model.model_names),
              'meta_embedding_dim': self.model.meta_embedding_dim,
              'non_linear': str(self.model.nonlinear),
              'attention_head_num': self.model.attention_head_num,
              'attention_dropout_ratio': self.model.attention_dropout_ratio,
              'tokenization_mode': self.model.tokenization_mode,
              'subword_pooling_method': self.model.subword_pooling_method,
              'sentence_pooling_method': self.model.sentence_pooling_method,
              'which_prime_output_to_use_in_testing': self.model.which_prime_output_to_use_in_testing,
              'learning_ratio': self.model.learning_ratio,
              'gradient_clip': self.model.gradient_clip,
              'weight_decay': self.model.weight_decay,
              'batch_size': self.model.batch_size,
              'lambda_e': self.model.lambda_e,
              'lambda_d': self.model.lambda_d,
              'tag': self.tag,
              'best_epoch': rets['best_epoch']}
        metrics = {'dev_pearson': rets['dev_pearson'],
                   'test_pearson': rets['test_pearson'],
                   'wpearson': rets['pearson'],
                   'wspearman': rets['spearman']}
        sw.add_hparams(hparam_dict=hp, metric_dict=metrics)
        sw.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='select device')
    args = parser.parse_args()

    if args.device != 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        es_metrics = 'pearson'
        if es_metrics == 'dev_loss':
            vw = ValueWatcher(mode='minimize')
        else:
            vw = ValueWatcher()
        cls = EvaluateSeq2seqModel(device=args.device)
        trainer = cls.model # TrainSeq2seqWithSTSBenchmark(device=args.device)
        # trainer.model_names = cls.model_names
        # trainer.set_tag(cls.tag)
        print(cls.tag)

        rets = cls.model.inference(mode='dev')
        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            cls.model.train_epoch()
            cls.model.datasets['train'].reset(with_shuffle=True)
            rets = cls.model.inference(mode='dev')
            if es_metrics == 'pearson':
                vw.update(rets[es_metrics])
            else:
                vw.update(rets[es_metrics])
            if vw.is_updated():
                cls.model.save_model()
                dp.set('best-epoch', vw.epoch)
                dp.set('best-score', vw.max_score)
            dp.set(f'scores', rets)
        print(f'dev best scores: {cls.model.get_round_score(dp.get("best-score")[-1]) :.2f}')

        cls.model.load_model()
        rets = cls.model.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        rets = cls.single_eval(cls.model_tag[0])
        cls.model.append_information_file([f'es_metrics: {es_metrics}'])
        cls.model.append_information_file(rets['text'])
    else:
        cls = EvaluateSeq2seqModel(device=args.device)
        trainer = cls.model # TrainSeq2seqWithSTSBenchmark(device=args.device)
        tag = None # '03152021104413330773'
        if tag is not None:
            trainer.set_tag(tag)
            cls.set_tag(tag)
            trainer.load_model()
            rets = trainer.inference(mode='test')
            print(f'test best scores: ' + ' '.join(rets['prints']))
            cls.model = trainer
        model_tag = cls.model_tag[0]
        if cls.tag != trainer.tag:
            model_tag = f'{model_tag}-{trainer.tag}'
        print('single eval')
        rets = cls.single_eval(model_tag)
        trainer.append_information_file([f'es_metrics: {es_metrics}'])
        trainer.append_information_file(rets['text'])

