import os
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from STSDataset import STSDataset
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device


class TrainSeq2seqWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu'):
        self.device = get_device(device)
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.embedding_dims = {model: self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])

        self.meta_embedding_dim = 300
        self.projection_matrices = {key: torch.randn((self.embedding_dims[key], self.meta_embedding_dim), requires_grad=True).to(self.device) for key in self.model_names}

        self.nonlinear = nn.ReLU()
        self.parameter_vector = torch.randn(self.meta_embedding_dim).to(self.device)

        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.attention_head_num = 1
        self.attention_dropout_ratio = 0.2
        self.sentence_pooling_method = 'avg'
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=self.attention_head_num, dropout=self.attention_dropout_ratio).to(self.device)
        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.weight_decay = 0.005
        self.lambda_e, self.lambda_d = 0.01, 0.01
        self.parameters = list(self.attention.parameters()) + list(self.projection_matrices.values()) + [self.parameter_vector]

        super().__init__()

        self.save_model_path = f'../models/seq2seq-{self.tag}.pkl'
        self.which_prime_output_to_use_in_testing = 'decoder'

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=True):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores = [], []
        for embeddings, score in zip(batch_embeddings, scores):
            fe_prime_outputs, fd_prime_outputs = [], []
            for i in range(2):  # for input sentences, sentence1 and sentence2
                fe_prime, fd_prime = self.step({model_name: torch.FloatTensor(embeddings[model_name][i]) for
                    model_name in self.model_names})

                ## calculate loss
                target_embeddings = {model_name: torch.FloatTensor(embeddings[model_name][i]) for
                    model_name in self.model_names}
                ld_2norm = torch.norm(target_embeddings[self.model_names[self.get_decoder_model_idx()]] - fd_prime, dim=1)
                le_2norm = torch.norm(target_embeddings[self.model_names[self.get_encoder_model_idx()]] - fe_prime, dim=1)

                ld_2norm_square = torch.square(ld_2norm)
                le_2norm_square = torch.square(le_2norm)

                ld = torch.sum(ld_2norm_square, dim=0)
                le = torch.sum(le_2norm_square, dim=0)

                we = self.projection_matrices[self.model_names[self.get_encoder_model_idx()]]
                l_we = torch.square(
                    torch.norm(torch.einsum('pq, rs->qr', we, we.T) - torch.eye(self.meta_embedding_dim)))
                wd = self.projection_matrices[self.model_names[self.get_decoder_model_idx()]]
                l_wd = torch.square(
                    torch.norm(torch.einsum('pq, rs->qr', wd, wd.T) - torch.eye(self.meta_embedding_dim)))

                loss = le + ld + self.lambda_e * l_we + self.lambda_d * l_wd

                if with_training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
                    self.optimizer.step()

                running_loss += loss.item()
                fe_prime_outputs.append(fe_prime)
                fd_prime_outputs.append(fd_prime)

            if with_calc_similality:
                if self.which_prime_output_to_use_in_testing == 'encoder':
                    sys_score = self.similarity(fe_prime_outputs[0].tolist(), fe_prime_outputs[1].tolist())
                else:
                    sys_score = self.similarity(fd_prime_outputs[0].tolist(), fd_prime_outputs[1].tolist())

                sys_scores.append(sys_score)
                gs_scores.append(score)

        return gs_scores, sys_scores, running_loss

    def step(self, feature):
        # dim: sentence length, meta embedding dim
        projected_feature = [
            torch.matmul(torch.FloatTensor(feature[model_name]), self.projection_matrices[model_name]) for
            model_name in self.model_names]

        # dim: sentence length, meta embedding dim
        nonlineared_feature = self.nonlinear(projected_feature[0] + projected_feature[1])
        # dim: sentence length
        projected_feature = torch.einsum('p, qr->q', self.parameter_vector, nonlineared_feature)
        projected_feature = projected_feature.view(-1, 1, 1)

        # feature = torch.cat([torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names], dim=1).unsqueeze(0).transpose(0, 1)
        attention_score, attention_weight = self.attention(projected_feature, projected_feature, projected_feature)
        attention_score = attention_score.view(-1, 1)

        target_embeddings = {model_name: torch.FloatTensor(feature[model_name]) for model_name in
                             self.model_names}

        # dim: sentence length, source embedding dim
        fd_prime = torch.einsum('pq, rs->ps', attention_score,
                                target_embeddings[self.model_names[self.get_encoder_model_idx()]])
        fe_prime = torch.einsum('pq, rs->ps', attention_score,
                                target_embeddings[self.model_names[self.get_decoder_model_idx()]])

        if self.sentence_pooling_method == 'avg':
            pooled_fe_prime = torch.mean(fe_prime, dim=0)
            pooled_fd_prime = torch.mean(fd_prime, dim=0)
        elif self.sentence_pooling_method == 'concat':
            pooled_fe_prime = torch.cat([out for out in fe_prime])
            pooled_fd_prime = torch.cat([out for out in fd_prime])
        elif self.sentence_pooling_method == 'max':
            pooled_fe_prime, _ = torch.max(fe_prime, dim=0)
            pooled_fd_prime, _ = torch.max(fd_prime, dim=0)

        return pooled_fe_prime, pooled_fd_prime

    def get_encoder_model_idx(self):
        return 0

    def get_decoder_model_idx(self):
        return 1

    def get_save_path(self, tag):
        return f'../models/seq2seq-{self.tag}-{tag}.pkl'

    def save_model(self):
        for key, value in self.projection_matrices.items():
            torch.save(value, self.get_save_path('projection-' + key))
        torch.save(self.attention.state_dict(), self.get_save_path('attention'))
        torch.save(self.parameter_vector, self.get_save_path('param_vector'))
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            for key in self.projection_matrices.keys():
                self.projection_matrices[key] = torch.load(self.get_save_path('projection-' + key))
            self.attention.load_state_dict(torch.load(self.get_save_path('attention')))
            self.parameter_vector = torch.load(self.get_save_path('param_vector'))

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

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/seq2seq-{self.tag}.pkl'
        self.information_file = f'../results/seq2seq/info-{self.tag}.txt'


class EvaluateSeq2seqModel(AbstractGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.tag = get_now()
        self.model_tag = ['seq2seq']
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainSeq2seqWithSTSBenchmark()
        self.model_tag = [f'seq2seq-{self.tag}']
        self.output_file_name = 'seq2seq.txt'
        self.which_prime_output_to_use_in_testing = self.model.which_prime_output_to_use_in_testing

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        fe_prime_outputs, fd_prime_outputs = [], []
        with torch.no_grad():
            for sentence in batch:
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                # dim: sentence length, input dim
                pooled_fe_prime, pooled_fd_prime = self.model.step({model_name: torch.FloatTensor(embeddings[model_name])
                                                      for model_name in self.model_names})

                fe_prime_outputs.append(pooled_fe_prime.tolist())
                fd_prime_outputs.append(pooled_fd_prime.tolist())

        if self.which_prime_output_to_use_in_testing == 'encoder':
            outputs = fe_prime_outputs
        else:
            outputs = fd_prime_outputs
        return np.array(outputs)


if __name__ == '__main__':
    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        vw = ValueWatcher()
        cls = EvaluateSeq2seqModel()
        trainer = TrainSeq2seqWithSTSBenchmark()

        trainer.model_names = cls.model_names
        trainer.set_tag(cls.tag)

        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            trainer.train_epoch()
            trainer.datasets['train'].reset(with_shuffle=True)
            rets = trainer.inference(mode='dev')
            vw.update(rets['pearson'][0])
            if vw.is_updated():
                trainer.save_model()
                dp.set('best-epoch', vw.epoch)
                dp.set('best-score', vw.max_score)
            dp.set(f'scores', rets)
        print(f'dev best scores: {trainer.get_round_score(dp.get("best-score")[-1]) :.2f}')

        trainer.load_model()
        rets = trainer.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        cls.model = trainer
        rets = cls.single_eval(cls.model_tag[0])
        trainer.append_information_file(rets)
    else:
        trainer = TrainSeq2seqWithSTSBenchmark()
        trainer.train()
