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
from GetGloVeEmbedding import GloVeModel
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device

class VectorAttention(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.model_names = model_names
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024, 'glove': 300}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.meta_embedding_dim = 512
        self.projection_matrices = nn.ModuleDict({model: nn.Linear(self.embedding_dims[model], self.meta_embedding_dim, bias=False) for model in self.model_names})
        self.max_sentence_length = 128
        self.vector_attention = nn.ModuleDict({model: nn.Linear(self.max_sentence_length, 1, bias=False) for model in self.model_names})
        self.normalizer = nn.ModuleDict({model: nn.LayerNorm(self.model_dims[model]) for model in self.model_names})
        self.activation = nn.GELU()

class TrainVectorAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu'):
        self.device = get_device(device)
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove']
        self.va = VectorAttention(model_names=self.model_names).to(self.device)
        self.va.train()

        self.model_dims = self.va.model_dims
        self.source = self.get_source_embeddings()
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.total_dim = sum(self.embedding_dims.values())

        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'avg' # avg, max

        self.learning_ratio = 0.005
        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'
        self.vw.threshold = 5

    def get_source_embeddings(self):
        sources = {}
        for model in self.model_names:
            if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']):
                sources[model] = GetSentenceBertWordEmbedding(model, device=self.device)
            elif model == 'glove':
                sources[model] = GloVeModel()
            else:
                source[model] = GetHuggingfaceWordEmbedding(model, device=self.device)
        return sources

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            if not self.va.training:
                self.va.train()
            self.optimizer.zero_grad()
        else:
            if self.va.training:
                self.va.eval()

        gs_scores, sys_scores, losses = [], [], []
        padded_sequences, _ = self.modify_batch_embeddings_to_easy_to_compute(batch_embeddings)

        sentence_embeddings, word_embeddings = [], []
        for i in range(2):  # for input sentences, sentence1 and sentence2
            pooled_sentence_embedding, word_embedding = self.step({model_name: padded_sequences[model_name][i] for model_name in self.model_names})
            sentence_embeddings.append(pooled_sentence_embedding)
            word_embeddings.append(word_embedding)

        if self.loss_mode == 'cos':
            loss = 1. - self.cos1(sentence_embeddings[0], sentence_embeddings[1])
            loss = torch.mean(loss)
        elif self.loss_mode == 'word':
            # dimensions: sentence, source, words, hidden
            loss = []
            for word_embedding in word_embeddings:
                combinations = set()

                for iw1 in range(len(self.model_names)):
                    for iw2 in range(len(self.model_names)):
                        if iw1 == iw2 or (iw1, iw2) in combinations:
                            continue
                        combinations.add((iw1, iw2))
                        combinations.add((iw2, iw1))

                        words1 = word_embedding[self.model_names[iw1]]
                        words2 = word_embedding[self.model_names[iw2]]

                        sentence_length = words1.shape[1]
                        for i in range(sentence_length):
                            for j in range(sentence_length):
                                if i == j: # 同じ文の同じ単語を比較　同じ単語は近い位置に
                                    # loss.append((1. - self.cos1(words1[:, i], words2[:, j])))
                                    loss.append(torch.norm(words1[:, i] - words2[:, j], dim=1))
                                else: # 同じ文の違う単語を比較　違う単語は遠くに
                                    # loss.append((1. + self.cos1(words1[:, i], words2[:, j])))
                                    loss.append(-torch.norm(words1[:, i] - words2[:, j], dim=1))

                        # 違う文の比較　
                        # loss.append((1. + self.cos1(sentence_embeddings[0], sentence_embeddings[1])))
                        loss.append(-torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1))

            # loss の和
            loss = torch.mean(torch.abs(torch.stack(loss)))
        elif self.loss_mode == 'rscore':
            loss = torch.abs(torch.einsum('bq,bq->b', sentence_embeddings[0], sentence_embeddings[1]) - (torch.FloatTensor(scores).to(self.device)))
            loss = torch.mean(loss)

        if with_calc_similality:
            sys_score = [self.similarity(e1, e2) for e1, e2 in zip(sentence_embeddings[0].tolist(), sentence_embeddings[1].tolist())]
            sys_scores.extend(sys_score)
            gs_scores.extend(scores)

        running_loss += loss.item()
        print(running_loss)

        if with_training:
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
            self.optimizer.step()

        torch.cuda.empty_cache()

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        skips = feature
        feature = {model_name: self.va.normalizer[model_name](feature[model_name]) for model_name in self.model_names}

        if self.with_projection_matrix:
            projected_embeddings = {
                model_name: self.va.projection_matrices[model_name](feature[model_name])
                for model_name in self.model_names
            }
        else:
            projected_embeddings = feature

        if self.with_vector_attention:
            # word_embeddings = {
            #     model_name: self.vector_attention[model_name](projected_embeddings[model_name])
            #     for model_name in self.model_names}
            pad_embeddings = {
                model_name: torch.cat((projected_embeddings[model_name],
                               torch.FloatTensor([[[0.0] * self.va.meta_embedding_dim]
                                   * (self.va.max_sentence_length - projected_embeddings[model_name].shape[1])]
                                   * projected_embeddings[model_name].shape[0]).to(self.device)),
                    dim=1)
                for model_name in self.model_names}
            word_embeddings = {
                model_name: torch.einsum('ble,bl->ble',
                                        (pad_embeddings[model_name], self.va.vector_attention[model_name].weight)
                            )
                for model_name in self.model_names}
            word_embeddings = {
                model_name: torch.narrow(word_embeddings[model_name], dim=1, start=0, length=feature[model_name].shape[1])
                for model_name in self.model_names}
        else:
            word_embeddings = projected_embeddings

        # word_embeddings = {model_name: self.va.activation(word_embeddings[model_name]) for model_name in self.model_names}
        # word_embeddings = {model_name: word_embeddings[model_name] + skips[model_name] for model_name in self.model_names}

        # multiple source embedding and vector attention
        if self.source_pooling_method == 'avg':
            pooled_word_embeddings = torch.mean(torch.stack([word_embeddings[model_name] for model_name in self.model_names]), dim=0)
        elif self.source_pooling_method == 'concat':
            pooled_word_embeddings = torch.cat([word_embeddings[model_name] for model_name in self.model_names], dim=2)

        # aggregate word embeddings to sentence embedding
        if self.sentence_pooling_method == 'avg':
            pooled_sentence_embedding = torch.mean(pooled_word_embeddings, dim=1)
        elif self.sentence_pooling_method == 'max':
            pooled_sentence_embedding, _ = torch.max(pooled_word_embeddings, dim=1)

        return pooled_sentence_embedding, word_embeddings

    def get_save_path(self, tag):
        return f'../models/vec_attention-{self.tag}-{tag}.pkl'

    def save_model(self):
        torch.save(self.va, self.get_save_path('va'))
        # if self.with_vector_attention:
        #     torch.save(self.vector_attention, self.get_save_path('vector'))
        # torch.save(self.projection_matrices, self.get_save_path('projection_matrices'))
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.get_save_path('va')):
            pass
        else:
            self.va = torch.load(self.get_save_path('va'))
            self.va.to(self.device)

        # if self.with_vector_attention:
        #     if not os.path.exists(self.get_save_path('vector')):
        #         pass
        #     else:
        #         self.vector_attention = torch.load(self.get_save_path('vector'))
        #
        # if not os.path.exists(self.get_save_path('projection_matrices')):
        #     pass
        # else:
        #     self.projection_matrices = torch.load(self.get_save_path('projection_matrices'))


    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'with_vector_attention: {self.with_vector_attention}\n')
            f.write(f'with_projection_matrix: {self.with_projection_matrix}\n')
            f.write(f'loss_mode: {self.loss_mode}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'

    def update_hyper_parameters(self, hyper_params):
        self.va = VectorAttention(model_names=self.model_names).to(self.device)
        self.projection_matrices = self.va.projection_matrices
        self.vector_attention = self.va.vector_attention

        self.source_pooling_method = hyper_params['source_pooling_method']
        self.sentence_pooling_method = hyper_params['sentence_pooling_method']

        self.learning_ratio = hyper_params['learning_ratio']
        self.gradient_clip = hyper_params['gradient_clip']
        self.weight_decay = hyper_params['weight_decay']
        self.with_vector_attention = hyper_params['with_vector_attention']
        self.parameters = self.va.parameters()
        self.loss_mode = hyper_params['loss_mode']

        super().__init__()

        self.batch_size = hyper_params['batch_size']
        self.datasets['train'].batch_size = self.batch_size


class EvaluateVectorAttentionModel(AbstractGetSentenceEmbedding):
    def __init__(self, device):
        super().__init__()
        self.device = get_device(device)
        self.tag = get_now()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainVectorAttentionWithSTSBenchmark(device=device)
        self.model.model_names = self.model_names
        self.model_tag = [f'vec_attention-{self.tag}']
        self.output_file_name = 'vec_attention.txt'
        self.model.va.eval()

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings = []
        with torch.no_grad():
            padded_sequences, padding_masks = self.modify_batch_sentences_for_senteval(batch)
            # get attention output
            sentence_embeddings, attention_weights = self.model.step({model_name: padded_sequences[model_name] for model_name in self.model_names},
                                                                   padding_mask={model_name: padding_masks[model_name] for model_name in self.model_names})

        return np.array(sentence_embeddings.tolist())

    def set_tag(self, tag):
        self.model_tag[0] = f'{self.model_tag[0]}-{tag}'
        self.tag = tag

    def save_summary_writer(self, rets):
        sw = SummaryWriter('runs/VectorAttention')

        hp = {'source': ','.join(self.model.model_names),
              'tokenization_mode': self.model.tokenization_mode,
              'subword_pooling_method': self.model.subword_pooling_method,
              'source_pooling_method': self.model.source_pooling_method,
              'sentence_pooling_method': self.model.sentence_pooling_method,
              'learning_ratio': self.model.learning_ratio,
              'gradient_clip': self.model.gradient_clip,
              'weight_decay': self.model.weight_decay,
              'batch_size': self.model.batch_size,
              'with_vector_attention': self.model.with_vector_attention,
              'loss_mode': self.model.loss_mode,
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
        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = cls.model # TrainVectorAttentionWithSTSBenchmark(args.device)
        # trainer.model_names = cls.model_names
        # trainer.set_tag(cls.tag)
        print(cls.tag)

        dev_rets = cls.model.inference(mode='dev')
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
        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = TrainVectorAttentionWithSTSBenchmark(args.device)
        tag = '03062021183728375245'
        trainer.set_tag(tag)
        cls.set_tag(tag)
        trainer.load_model()
        rets = trainer.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        cls.model = trainer
        model_tag = cls.model_tag[0]
        if cls.tag != trainer.tag:
            model_tag = f'{model_tag}-{trainer.tag}'
        rets = cls.single_eval(model_tag)
