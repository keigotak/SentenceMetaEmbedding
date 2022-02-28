from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

import dask
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from GetGloVeEmbedding import GloVeModel
from GetUniversalSentenceEmbedding import GetUniversalSentenceEmbedding
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device, get_metrics

class VectorAttention(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        set_seed(0)
        self.model_names = model_names
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024, 'glove': 300, 'use': 512, 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens': 768, 'stsb-mpnet-base-v2': 768, 'sentence-transformers/stsb-bert-large': 1024, 'sentence-transformers/stsb-roberta-large': 1024, 'sentence-transformers/stsb-distilbert-base': 768, 'stsb-bert-large': 1024, 'stsb-roberta-large': 1024, 'stsb-distilbert-base': 768}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.meta_embedding_dim = 1024
        self.projection_matrices = nn.ModuleDict({model: nn.Linear(self.embedding_dims[model], self.meta_embedding_dim, bias=False) for model in self.model_names})
        self.max_sentence_length = 128
        self.vector_attention = nn.ModuleDict({model: nn.Linear(1, 1, bias=False) for model in self.model_names})
        self.normalizer = nn.ModuleDict({model: nn.LayerNorm([self.max_sentence_length, self.meta_embedding_dim]) for model in self.model_names})
        self.activation = nn.GELU()

class TrainVectorAttentionWithSTS(AbstractTrainer):
    def __init__(self, device='cpu', model_names=None):
        self.device = get_device(device)
        if model_names is not None:
            self.model_names = model_names
        else:
            self.model_names = ['stsb-bert-large', 'stsb-distilbert-base', 'stsb-mpnet-base-v2'] # ['stsb-mpnet-base-v2', 'bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens'] # , 'glove', 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
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

        self.gradient_clip = 0.0
        self.weight_decay = 1e-2
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.with_train_coefficients = True
        self.with_train_model = False
        self.loss_mode = 'rscore' # word, cos, rscore

        self.alpha = nn.Linear(1, 1, bias=False).to(device=self.device)
        self.lam = nn.Linear(1, 1, bias=False).to(device=self.device)
        self.beta = nn.Linear(1, 1, bias=False).to(device=self.device)
        if self.with_train_coefficients:
            self.parameters = list(self.va.parameters()) + list(self.alpha.parameters()) + list(self.lam.parameters()) + list(self.beta.parameters())
        else:
            nn.init.constant_(self.alpha.weight, 1.0)
            nn.init.constant_(self.lam.weight, 1.0)
            nn.init.constant_(self.beta.weight, 1.0)
            # self.alpha = 0.01
            # self.lam = 0.01
            # self.beta = 0.01
            self.parameters = list(self.va.parameters())

        if self.with_train_model:
            for model_name in self.model_names:
                self.source[model_name].model.train()
                self.parameters += list(self.source[model_name].model.parameters())

        if self.loss_mode == 'word':
            self.learning_ratio = 7e-5
        else:
            self.learning_ratio = 7e-5

        super().__init__()

        self.batch_size = 512
        self.datasets_stsb['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'
        self.vw.threshold = 5

    def get_source_embeddings(self):
        sources = {}
        for model in self.model_names:
            if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'stsb-mpnet-base-v2', 'stsb-roberta-large', 'stsb-mpnet-base-v2', 'stsb-bert-large', 'stsb-distilbert-base']):
                sources[model] = GetSentenceBertWordEmbedding(model, device=self.device)
            elif model == 'glove':
                sources[model] = GloVeModel()
            elif model == 'use':
                sources[model] = GetUniversalSentenceEmbedding()
            else:
                sources[model] = GetHuggingfaceWordEmbedding(model, device=self.device)
        return sources

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            if not self.va.training:
                self.va.train()
            self.optimizer.zero_grad(set_to_none=True)
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
            loss1, loss2, loss3 = [], [], []
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
                        distance = torch.cdist(words1.contiguous(), words2.contiguous())
                        cosine_similarity = self.cosine_similarity(words1, words2)

                        placeholder = torch.eye(distance.shape[1]).unsqueeze(0).repeat(distance.shape[0], 1, 1)
                        loss1.append(distance[placeholder == 1.])
                        loss2.append(-(self.alpha.weight * distance[placeholder == 0.]).squeeze())

                        # sentence_length = words1.shape[1]
                        # for i in range(sentence_length):
                        #     for j in range(sentence_length):
                        #         if i == j: # 同じ文の同じ単語を比較　同じ単語は近い位置に
                        #             # loss.append((1. - self.cos1(words1[:, i], words2[:, j])))
                        #             loss1.append(torch.norm(words1[:, i] - words2[:, j], dim=1))
                        #         else: # 同じ文の違う単語を比較　違う単語は遠くに
                        #             # loss.append((1. + self.cos1(words1[:, i], words2[:, j])))
                        #             loss2.append((-self.alpha.weight * torch.norm(words1[:, i] - words2[:, j], dim=1)).squeeze(0))

                        # 違う文の比較　
                        # loss.append((1. + self.cos1(sentence_embeddings[0], sentence_embeddings[1])))
                        # loss3.append((-self.beta.weight * (torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1))).squeeze(0))
                        loss3.append(-(self.beta.weight * torch.dist(sentence_embeddings[0].unsqueeze(1), sentence_embeddings[1].unsqueeze(1))).squeeze())

            embedding_loss = [(self.lam.weight * torch.norm(self.va.projection_matrices[model_name].weight.T @ self.va.projection_matrices[model_name].weight - torch.eye(self.va.embedding_dims[model_name], device=self.device))).squeeze() for model_name in self.model_names]

            # loss の和
            if self.lam == 0.0:
                loss = torch.mean(torch.stack(loss1)) + torch.mean(torch.stack(loss2)) + torch.mean(torch.stack(loss3))
            else:
                # loss = torch.mean(torch.abs(torch.stack(loss))) + torch.mean(torch.stack(embedding_loss))
                # loss = torch.abs(torch.mean(torch.stack(loss1))) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))
                # loss = torch.abs(torch.mean(torch.stack(loss1))) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))
                loss = torch.abs(torch.mean(torch.cat(loss1))) + torch.abs(torch.mean(torch.cat(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))

        elif self.loss_mode == 'rscore':
            loss = torch.abs(torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1)) - torch.as_tensor(scores, dtype=torch.float, device=self.device)
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
            del loss

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        if self.with_projection_matrix:
            projected_embeddings = {
                model_name: self.va.projection_matrices[model_name](feature[model_name])
                for model_name in self.model_names
            }
        else:
            projected_embeddings = feature

        if self.with_vector_attention:
            pad_embeddings = {
                model_name: torch.cat((projected_embeddings[model_name],
                               torch.as_tensor([[[0.0] * self.va.meta_embedding_dim]
                                   * (self.va.max_sentence_length - projected_embeddings[model_name].shape[1])]
                                   * projected_embeddings[model_name].shape[0], dtype=torch.float, device=self.device)),
                    dim=1)
                for model_name in self.model_names}

            word_embeddings = {
                model_name: pad_embeddings[model_name] * self.va.vector_attention[model_name].weight.squeeze(0)
                for model_name in self.model_names}
            word_embeddings = {
                model_name: torch.narrow(word_embeddings[model_name], dim=1, start=0, length=feature[model_name].shape[1])
                for model_name in self.model_names}
        else:
            word_embeddings = projected_embeddings

        # multiple source embedding and vector attention
        if self.source_pooling_method == 'avg':
            pooled_word_embeddings = torch.mean(torch.stack([word_embeddings[model_name] for model_name in self.model_names]), dim=0)
        elif self.source_pooling_method == 'concat':
            try:
                pooled_word_embeddings = torch.cat([word_embeddings[model_name] for model_name in self.model_names], dim=2)
            except:
                print("shape error")

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
        self.save_information_file()
        for model_name in self.model_names:
            if self.source[model_name].with_embedding_updating:
                with Path(f'./{model_name}.pt').open('wb') as f:
                    torch.save(self.source[model_name].word_embeddings, f)

    def load_model(self):
        if not os.path.exists(self.get_save_path('va')):
            pass
        else:
            self.va = torch.load(self.get_save_path('va'))
            self.va.to(self.device)
            print('\n'.join([f'{k}: {float(v.weight)}' for k, v in self.va.vector_attention.items()]))


    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'tag: {self.tag}\n')
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'meta_embedding_dim: {self.va.meta_embedding_dim}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'alpha: {self.alpha.weight}\n')
            f.write(f'lambda: {self.lam.weight}\n')
            f.write(f'beta: {self.beta.weight}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'with_vector_attention: {self.with_vector_attention}\n')
            f.write(f'with_projection_matrix: {self.with_projection_matrix}\n')
            f.write(f'with_train_coefficients: {self.with_train_coefficients}\n')
            f.write(f'with_train_model: {self.with_train_model}\n')
            f.write(f'dataset_type: {self.dataset_type}\n')
            f.write(f'loss_mode: {self.loss_mode}\n')
            f.write(f'weights: \n')
            f.write('\n'.join([f'{k}: {float(v.weight)}' for k, v in self.va.vector_attention.items()]))
            f.write('\n')
            f.write(str(self.optimizer))
            f.write('\n')


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
        self.datasets_stsb['train'].batch_size = self.batch_size


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
        model = TrainVectorAttentionWithSTS(args.device)

        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            model.train_epoch()
            model.datasets_stsb['train'].reset(with_shuffle=True)
            rets = model.inference(mode='dev')
            if es_metrics == 'pearson':
                vw.update(rets[es_metrics])
            else:
                vw.update(rets[es_metrics])
            if vw.is_updated():
                model.save_model()
                dp.set('best-epoch', vw.epoch)
                dp.set('best-score', vw.max_score)
            dp.set(f'scores', rets)
        print(f'dev best scores: {model.get_round_score(dp.get("best-score")[-1]) :.2f}')
        print(model.information_file)
        model.append_information_file([f'es_metrics: {es_metrics}'])
        model.append_information_file([f'dev best scores: {model.get_round_score(dp.get("best-score")[-1])}'])

        model.load_model()
        rets = model.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        model.append_information_file([f'test best scores: {" ".join(rets["prints"])}'])
        for mode in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            dev_rets = model.inference_sts(mode=mode)
            metrics = get_metrics(dev_rets['sys_scores'], dev_rets['gold_scores'], dev_rets['tags'])
            dev_rets['prints'] = dev_rets['prints'] + [f'{k}: {v}' for k, v in metrics.items()]
            model.append_information_file(dev_rets['prints'])
        # rets = cls.single_eval(model_tag[0])
        print(model.information_file)
    else:
        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = TrainVectorAttentionWithSTSBenchmark(args.device)
        tag = '01052022215941305176' # '01052022200755525916' #  # '01052022220109336377' # '01052022195459718277' # '11222021182523445587' # '10302021131616868619' # '10272021232254714917' # '10252021190301856515' # 10222021201617472745, , 10192021082737054376
        trainer.set_tag(tag)
        cls.set_tag(tag)
        trainer.load_model()
        # rets = trainer.inference(mode='test')
        # print(f'test best scores: ' + ' '.join(rets['prints']))
        model = trainer
        model_tag = model_tag[0]
        if cls.tag != trainer.tag:
            model_tag = f'{model_tag}-{trainer.tag}'
        rets = cls.single_eval(model_tag)

'''
rscore
../results/vec_attention/info-02202022093409204239.txt

word
info-02202022152804196696.txt

'''

