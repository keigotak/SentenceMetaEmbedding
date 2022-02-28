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
from HelperFunctions import set_seed, get_now, get_device

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
        # self.vector_attention = nn.ModuleDict({model: nn.Linear(self.max_sentence_length, 1, bias=False) for model in self.model_names})
        self.normalizer = nn.ModuleDict({model: nn.LayerNorm([self.max_sentence_length, self.meta_embedding_dim]) for model in self.model_names})
        self.activation = nn.GELU()

class TrainVectorAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu', model_names=None):
        self.device = get_device(device)
        # self.model_names = ['stsb-mpnet-base-v2']
        if model_names is not None:
            self.model_names = model_names
        else:
            self.model_names = ['stsb-bert-large', 'stsb-distilbert-base'] # ['stsb-mpnet-base-v2', 'bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens'] # , 'glove', 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
        # self.model_names = ['stsb-roberta-large', 'stsb-mpnet-base-v2', 'stsb-bert-large', 'stsb-distilbert-base']
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
        self.weight_decay = 1e-4
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
            for model_name in model_names:
                self.parameters += list(self.source[model_name].model.parameters())

        if self.loss_mode == 'word':
            self.learning_ratio = 1e-2
        else:
            self.learning_ratio = 1e-2

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
                # sources[model].train()
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

                        sentence_length = words1.shape[1]
                        for i in range(sentence_length):
                            for j in range(sentence_length):
                                if i == j: # 同じ文の同じ単語を比較　同じ単語は近い位置に
                                    # loss.append((1. - self.cos1(words1[:, i], words2[:, j])))
                                    loss1.append(torch.norm(words1[:, i] - words2[:, j], dim=1))
                                else: # 同じ文の違う単語を比較　違う単語は遠くに
                                    # loss.append((1. + self.cos1(words1[:, i], words2[:, j])))
                                    loss2.append((-self.alpha.weight * torch.norm(words1[:, i] - words2[:, j], dim=1)).squeeze(0))

                        # 違う文の比較　
                        # loss.append((1. + self.cos1(sentence_embeddings[0], sentence_embeddings[1])))
                        loss3.append((-self.beta.weight * (torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1))).squeeze(0))

            embedding_loss = [(self.lam.weight * torch.norm(self.va.projection_matrices[model_name].weight.T @ self.va.projection_matrices[model_name].weight - torch.eye(self.va.embedding_dims[model_name], device=self.device))).squeeze(0) for model_name in self.model_names]

            # loss の和
            # if float(self.lam.weight) == 0.0:
            if self.lam == 0.0:
                loss = torch.mean(torch.stack(loss1)) + torch.mean(torch.stack(loss2)) + torch.mean(torch.stack(loss3))
            else:
                # loss = torch.mean(torch.abs(torch.stack(loss))) + torch.mean(torch.stack(embedding_loss))
                # loss = torch.abs(torch.mean(torch.stack(loss1))) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))
                loss = torch.mean(torch.stack(loss1)) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))

        elif self.loss_mode == 'rscore':
            loss = torch.abs(torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1)) - torch.as_tensor(scores, dtype=torch.float, device=self.device)
            # loss = torch.einsum('bq,rs->r', sentence_embeddings[0], sentence_embeddings[1]) - (torch.FloatTensor(scores).to(self.device))
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

        # torch.cuda.empty_cache()

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        # skips = feature

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
                               torch.as_tensor([[[0.0] * self.va.meta_embedding_dim]
                                   * (self.va.max_sentence_length - projected_embeddings[model_name].shape[1])]
                                   * projected_embeddings[model_name].shape[0], dtype=torch.float, device=self.device)),
                    dim=1)
                for model_name in self.model_names}

            # pad_embeddings = {model_name: self.va.normalizer[model_name](pad_embeddings[model_name]) for model_name in
            #            self.model_names}
            word_embeddings = {
                model_name: pad_embeddings[model_name] * self.va.vector_attention[model_name].weight.squeeze(0)
                for model_name in self.model_names}
            # word_embeddings = {
            #     model_name: torch.einsum('ble,t->ble', pad_embeddings[model_name],
            #              self.va.vector_attention[model_name].weight.squeeze(0))
            #     for model_name in self.model_names}
            # word_embeddings = {
            #     model_name: torch.einsum('ble,it->blt',
            #                             (pad_embeddings[model_name], self.va.vector_attention[model_name].weight)
            #                 )
            #     for model_name in self.model_names}
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
        # if self.with_vector_attention:
        #     torch.save(self.vector_attention, self.get_save_path('vector'))
        # torch.save(self.projection_matrices, self.get_save_path('projection_matrices'))
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
            # print('\n'.join([f'{k}: {float(v.weight)}' for k, v in self.va.projection_matrices.items()]))
            # self.va.projection_matrices = nn.ModuleDict({model: nn.Linear(self.embedding_dims[model], self.meta_embedding_dim, bias=False) for model in self.model_names})


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


class EvaluateVectorAttentionModel(AbstractGetSentenceEmbedding):
    def __init__(self, device):
        super().__init__()
        self.device = get_device(device)
        self.tag = get_now()
        # self.model_names = ['stsb-mpnet-base-v2']
        # self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'stsb-mpnet-base-v2', 'bert-large-nli-stsb-mean-tokens'] # 'roberta-large-nli-stsb-mean-tokens', , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
        # self.model_names = ['stsb-bert-large', 'stsb-distilbert-base']# ['stsb-roberta-large', 'stsb-bert-large', 'stsb-distilbert-base'] # , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
        # self.model_names = ['sentence-transformers/stsb-roberta-large', 'sentence-transformers/stsb-bert-large', 'sentence-transformers/stsb-distilbert-base'] # , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
        self.model_names = ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
        # self.model_names = ['bert-large-nli-stsb-mean-tokens', 'use']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = True
        self.model = TrainVectorAttentionWithSTSBenchmark(device=device, model_names=self.model_names)
        self.model_tag = [f'vec_attention-{self.tag}']
        self.output_file_name = 'vec_attention.txt'
        self.model.va.eval()

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings = []
        # print(len(batch))
        with torch.inference_mode():
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
              'lambda': self.model.lam,
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
        trainer.model_names = cls.model_names
        trainer.set_tag(cls.tag)
        print(cls.tag)

        # dev_rets = cls.model.inference(mode='dev')
        # rets = cls.single_eval(cls.model_tag[0])
        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            cls.model.train_epoch()
            cls.model.datasets_stsb['train'].reset(with_shuffle=True)
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
        print(cls.model.information_file)

        cls.model.load_model()
        rets = cls.model.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        rets = cls.single_eval(cls.model_tag[0])
        cls.model.append_information_file([f'es_metrics: {es_metrics}'])
        cls.model.append_information_file(rets['text'])
        print(cls.model.information_file)
    else:

        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = TrainVectorAttentionWithSTSBenchmark(args.device)
        tag = '01052022215941305176' # '01052022200755525916' #  # '01052022220109336377' # '01052022195459718277' # '11222021182523445587' # '10302021131616868619' # '10272021232254714917' # '10252021190301856515' # 10222021201617472745, , 10192021082737054376
        trainer.set_tag(tag)
        cls.set_tag(tag)
        trainer.load_model()
        # rets = trainer.inference(mode='test')
        # print(f'test best scores: ' + ' '.join(rets['prints']))
        cls.model = trainer
        model_tag = cls.model_tag[0]
        if cls.tag != trainer.tag:
            model_tag = f'{model_tag}-{trainer.tag}'
        rets = cls.single_eval(model_tag)

'''
256
      vec_attention-01172022224601439416      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              78.93              77.33              76.48              74.16
                               STS13-all              85.65              85.43              77.43              77.48
                               STS14-all              90.10              89.16              91.20              90.46
                               STS15-all              85.93              85.92              84.05              83.80
                               STS16-all              82.20              83.01              81.94              82.73
                        STSBenchmark-all              83.50              83.99                  -                  -

../results/vec_attention/info-12022021075615227597.txt
stsb-roberta-large: 0.25510174036026
stsb-bert-large: -0.11384961754083633
stsb-distilbert-base: 0.2495175451040268
      vec_attention-12022021075615227597      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.35              78.01              77.53              74.68
                               STS13-all              88.86              88.08              81.89              81.46
                               STS14-all              91.94              90.64              92.69              91.61
                               STS15-all              88.00              88.31              86.04              86.35
                               STS16-all              84.54              84.95              84.25              84.65
                        STSBenchmark-all              84.95              85.60                  -                  -



vec_attention-11142021130216240370-10302021105515772693      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.41              78.45              77.57              75.34
                               STS13-all              89.70              88.57              82.91              82.12
                               STS14-all              92.47              90.80              93.00              91.48
                               STS15-all              88.91              89.22              87.35              87.62
                               STS16-all              86.44              87.03              86.15              86.76
                        STSBenchmark-all              86.44              87.20                  -                  -

        self.meta_embedding_dim = 768
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001
roberta-large-nli-stsb-mean-tokens: -0.016756288707256317
bert-large-nli-stsb-mean-tokens: 0.010068194009363651
vec_attention-10302021084430699531-10272021232254714917      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.92              77.93              76.82              74.54
                               STS13-all              88.88              87.90              81.91              81.32
                               STS14-all              92.05              90.60              92.87              91.60
                               STS15-all              88.63              88.78              86.92              87.01
                               STS16-all              85.21              85.26              84.91              84.96
                        STSBenchmark-all              85.53              86.12                  -                  -
----
10252021190207303220
STSBenchmark-dev pearson: 85.53 spearman: 83.80
dev best scores: 85.53
STSBenchmark-test pearson: 85.02 spearman: 83.84
test best scores: STSBenchmark-test pearson: 85.02 spearman: 83.84

roberta-large-nli-stsb-mean-tokens: -0.021116899326443672
bert-large-nli-stsb-mean-tokens: 0.012464968487620354

-----
        self.meta_embedding_dim = 512
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001

      vec_attention-10232021131936664580      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.73              77.78              76.74              74.56
                               STS13-all              88.16              87.18              80.56              79.58
                               STS14-all              91.96              90.57              92.76              91.52
                               STS15-all              88.12              88.26              86.24              86.32
                               STS16-all              84.85              85.02              84.54              84.72
                        STSBenchmark-all              85.78              86.52                  -                  -
../results/vec_attention/info-10232021132007465076.txt




        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = False
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'rscore' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001
      vec_attention-10222021033234396076      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.44              78.35              77.74              75.23
                               STS13-all              88.70              87.74              81.38              80.62
                               STS14-all              91.59              90.48              92.39              91.47
                               STS15-all              87.82              87.97              85.93              86.05
                               STS16-all              84.76              85.16              84.50              84.90
                        STSBenchmark-all              84.07              84.52                  -                  -

        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001

      vec_attention-10192021082618186513      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.94              78.21              76.91              74.90
                               STS13-all              88.45              87.46              81.24              80.40
                               STS14-all              91.97              90.68              92.80              91.70
                               STS15-all              88.47              88.49              86.62              86.52
                               STS16-all              84.74              84.79              84.43              84.47
                        STSBenchmark-all              85.41              86.06                  -                  -

        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'rscore' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'
        self.vw.threshold = 5

      vec_attention-10192021234059393271      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.74              78.72              78.11              75.63
                               STS13-all              89.22              88.46              82.49              82.07
                               STS14-all              91.80              90.71              92.61              91.78
                               STS15-all              88.16              88.25              86.30              86.37
                               STS16-all              84.28              84.64              84.05              84.40
                        STSBenchmark-all              84.56              85.05                  -                  -

        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'avg' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'rscore' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001
      vec_attention-10202021075207625266      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.98              79.22              78.09              75.87
                               STS13-all              89.88              88.91              83.39              82.60
                               STS14-all              92.58              91.28              93.54              92.45
                               STS15-all              88.66              88.75              86.82              86.77
                               STS16-all              84.95              85.60              84.68              85.35
                        STSBenchmark-all              83.05              83.64                  -                  -

        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'avg' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001
      vec_attention-10212021181735132000      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.70              78.22              76.34              74.78
                               STS13-all              89.44              88.55              82.82              82.13
                               STS14-all              92.44              91.05              93.32              92.11
                               STS15-all              88.59              88.61              86.72              86.63
                               STS16-all              85.29              85.70              84.96              85.38
                        STSBenchmark-all              53.03              47.41                  -                  -

        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = False
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'
        self.vw.threshold = 5

      vec_attention-10222021001112920607      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.53              78.53              77.81              75.43
                               STS13-all              88.70              87.58              80.86              79.87
                               STS14-all              91.97              90.78              92.76              91.72
                               STS15-all              88.68              88.77              87.07              87.11
                               STS16-all              85.80              85.62              85.51              85.33
                        STSBenchmark-all              84.54              84.88                  -                  -



        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'max' # avg, max

        self.gradient_clip = 0.0
        self.weight_decay = 1e-4
        self.with_vector_attention = True
        self.with_projection_matrix = True
        self.parameters = self.va.parameters()
        self.loss_mode = 'word' # word, cos, rscore
        if self.loss_mode == 'word':
            self.learning_ratio = 0.01
        else:
            self.learning_ratio = 0.0001

      vec_attention-10222021201440521085      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.31              77.44              76.24              74.06
                               STS13-all              87.66              86.82              79.49              78.86
                               STS14-all              91.51              90.30              92.32              91.30
                               STS15-all              87.67              87.82              85.72              85.78
                               STS16-all              85.04              85.22              84.76              84.94
                        STSBenchmark-all              85.45              86.01                  -                  -
'''


'''
sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021134320053265      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.25              78.89              78.43              75.72
                               STS13-all              89.45              88.32              82.87              81.83
                               STS14-all              92.37              90.86              93.16              91.85
                               STS15-all              88.03              87.94              86.49              86.30
                               STS16-all              85.86              86.14              85.54              85.85
                        STSBenchmark-all              83.24              83.65                  -                  -

sentence: mean
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021191828857843      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.03              78.42              77.17              75.43
                               STS13-all              88.67              87.56              82.19              81.37
                               STS14-all              91.13              89.80              92.04              90.95
                               STS15-all              87.69              87.65              85.98              85.84
                               STS16-all              83.15              83.64              82.83              83.32
                        STSBenchmark-all              82.26              82.68                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021211831284030      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.49              77.14              75.32              73.90
                               STS13-all              87.48              86.49              80.51              79.43
                               STS14-all              90.95              89.27              91.66              90.11
                               STS15-all              88.03              88.10              86.19              86.17
                               STS16-all              84.94              85.04              84.61              84.71
                        STSBenchmark-all              82.92              83.65                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: with
      vec_attention-06202021232303408276      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.23              78.85              78.43              75.69
                               STS13-all              89.83              88.70              83.19              82.35
                               STS14-all              92.45              90.71              93.28              91.78
                               STS15-all              88.94              88.84              87.26              87.08
                               STS16-all              85.28              85.64              84.97              85.35
                        STSBenchmark-all              83.56              84.21                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06212021034856116616      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.52              78.28              78.16              75.23
                               STS13-all              87.90              86.90              79.88              78.92
                               STS14-all              91.10              90.00              91.77              90.93
                               STS15-all              87.16              87.51              85.19              85.49
                               STS16-all              84.54              85.16              84.30              84.93
                        STSBenchmark-all              84.31              84.66                  -                  -
06212021034933950899

sentence: max
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06212021072055501514      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.76              77.22              76.35              74.35
                               STS13-all              85.56              84.75              76.92              76.10
                               STS14-all              89.96              88.96              90.65              89.92
                               STS15-all              85.77              86.25              83.80              84.12
                               STS16-all              82.12              82.86              81.91              82.64
                        STSBenchmark-all              82.79              82.99                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06212021084720465906      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.49              76.61              75.02              73.73
                               STS13-all              85.13              84.68              76.39              76.05
                               STS14-all              89.42              88.20              89.94              88.87
                               STS15-all              85.83              86.58              83.92              84.59
                               STS16-all              83.26              83.81              83.01              83.57
                        STSBenchmark-all              83.20              83.86                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: with
      vec_attention-06212021101719825925      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.44              78.29              77.73              75.02
                               STS13-all              88.83              87.86              81.56              81.02
                               STS14-all              91.75              90.33              92.57              91.41
                               STS15-all              88.04              88.15              86.13              86.20
                               STS16-all              84.22              84.70              83.94              84.43
                        STSBenchmark-all              84.36              84.85                  -                  -


sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06192021175638248329      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.01              79.59              77.85              76.47
                               STS13-all              90.23              89.26              83.92              83.49
                               STS14-all              92.91              91.30              93.68              92.26
                               STS15-all              89.16              89.03              87.66              87.43
                               STS16-all              85.97              86.06              85.69              85.76
                        STSBenchmark-all              83.08              83.43                  -                  -

sentence: mean
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06192021223542417956      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.12              78.21              77.37              75.02
                               STS13-all              89.02              88.19              82.20              81.58
                               STS14-all              91.66              90.11              92.52              91.21
                               STS15-all              88.02              88.07              86.35              86.30
                               STS16-all              83.91              84.32              83.63              84.03
                        STSBenchmark-all              82.77              82.54                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021002704851388      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.49              76.89              75.12              73.56
                               STS13-all              87.87              86.84              81.95              80.92
                               STS14-all              90.59              89.05              91.26              89.85
                               STS15-all              87.65              87.73              86.21              86.19
                               STS16-all              85.63              85.66              85.33              85.37
                        STSBenchmark-all              82.18              82.46                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: with
      vec_attention-06202021024533979613      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.79              78.93              77.75              75.64
                               STS13-all              90.36              89.24              84.08              83.17
                               STS14-all              92.78              91.28              93.61              92.31
                               STS15-all              89.25              89.24              87.82              87.75
                               STS16-all              85.82              85.91              85.50              85.60
                        STSBenchmark-all              83.24              83.79                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021061709924592      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.66              78.85              78.13              75.79
                               STS13-all              88.24              87.45              80.90              79.87
                               STS14-all              91.34              90.31              91.98              91.18
                               STS15-all              87.91              88.11              86.20              86.31
                               STS16-all              84.91              85.40              84.67              85.17
                        STSBenchmark-all              83.93              84.45                  -                  -

sentence: max
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021085629903832      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.25              76.80              76.07              73.90
                               STS13-all              85.02              84.66              76.27              75.94
                               STS14-all              89.06              88.22              89.71              89.08
                               STS15-all              85.57              85.88              83.64              83.79
                               STS16-all              82.52              83.66              82.38              83.51
                        STSBenchmark-all              82.86              82.98                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: with
      vec_attention-06202021101421921806      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.06              75.77              74.50              72.79
                               STS13-all              84.19              83.57              75.19              74.33
                               STS14-all              88.30              87.28              88.80              87.88
                               STS15-all              84.97              85.70              83.25              83.87
                               STS16-all              83.41              83.90              83.19              83.65
                        STSBenchmark-all              82.09              82.80                  -                  -

06202021113311789761
sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: with
      vec_attention-06202021113257626998      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.56              78.71              77.70              75.43
                               STS13-all              89.60              88.76              83.01              82.35
                               STS14-all              92.31              91.04              93.08              92.03
                               STS15-all              88.53              88.57              86.88              86.86
                               STS16-all              84.99              85.09              84.70              84.81
                        STSBenchmark-all              84.88              85.52                  -                  -


sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021170453671952      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.97              79.13              78.05              75.84
                               STS13-all              89.72              88.88              83.60              82.85
                               STS14-all              92.65              91.32              93.50              92.36
                               STS15-all              88.23              88.24              86.45              86.35
                               STS16-all              85.36              86.12              85.08              85.86
                        STSBenchmark-all              82.79              83.41                  -                  -

sentence: mean
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021201127775047      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.28              78.37              77.52              75.26
                               STS13-all              88.46              87.54              81.85              81.12
                               STS14-all              91.13              90.00              92.08              91.20
                               STS15-all              87.61              87.50              85.64              85.38
                               STS16-all              83.27              83.69              82.99              83.41
                        STSBenchmark-all              81.23              81.70                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021215344837782      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.42              77.38              75.16              74.01
                               STS13-all              87.02              86.17              79.66              78.72
                               STS14-all              90.99              89.53              91.80              90.48
                               STS15-all              87.28              87.51              85.17              85.32
                               STS16-all              84.21              84.49              83.86              84.14
                        STSBenchmark-all              82.14              82.68                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: without
      vec_attention-06212021232355535792      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.19              78.93              78.36              75.65
                               STS13-all              89.75              88.77              82.99              82.28
                               STS14-all              92.63              91.00              93.47              92.10
                               STS15-all              88.96              88.95              87.12              87.02
                               STS16-all              85.17              85.70              84.86              85.42
                        STSBenchmark-all              83.64              84.11                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021021314442802      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.93              77.88              77.60              74.97
                               STS13-all              87.60              86.60              79.49              78.49
                               STS14-all              90.77              89.52              91.45              90.41
                               STS15-all              87.08              87.32              85.15              85.31
                               STS16-all              84.13              84.61              83.90              84.39
                        STSBenchmark-all              83.35              83.83                  -                  -

sentence: max
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021051101140327      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.30              75.82              75.04              73.23
                               STS13-all              84.64              83.62              75.46              74.38
                               STS14-all              88.90              87.70              89.40              88.45
                               STS15-all              85.03              85.51              83.17              83.50
                               STS16-all              81.80              82.42              81.61              82.21
                        STSBenchmark-all              81.85              82.01                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021064032608863      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              75.61              74.84              73.35              72.23
                               STS13-all              84.00              83.53              74.51              74.22
                               STS14-all              88.18              86.67              88.49              87.15
                               STS15-all              85.17              85.81              83.37              83.94
                               STS16-all              82.42              82.88              82.20              82.68
                        STSBenchmark-all              81.27              82.06                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: without
      vec_attention-06222021080517995212      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.07              78.03              77.36              74.83
                               STS13-all              88.51              87.59              81.41              80.99
                               STS14-all              91.37              90.00              92.25              91.10
                               STS15-all              87.91              87.84              85.99              85.86
                               STS16-all              83.60              83.90              83.33              83.64
                        STSBenchmark-all              83.27              83.74                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021170521954600      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.23              79.32              78.24              76.12
                               STS13-all              90.35              89.41              84.13              83.53
                               STS14-all              92.84              91.38              93.69              92.42
                               STS15-all              89.48              89.42              87.90              87.77
                               STS16-all              86.37              86.44              86.08              86.15
                        STSBenchmark-all              83.29              83.33                  -                  -

sentence: mean
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021203849623395      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.36              78.43              77.65              75.26
                               STS13-all              89.15              88.42              82.44              81.92
                               STS14-all              91.90              90.41              92.80              91.56
                               STS15-all              88.04              88.04              86.26              86.13
                               STS16-all              84.00              84.36              83.73              84.08
                        STSBenchmark-all              82.34              82.09                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06212021220755005568      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.79              77.12              75.59              73.86
                               STS13-all              87.57              86.68              81.70              80.85
                               STS14-all              90.37              88.97              91.09              89.81
                               STS15-all              87.13              87.24              85.68              85.70
                               STS16-all              85.68              85.72              85.38              85.42
                        STSBenchmark-all              81.44              81.72                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: without
      vec_attention-06222021001413094838      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.95              79.06              77.98              75.77
                               STS13-all              90.49              89.40              84.36              83.44
                               STS14-all              92.94              91.48              93.79              92.53
                               STS15-all              89.11              89.06              87.59              87.44
                               STS16-all              85.84              86.03              85.52              85.72
                        STSBenchmark-all              82.67              83.09                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021030439462023      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.21              78.53              77.74              75.64
                               STS13-all              87.94              87.22              80.48              79.61
                               STS14-all              90.92              89.78              91.51              90.58
                               STS15-all              87.60              87.78              85.93              86.02
                               STS16-all              84.71              85.08              84.48              84.86
                        STSBenchmark-all              83.04              83.68                  -                  -

sentence: max
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021055104928546      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.07              75.73              74.93              73.06
                               STS13-all              84.24              83.78              74.65              74.17
                               STS14-all              88.33              87.25              88.82              87.96
                               STS15-all              85.13              85.40              83.33              83.44
                               STS16-all              82.35              83.20              82.23              83.07
                        STSBenchmark-all              81.65              81.97                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: word
vector_attention: without
      vec_attention-06222021071122611887      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              75.42              74.22              73.11              71.55
                               STS13-all              82.84              82.40              72.81              72.34
                               STS14-all              86.80              85.51              87.10              85.90
                               STS15-all              83.89              84.54              82.23              82.74
                               STS16-all              82.55              83.12              82.37              82.91
                        STSBenchmark-all              80.13              80.86                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: word
vector_attention: without
      vec_attention-06222021083142490849      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.36              78.47              77.49              75.19
                               STS13-all              89.65              88.90              83.49              83.08
                               STS14-all              92.19              90.99              93.04              92.03
                               STS15-all              88.45              88.39              86.78              86.65
                               STS16-all              84.59              84.66              84.30              84.36
                        STSBenchmark-all              83.66              84.21                  -                  -




sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021134319978848      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.83              79.04              78.05              75.78
                               STS13-all              89.18              88.63              82.51              82.14
                               STS14-all              92.19              91.05              93.05              92.11
                               STS15-all              88.25              88.48              86.42              86.49
                               STS16-all              85.20              85.56              84.93              85.30
                        STSBenchmark-all              81.97              82.83                  -                  -

sentence: mean
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021160750622855      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.66              78.03              76.86              74.67
                               STS13-all              88.29              87.70              81.95              81.51
                               STS14-all              90.84              90.05              91.89              91.32
                               STS15-all              86.88              87.10              84.67              84.71
                               STS16-all              82.18              83.02              81.93              82.75
                        STSBenchmark-all              79.91              80.33                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021172028151182      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.70              77.25              74.54              73.95
                               STS13-all              86.40              86.11              79.30              79.03
                               STS14-all              90.29              89.29              91.13              90.27
                               STS15-all              86.52              86.99              84.27              84.64
                               STS16-all              83.41              83.91              83.08              83.56
                        STSBenchmark-all              80.50              80.85                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: with
      vec_attention-06202021183408258954      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.00              78.98              78.16              75.73
                               STS13-all              89.53              88.68              83.00              82.34
                               STS14-all              92.48              90.99              93.32              92.11
                               STS15-all              88.62              88.75              86.68              86.70
                               STS16-all              84.73              85.45              84.42              85.16
                        STSBenchmark-all              83.44              84.01                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021205503236865      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.69              78.36              78.52              75.38
                               STS13-all              87.68              86.94              80.20              79.41
                               STS14-all              90.81              89.96              91.59              90.95
                               STS15-all              87.34              87.68              85.38              85.63
                               STS16-all              84.03              84.57              83.79              84.33
                        STSBenchmark-all              84.21              84.72                  -                  -

sentence: max
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021231806098332      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.95              77.32              76.55              74.28
                               STS13-all              85.51              84.85              77.29              76.43
                               STS14-all              90.11              89.17              90.94              90.23
                               STS15-all              86.07              86.54              84.08              84.40
                               STS16-all              82.08              82.71              81.89              82.50
                        STSBenchmark-all              82.87              83.08                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06212021002957252286      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.53              75.88              74.26              73.06
                               STS13-all              84.19              84.12              75.83              75.82
                               STS14-all              88.50              87.64              89.12              88.37
                               STS15-all              84.93              85.91              82.84              83.81
                               STS16-all              82.26              83.04              82.03              82.80
                        STSBenchmark-all              83.42              84.00                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: with
      vec_attention-06212021014312157611      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.30              78.24              77.76              75.15
                               STS13-all              88.52              87.73              81.44              80.99
                               STS14-all              91.39              90.21              92.21              91.29
                               STS15-all              87.72              88.04              85.73              86.04
                               STS16-all              83.35              83.98              83.09              83.73
                        STSBenchmark-all              84.02              84.47                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06192021175638314328      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.54              79.23              77.35              75.85
                               STS13-all              90.12              89.31              84.05              83.46
                               STS14-all              92.52              91.38              93.42              92.47
                               STS15-all              89.06              89.08              87.30              87.20
                               STS16-all              85.60              86.17              85.33              85.91
                        STSBenchmark-all              83.24              83.53                  -                  -

sentence: mean
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06192021202800937576      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.83              78.31              77.16              74.94
                               STS13-all              88.46              88.12              81.92              81.61
                               STS14-all              91.33              90.38              92.32              91.59
                               STS15-all              87.26              87.45              85.11              85.09
                               STS16-all              83.02              83.94              82.78              83.70
                        STSBenchmark-all              81.45              81.38                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06192021214310763083      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.85              76.91              74.58              73.43
                               STS13-all              86.89              86.47              80.14              79.75
                               STS14-all              90.47              89.63              91.33              90.64
                               STS15-all              86.64              86.98              84.58              84.76
                               STS16-all              84.09              84.59              83.77              84.27
                        STSBenchmark-all              81.37              81.79                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: with
      vec_attention-06192021225921633180      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.80              79.02              77.87              75.67
                               STS13-all              90.05              89.11              83.65              82.89
                               STS14-all              92.89              91.54              93.78              92.65
                               STS15-all              89.01              89.07              87.30              87.21
                               STS16-all              85.38              85.95              85.09              85.67
                        STSBenchmark-all              82.78              83.18                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021012934330200      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.73              78.73              78.29              75.65
                               STS13-all              88.20              87.47              81.10              80.26
                               STS14-all              91.34              90.40              92.05              91.32
                               STS15-all              88.09              88.25              86.35              86.44
                               STS16-all              84.80              85.28              84.57              85.05
                        STSBenchmark-all              83.80              84.44                  -                  -

sentence: max
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021035849060986      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.68              77.19              76.42              74.11
                               STS13-all              85.46              85.28              77.55              77.39
                               STS14-all              89.76              88.88              90.62              89.89
                               STS15-all              86.37              86.52              84.56              84.54
                               STS16-all              82.93              84.05              82.78              83.88
                        STSBenchmark-all              83.50              83.48                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: with
      vec_attention-06202021051119642172      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.87              75.45              74.42              72.42
                               STS13-all              83.92              83.41              75.48              74.68
                               STS14-all              88.19              87.47              88.83              88.17
                               STS15-all              85.06              85.71              83.27              83.84
                               STS16-all              82.98              83.55              82.76              83.32
                        STSBenchmark-all              81.95              82.58                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: with
      vec_attention-06202021062522787447      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.65              78.58              77.96              75.33
                               STS13-all              89.30              88.56              82.69              82.36
                               STS14-all              91.98              90.89              92.77              91.93
                               STS15-all              88.19              88.31              86.31              86.36
                               STS16-all              84.26              84.66              84.00              84.40
                        STSBenchmark-all              84.43              85.00                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021170323499799      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.79              79.26              77.82              76.03
                               STS13-all              88.92              88.38              82.37              81.89
                               STS14-all              92.01              91.03              92.88              92.12
                               STS15-all              88.53              88.57              86.52              86.42
                               STS16-all              84.98              85.50              84.67              85.22
                        STSBenchmark-all              82.22              82.69                  -                  -

sentence: mean
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021195722675626      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.77              78.13              77.01              74.82
                               STS13-all              88.06              87.43              81.55              81.06
                               STS14-all              90.90              90.03              91.95              91.30
                               STS15-all              86.89              87.11              84.68              84.71
                               STS16-all              82.28              83.06              82.03              82.80
                        STSBenchmark-all              80.03              80.47                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021211406700958      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.59              77.16              74.40              73.85
                               STS13-all              86.45              86.10              79.38              78.97
                               STS14-all              90.32              89.27              91.16              90.25
                               STS15-all              86.44              86.89              84.17              84.51
                               STS16-all              83.44              83.83              83.12              83.50
                        STSBenchmark-all              80.70              81.14                  -                  -

sentence: mean
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: without
      vec_attention-06212021223215866811      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.76              78.86              77.86              75.58
                               STS13-all              89.54              88.71              83.05              82.52
                               STS14-all              92.39              90.88              93.25              92.01
                               STS15-all              88.51              88.61              86.56              86.56
                               STS16-all              84.69              85.36              84.38              85.08
                        STSBenchmark-all              83.18              83.60                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021011104638915      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.62              78.46              78.41              75.51
                               STS13-all              87.50              86.78              79.93              79.10
                               STS14-all              90.69              89.88              91.49              90.83
                               STS15-all              87.35              87.72              85.43              85.73
                               STS16-all              84.08              84.64              83.85              84.42
                        STSBenchmark-all              84.09              84.45                  -                  -

sentence: max
source: mean ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021034805904656      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.71              77.08              76.42              74.12
                               STS13-all              84.87              84.15              76.60              75.47
                               STS14-all              89.57              88.74              90.38              89.76
                               STS15-all              85.78              86.24              83.82              84.11
                               STS16-all              82.01              82.53              81.83              82.33
                        STSBenchmark-all              82.33              82.49                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021050532362388      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.33              75.78              74.03              72.99
                               STS13-all              83.49              83.43              74.71              74.64
                               STS14-all              88.34              87.50              88.96              88.22
                               STS15-all              84.76              85.76              82.72              83.73
                               STS16-all              82.20              82.93              81.99              82.70
                        STSBenchmark-all              82.58              83.19                  -                  -

sentence: max
source: mean ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: without
      vec_attention-06222021062411646117      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.34              78.29              77.79              75.20
                               STS13-all              88.60              87.79              81.48              80.93
                               STS14-all              91.40              90.24              92.23              91.32
                               STS15-all              87.82              88.10              85.86              86.12
                               STS16-all              83.46              84.09              83.20              83.85
                        STSBenchmark-all              83.62              84.14                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021170522008077      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.22              79.46              78.33              76.15
                               STS13-all              89.58              88.83              83.08              82.46
                               STS14-all              92.61              91.42              93.51              92.51
                               STS15-all              89.19              89.19              87.35              87.26
                               STS16-all              85.44              85.93              85.16              85.66
                        STSBenchmark-all              83.02              83.09                  -                  -

sentence: mean
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021195722768013      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.84              78.28              77.19              74.94
                               STS13-all              88.46              88.09              82.05              81.73
                               STS14-all              91.31              90.31              92.30              91.52
                               STS15-all              87.32              87.43              85.17              85.05
                               STS16-all              82.97              83.86              82.73              83.61
                        STSBenchmark-all              81.43              81.35                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06212021211119702528      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.74              76.75              74.45              73.24
                               STS13-all              86.89              86.37              80.19              79.60
                               STS14-all              90.55              89.62              91.40              90.62
                               STS15-all              86.60              86.88              84.53              84.66
                               STS16-all              84.00              84.40              83.67              84.06
                        STSBenchmark-all              81.62              82.07                  -                  -

sentence: mean
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: without
      vec_attention-06212021222505887178      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.67              78.84              77.72              75.46
                               STS13-all              90.06              89.11              83.81              83.04
                               STS14-all              92.87              91.50              93.76              92.60
                               STS15-all              88.94              88.96              87.22              87.10
                               STS16-all              85.28              85.83              84.99              85.55
                        STSBenchmark-all              82.32              82.76                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021005014178922      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.78              78.88              78.31              75.84
                               STS13-all              88.27              87.54              81.36              80.53
                               STS14-all              91.38              90.44              92.11              91.35
                               STS15-all              88.19              88.35              86.48              86.58
                               STS16-all              84.95              85.39              84.72              85.17
                        STSBenchmark-all              77.84              80.32                  -                  -

sentence: max
source: concat ('bert-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021031742339594      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.76              77.29              76.46              74.23
                               STS13-all              85.37              85.22              77.58              77.47
                               STS14-all              89.73              88.89              90.59              89.89
                               STS15-all              86.42              86.55              84.64              84.60
                               STS16-all              83.04              84.13              82.90              83.98
                        STSBenchmark-all              82.87              82.97                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'glove')
loss: rscore
vector_attention: without
      vec_attention-06222021043122269864      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.90              75.55              74.42              72.52
                               STS13-all              83.76              83.32              75.42              74.79
                               STS14-all              88.23              87.48              88.88              88.17
                               STS15-all              85.17              85.78              83.42              83.96
                               STS16-all              83.04              83.60              82.83              83.37
                        STSBenchmark-all              81.13              81.91                  -                  -

sentence: max
source: concat ('roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens')
loss: rscore
vector_attention: without
      vec_attention-06222021054540762902      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.68              78.57              77.98              75.31
                               STS13-all              89.44              88.70              83.02              82.70
                               STS14-all              92.09              90.98              92.89              92.02
                               STS15-all              88.28              88.33              86.43              86.40
                               STS16-all              84.43              84.77              84.17              84.51
                        STSBenchmark-all              83.16              84.23                  -                  -

'''
