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
from AbstructGetSentenceEmbedding import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now


class TrainVectorAttentionWithSTSBenchmark:
    def __init__(self):
        set_seed(0)

        self.datasets = {mode: STSDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'avg'
        self.sentence_pooling_method = 'avg'
        self.vector_attention = {model: torch.FloatTensor(self.source[model].model.embeddings.word_embeddings.embedding_dim).uniform_().requires_grad_(False) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model] / sum(self.vector_attention[model]) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model].requires_grad_(True) for model in self.model_names}
        self.learning_ratio = 0.1
        self.gradient_clip = 0.2
        self.weight_decay = 0.0001
        self.parameters = list(self.vector_attention.values())
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_ratio, weight_decay=self.weight_decay)

        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        self.tag = get_now()
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'


    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores = [], []
        for embeddings, score in zip(batch_embeddings, scores):
            sentence_embeddings = []
            for i in range(2):  # for input sentences, sentence1 and sentence2
                pooled_sentence_embedding = self.step({model_name: torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names})
                sentence_embeddings.append(pooled_sentence_embedding)

            loss = (torch.dot(*sentence_embeddings) - score) ** 2
            if with_training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
                self.optimizer.step()

            if with_calc_similality:
                sys_score = self.similarity(sentence_embeddings[0].tolist(), sentence_embeddings[1].tolist())
                sys_scores.append(sys_score)
                gs_scores.append(score)

            running_loss += loss.item()

        return gs_scores, sys_scores, running_loss

    def step(self, feature):
        word_embeddings = {
            model_name: torch.einsum('pq, r->pr', feature[model_name], self.vector_attention[model_name]) for
            model_name in self.model_names}

        # multiple source embedding and vector attention
        if self.source_pooling_method == 'avg':
            pooled_word_embeddings = []
            for j in range(word_embeddings[self.model_names[0]].shape[0]):
                pooled_word_embedding = []
                for model_name in self.model_names:
                    pooled_word_embedding.append(word_embeddings[model_name][j])
                pooled_word_embeddings.append(torch.mean(torch.stack(pooled_word_embedding), dim=0))
        elif self.source_pooling_method == 'concat':
            pooled_word_embeddings = []
            for j in range(word_embeddings[self.model_names[0]].shape[0]):
                pooled_word_embedding = []
                for model_name in self.model_names:
                    pooled_word_embedding.append(word_embeddings[model_name][j])
                pooled_word_embeddings.append(torch.cat(pooled_word_embedding, dim=0))

        # aggregate word embeddings to sentence embedding
        if self.sentence_pooling_method == 'avg':
            pooled_sentence_embedding = torch.mean(torch.stack(pooled_word_embeddings), dim=0)
        elif self.sentence_pooling_method == 'max':
            pooled_sentence_embedding = torch.max(torch.stack(pooled_word_embeddings), dim=0)

        return pooled_sentence_embedding

    def train_epoch(self, with_pbar=False):
        mode = 'train'
        if with_pbar:
            pbar = tqdm(total=self.datasets[mode].dataset_size)

        ## batch loop
        while not self.datasets[mode].is_batch_end():
            sentences1, sentences2, scores = self.datasets[mode].get_batch()

            ## get vector representation for each embedding and batch data
            batch_embeddings = []
            with torch.no_grad():
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size

            _, _, _ =self.batch_step(batch_embeddings, scores, with_training=True)

            if with_pbar:
                pbar.update(self.datasets[mode].batch_size)

        if with_pbar:
            pbar.close()

    def train(self, num_epoch=10):
        vw = ValueWatcher()
        for i in range(num_epoch):
            self.train_epoch()
            self.datasets['train'].reset(with_shuffle=True)
            rets = self.inference('dev')

            vw.update(rets['pearson'][0])
            if vw.is_max():
                trainer.save_model()

    def inference(self, mode='dev'):
        running_loss = 0.0
        results = {}
        sys_scores, gs_scores = [], []

        # batch loop
        while not self.datasets[mode].is_batch_end():
            sentences1, sentences2, scores = self.datasets[mode].get_batch()

            # get vector representation for each embedding
            batch_embeddings = []
            with torch.no_grad():
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size

                gs, sys, loss = self.batch_step(batch_embeddings, scores, with_calc_similality=True)
                sys_scores.extend(sys)
                gs_scores.extend(gs)
                running_loss += loss

        results = {'pearson': pearsonr(sys_scores, gs_scores),
                   'spearman': spearmanr(sys_scores, gs_scores),
                   'nsamples': len(sys_scores)}

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"][0]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"][0]) :.2f}']
        results['prints'] = print_contents

        print(f'[{mode}] ' + str(self.datasets[mode]) + f' loss: {running_loss}')
        print(' '.join(print_contents))

        self.datasets[mode].reset()

        return results

    def get_save_path(self, tag):
        return f'../models/vec_attention-{self.tag}-{tag}.pkl'

    def save_model(self):
        torch.save(self.vector_attention, self.get_save_path('vector'))
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.get_save_path('vector')):
            pass
        else:
            self.vector_attention = torch.load(self.get_save_path('vector'))

    def get_round_score(self, score):
        return Decimal(str(score * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    def append_information_file(self, results):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with information_file.open('a') as f:
            for print_all_content in print_all_contents:
                print(' '.join(['{: >40}'] + ['{: >18}'] * (len(print_all_header) - 1)).format(*print_all_content),
                      file=f)

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with information_file.open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'


class EvaluateVectorAttentionModel(AbstructGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.tag = get_now()
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainVectorAttentionWithSTSBenchmark()
        self.model.model_names = self.model_names
        self.model_tag = [f'vec_attention-{self.tag}']
        self.output_file_name = 'vec_attention.txt'

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings = []
        with torch.no_grad():
            for sentence in batch:
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                sentence_embedding = self.model.step({model_name: torch.FloatTensor(embeddings[model_name]) for model_name in
                               self.model_names})
                sentence_embeddings.append(sentence_embedding.tolist())

        return np.array(sentence_embeddings)


if __name__ == '__main__':
    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        vw = ValueWatcher()
        cls = EvaluateVectorAttentionModel()
        trainer = TrainVectorAttentionWithSTSBenchmark()

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
        trainer = TrainAttentionWithSTSBenchmark()
        trainer.train()
