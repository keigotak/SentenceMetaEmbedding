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
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now


class TrainVectorAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self):
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat'
        self.sentence_pooling_method = 'avg'
        self.vector_attention = {model: torch.FloatTensor(self.source[model].model.embeddings.word_embeddings.embedding_dim).uniform_().requires_grad_(False) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model] / sum(self.vector_attention[model]) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model].requires_grad_(True) for model in self.model_names}
        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.weight_decay = 0.01
        self.parameters = list(self.vector_attention.values())

        super().__init__()

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
            pooled_sentence_embedding, _ = torch.max(torch.stack(pooled_word_embeddings), dim=0)

        return pooled_sentence_embedding

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

    def set_tag(self, tag):
        self.tag = tag
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'


class EvaluateVectorAttentionModel(AbstractGetSentenceEmbedding):
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
