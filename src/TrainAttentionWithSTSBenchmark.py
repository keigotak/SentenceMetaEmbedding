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
from HelperFunctions import set_seed, get_now


class TrainAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self):
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.embedding_dims = [self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names]
        self.attention_head_num = 1
        self.dropout_ratio = 0.2
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'avg'
        self.sentence_pooling_method = 'avg'
        if self.source_pooling_method == 'avg':
            self.attention = nn.MultiheadAttention(embed_dim=max(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio)
        elif self.source_pooling_method == 'concat':
            self.attention = nn.MultiheadAttention(embed_dim=sum(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio)
        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.weight_decay = 0.0001
        self.parameters = list(self.attention.parameters())

        super().__init__()

        self.save_model_path = f'../models/attention-{self.tag}.pkl'
        self.information_file = f'../results/attention/info-{self.tag}.txt'


    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores = [], []
        for embeddings, score in zip(batch_embeddings, scores):
            sentence_embeddings, attention_weights, normalized_outputs = [], [], []
            for i in range(2):  # for input sentences, sentence1 and sentence2
                sentence_embedding, attention_weight = self.step({model_name: torch.FloatTensor(embeddings[model_name][i]) for
                    model_name in self.model_names})
                sentence_embeddings.append(sentence_embedding)
                attention_weights.append(attention_weight)

            loss = (torch.dot(sentence_embeddings[0].squeeze(0), sentence_embeddings[1].squeeze(0)) - score) ** 2
            if with_training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
                self.optimizer.step()

            if with_calc_similality:
                sys_score = self.similarity(sentence_embeddings[0].squeeze(0).tolist(), sentence_embeddings[1].squeeze(0).tolist())
                sys_scores.append(sys_score)
                gs_scores.append(score)

            running_loss += loss.item()

        return gs_scores, sys_scores, running_loss

    def step(self, feature):
        if self.source_pooling_method == 'avg':
            pooled_feature = torch.mean(torch.stack([torch.FloatTensor(feature[model_name]) for model_name in self.model_names]),
                                       dim=0).unsqueeze(0).transpose(0, 1)
        elif self.source_pooling_method == 'concat':
            pooled_feature = torch.cat([torch.FloatTensor(feature[model_name]) for model_name in self.model_names],
                                       dim=1).unsqueeze(0).transpose(0, 1)
        attention_output, attention_weight = self.attention(pooled_feature, pooled_feature, pooled_feature)
        attention_output = attention_output.transpose(0, 1)

        if self.sentence_pooling_method == 'avg':
            pooled_sentence_embedding = torch.mean(attention_output, dim=1)
        elif self.sentence_pooling_method == 'concat':
            pooled_sentence_embedding = torch.cat([out for out in attention_output.squeeze(0)])
        elif self.sentence_pooling_method == 'max':
            pooled_sentence_embedding = torch.max(attention_output, dim=1)

        return pooled_sentence_embedding, attention_weight

    def save_model(self):
        torch.save(self.attention.state_dict(), self.save_model_path)
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            self.attention.load_state_dict(torch.load(self.save_model_path))

    def save_information_file(self):
        super().save_information_file()

        with information_file.open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'attention_head_num: {self.attention_head_num}\n')
            f.write(f'attention_dropout_ratio: {self.dropout_ratio}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/attention-{self.tag}.pkl'
        self.information_file = f'../results/attention/info-{self.tag}.txt'


class EvaluateAttentionModel(AbstractGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.tag = get_now()
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainAttentionWithSTSBenchmark()
        self.model.model_names = self.model_names
        self.model_tag = [f'attention-{self.tag}']
        self.output_file_name = 'attention.txt'

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings, attention_weights = [], []
        with torch.no_grad():
            for sentence in batch:
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                # get attention output
                sentence_embedding, attention_weight = self.model.step({model_name: torch.FloatTensor(embeddings[model_name]) for model_name in self.model_names})
                sentence_embeddings.append(sentence_embedding.squeeze(0).tolist())
                attention_weights.append(attention_weight)

        return np.array(sentence_embeddings)


if __name__ == '__main__':
    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        vw = ValueWatcher()
        cls = EvaluateAttentionModel()
        trainer = TrainAttentionWithSTSBenchmark()

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
