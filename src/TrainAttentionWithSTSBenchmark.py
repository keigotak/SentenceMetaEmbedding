import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from STSDataset import STSDataset
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstructGetSentenceEmbedding import *


class TrainAttentionWithSTSBenchmark:
    def __init__(self):
        self.datasets = {mode: STSDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])
        self.attention_head_num = 1
        if self.attention_head_num > 1:
            self.attention = MultiheadSelfAttentionModel(dimensions=self.total_dim, num_head=self.attention_head_num)
        else:
            self.attention = AttentionModel(dimensions=self.total_dim)
        # self.regressor = nn.Linear(self.total_dim, 1)  # concatenate each sentence
        # self.dropout = nn.Dropout(0.2)
        # params = list(self.attention.parameters()) + list(self.regressor.parameters())
        params = list(self.attention.parameters())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(params, lr=0.001)
        # self.sigmoid = nn.Sigmoid()

        self.attention_output_pooling_method = 'avg'

    def train_epoch(self):
        mode = 'train'
        pbar = tqdm(total=self.datasets[mode].dataset_size)

        ## batch loop
        while not self.datasets[mode].is_batch_end():
            sentences1, sentences2, scores = self.datasets[mode].get_batch()

            ## get vector representation for each embedding
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

            ## get attention output
            running_loss = 0.0
            self.optimizer.zero_grad()
            for embeddings, score in zip(batch_embeddings, scores):
                attention_outputs, attention_weights = [], []
                for i in range(2):  # for input sentences, sentence1 and sentence2
                    feature = torch.cat([torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names], dim=1).unsqueeze(0)
                    if self.attention_head_num > 1:
                        output, attention_weight = self.attention(feature)
                    else:
                        output, attention_weight = self.attention(feature, feature)

                    if self.attention_output_pooling_method == 'avg':
                        pooled_output = torch.mean(output, dim=1)
                    elif self.attention_output_pooling_method == 'concat':
                        pooled_output = torch.cat([out for out in output.squeeze(0)])

                    attention_outputs.append(pooled_output)
                    attention_weights.append(attention_weight)

                # logits = self.regressor(self.dropout(torch.cat(attention_outputs, dim=1)))
                # sigmoid = self.sigmoid(logits)
                # loss = self.criterion(sigmoid, torch.FloatTensor([score]).unsqueeze(0))

                loss = (torch.dot(attention_outputs[0].squeeze(0), attention_outputs[1].squeeze(0)) - score) ** 2
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            pbar.update(self.datasets[mode].batch_size)

            # print(str(self.datasets[mode]) + f' loss: {running_loss}')
        pbar.close()

    def train(self, num_epoch=10):
        for i in range(num_epoch):
            self.train_epoch()
            self.datasets['train'].reset(with_shuffle=True)
            self.inference('dev')

    def inference(self, mode='dev'):
        running_loss = 0.0

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

                # get attention output
                for embeddings, score in zip(batch_embeddings, scores):
                    attention_outputs, attention_weights = [], []
                    for i in range(2):  # for input sentences, sentence1 and sentence2
                        feature = torch.cat([torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names], dim=1).unsqueeze(0)
                        if self.attention_head_num > 1:
                            output, attention_weight = self.attention(feature)
                        else:
                            output, attention_weight = self.attention(feature, feature)

                        if self.attention_output_pooling_method == 'avg':
                            pooled_output = torch.mean(output, dim=1)
                        elif self.attention_output_pooling_method == 'concat':
                            pooled_output = torch.cat([out for out in output.squeeze(0)])

                        attention_outputs.append(pooled_output)
                        attention_weights.append(attention_weight)

                    # logits = self.regressor(self.dropout(torch.cat(attention_outputs, dim=1)))
                    # sigmoid = self.sigmoid(logits)
                    # loss = self.criterion(sigmoid, torch.FloatTensor([score]).unsqueeze(0))

                    loss = (torch.dot(attention_outputs[0].squeeze(0), attention_outputs[1].squeeze(0)) - score) ** 2

                    running_loss += loss.item()

        print(f'[{mode}] ' + str(self.datasets[mode]) + f' loss: {running_loss}')
        self.datasets[mode].reset()

    def save_model(self):
        torch.save(self.attention.state_dict(), '../models/attention.pkl')

    def load_model(self):
        if not os.path.exists('../models/attention.pkl'):
            pass
        else:
            self.attention.load_state_dict(torch.load('../models/attention.pkl'))


class EvaluateAttentionModel(AbstructGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_tag = ['attention']
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainAttentionWithSTSBenchmark()

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        attention_outputs, attention_weights = [], []
        for sentence in batch:
            with torch.no_grad():
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                # get attention output
                feature = torch.cat([torch.FloatTensor(embeddings[model_name]) for model_name in self.model_names], dim=1).unsqueeze(0)
                output, attention_weight = self.model.attention(feature, feature)
                # output, attention_weight = self.model.attention(feature)

                if self.model.attention_output_pooling_method == 'avg':
                    pooled_output = torch.mean(output, dim=1).squeeze(0).tolist()
                elif self.model.attention_output_pooling_method == 'concat':
                    pooled_output = torch.cat([out for out in output.squeeze(0)])

                attention_outputs.append(pooled_output)
                attention_weights.append(attention_weight)

                # logits = self.model.regressor(torch.cat(attention_outputs, dim=1))
                # sigmoid = self.model.sigmoid(logits)
        return np.array(attention_outputs)


if __name__ == '__main__':
    cls = EvaluateAttentionModel()
    trainer = TrainAttentionWithSTSBenchmark()
    for e in range(10):
        trainer.train_epoch()
        trainer.save_model()

        cls.load_model()
        cls.single_eval(cls.model_tag[0])
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False
