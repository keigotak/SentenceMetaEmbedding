import os
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from STSDataset import STSDataset
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstructGetSentenceEmbedding import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed


class TrainAttentionWithSTSBenchmark:
    def __init__(self):
        set_seed(0)

        self.datasets = {mode: STSDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])
        self.attention_head_num = 1
        self.attention_output_pooling_method = 'avg'
        self.attention = nn.MultiheadAttention(embed_dim=self.total_dim, num_heads=self.attention_head_num, dropout=0.2)
        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.parameters = list(self.attention.parameters())
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_ratio)

        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        self.save_model_path = '../models/attention.pkl'

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

            ## get attention output
            running_loss = 0.0
            self.optimizer.zero_grad()
            for embeddings, score in zip(batch_embeddings, scores):
                attention_outputs, attention_weights, normalized_outputs = [], [], []
                for i in range(2):  # for input sentences, sentence1 and sentence2
                    feature = torch.cat([torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names], dim=1).unsqueeze(0).transpose(0, 1)
                    output, attention_weight = self.attention(feature, feature, feature)
                    output = output.transpose(0, 1)

                    if self.attention_output_pooling_method == 'avg':
                        pooled_output = torch.mean(output, dim=1)
                    elif self.attention_output_pooling_method == 'concat':
                        pooled_output = torch.cat([out for out in output.squeeze(0)])
                    elif self.attention_output_pooling_method == 'max':
                        pooled_output = torch.max(output, dim=1)

                    attention_outputs.append(pooled_output)
                    attention_weights.append(attention_weight)
                    normalized_outputs.append(pooled_output / torch.sum(pooled_output))

                loss = (torch.dot(attention_outputs[0].squeeze(0), attention_outputs[1].squeeze(0)) - score) ** 2
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
                self.optimizer.step()

                running_loss += loss.item()

            if with_pbar:
                pbar.update(self.datasets[mode].batch_size)

            # print(str(self.datasets[mode]) + f' loss: {running_loss}')

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

                # get attention output
                for embeddings, score in zip(batch_embeddings, scores):
                    attention_outputs, attention_weights, normalized_outputs = [], [], []
                    for i in range(2):  # for input sentences, sentence1 and sentence2
                        feature = torch.cat([torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names], dim=1).unsqueeze(0).transpose(0, 1)
                        output, attention_weight = self.attention(feature, feature, feature)
                        output = output.transpose(0, 1)

                        if self.attention_output_pooling_method == 'avg':
                            pooled_output = torch.mean(output, dim=1)
                        elif self.attention_output_pooling_method == 'concat':
                            pooled_output = torch.cat([out for out in output.squeeze(0)])
                        elif self.attention_output_pooling_method == 'max':
                            pooled_output = torch.max(output, dim=1)

                        attention_outputs.append(pooled_output)
                        attention_weights.append(attention_weight)
                        normalized_outputs.append(pooled_output / torch.sum(pooled_output))

                    loss = (torch.dot(attention_outputs[0].squeeze(0), attention_outputs[1].squeeze(0)) - score) ** 2

                    running_loss += loss.item()

                    sys_score = self.similarity(attention_outputs[0].squeeze(0).tolist(), attention_outputs[1].squeeze(0).tolist())
                    sys_scores.append(sys_score)
                    gs_scores.append(score)

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

    def save_model(self):
        torch.save(self.attention.state_dict(), self.save_model_path)

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            self.attention.load_state_dict(torch.load(self.save_model_path))

    def get_round_score(self, score):
        return Decimal(str(score * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)


class EvaluateAttentionModel(AbstructGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_tag = ['attention']
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainAttentionWithSTSBenchmark()
        self.output_file_name = 'attention_test.txt'

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        attention_outputs, attention_weights = [], []
        with torch.no_grad():
            for sentence in batch:
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                # get attention output
                feature = torch.cat([torch.FloatTensor(embeddings[model_name]) for model_name in self.model_names], dim=1).unsqueeze(0).transpose(0, 1)
                output, attention_weight = self.model.attention(feature, feature, feature)
                output = output.transpose(0, 1)

                if self.model.attention_output_pooling_method == 'avg':
                    pooled_output = torch.mean(output, dim=1).squeeze(0).tolist()
                elif self.model.attention_output_pooling_method == 'concat':
                    pooled_output = torch.cat([out for out in output.squeeze(0)])
                elif self.model.attention_output_pooling_method == 'max':
                    pooled_output = torch.max(output, dim=1)

                attention_outputs.append(pooled_output)
                attention_weights.append(attention_weight)

        return np.array(attention_outputs)


if __name__ == '__main__':
    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        vw = ValueWatcher()
        cls = EvaluateAttentionModel()
        trainer = TrainAttentionWithSTSBenchmark()
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
        cls.single_eval(cls.model_tag[0])
    else:
        trainer = TrainAttentionWithSTSBenchmark()
        trainer.train()
