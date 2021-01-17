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


class TrainSeq2seqWithSTSBenchmark:
    def __init__(self):
        set_seed(0)

        self.datasets = {mode: STSDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.embedding_dims = {model: self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])

        self.meta_embedding_dim = 300
        self.projection_matrices = {key: torch.randn((self.embedding_dims[key], self.meta_embedding_dim), requires_grad=True) for key in self.model_names}

        self.nonlinear = nn.ReLU()
        self.parameter_vector = torch.randn(self.meta_embedding_dim)

        self.attention_head_num = 1
        self.attention_output_pooling_method = 'avg'
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=self.attention_head_num, dropout=0.2)
        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.weight_decay = 0.001
        self.lambda_e, self.lambda_d = 0.001, 0.001
        self.parameters = list(self.attention.parameters()) + list(self.projection_matrices.values()) + [self.parameter_vector]
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_ratio, weight_decay=self.weight_decay)

        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        self.save_model_path = '../models/seq2seq.pkl'
        self.which_prime_output_to_use_in_testing = 'decoder'

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores = [], []
        for embeddings, score in zip(batch_embeddings, scores):
            attention_outputs, attention_weights, normalized_outputs = [], [], []
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
        fd_prime = torch.einsum('pq, pr->pr', attention_score,
                                target_embeddings[self.model_names[self.get_encoder_model_idx()]])
        fe_prime = torch.einsum('pq, pr->pr', attention_score,
                                target_embeddings[self.model_names[self.get_decoder_model_idx()]])
        return fe_prime, fd_prime

    def get_encoder_model_idx(self):
        return 0

    def get_decoder_model_idx(self):
        return 1

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
            _, _, _ = self.batch_step(batch_embeddings, scores, with_training=True)

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

                ## get attention output
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
        return f'../models/seq2seq-{tag}.pkl'

    def save_model(self):
        for key, value in self.projection_matrices.items():
            torch.save(value, self.get_save_path('projection-' + key))
        torch.save(self.attention.state_dict(), self.get_save_path('attention'))
        torch.save(self.parameter_vector, self.get_save_path('param_vector'))

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            for key in self.projection_matrices.keys():
                self.projection_matrices[key] = torch.load(self.get_save_path('projection-' + key))
            self.attention.load_state_dict(torch.load(self.get_save_path('attention')))
            self.parameter_vector = torch.load(self.get_save_path('param_vector'))


    def get_round_score(self, score):
        return Decimal(str(score * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)


class EvaluateSeq2seqModel(AbstructGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_tag = ['seq2seq']
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainSeq2seqWithSTSBenchmark()
        self.output_file_name = 'seq2seq_test.txt'
        self.output_by = self.model.output_by

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
                fe_prime, fd_prime = self.model.step({model_name: torch.FloatTensor(embeddings[model_name])
                                                      for model_name in self.model_names})

                if self.model.attention_output_pooling_method == 'avg':
                    pooled_fe_prime = torch.mean(fe_prime, dim=0)
                    pooled_fd_prime = torch.mean(fd_prime, dim=0)
                elif self.model.attention_output_pooling_method == 'concat':
                    pooled_fe_prime = torch.cat([out for out in fe_prime])
                    pooled_fd_prime = torch.cat([out for out in fd_prime])
                elif self.model.attention_output_pooling_method == 'max':
                    pooled_fe_prime = torch.max(fe_prime, dim=0)
                    pooled_fd_prime = torch.max(fd_prime, dim=0)

                fe_prime_outputs.append(pooled_fe_prime.tolist())
                fd_prime_outputs.append(pooled_fd_prime.tolist())

        if self.output_by == 'encoder':
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
        trainer = TrainSeq2seqWithSTSBenchmark()
        trainer.train()
