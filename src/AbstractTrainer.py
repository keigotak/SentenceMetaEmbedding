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
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now


class AbstractTrainer:
    def __init__(self):
        set_seed(0)
        self.datasets = {mode: STSDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_ratio, weight_decay=self.weight_decay)
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        self.tag = get_now()
        self.cos0 = nn.CosineSimilarity(dim=0)
        self.cos1 = nn.CosineSimilarity(dim=1)

    def batch_step(self, batch_embeddings, scores, tokens, with_training=False, with_calc_similality=False):
        raise NotImplementedError

    def step(self, feature):
        raise NotImplementedError

    def train_epoch(self, with_pbar=False):
        mode = 'train'
        if with_pbar:
            pbar = tqdm(total=self.datasets[mode].dataset_size)

        ## batch loop
        while not self.datasets[mode].is_batch_end():
            sentences1, sentences2, scores = self.datasets[mode].get_batch()

            ## get vector representation for each embedding and batch data
            batch_embeddings = []
            batch_tokens = []
            with torch.no_grad():
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    tokens = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                        tokens[model_name] = rets['tokens']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size
                    batch_tokens.append(tokens)

            ## get attention output
            _, _, _ = self.batch_step(batch_embeddings, scores, with_training=True)

            if with_pbar:
                pbar.update(self.datasets[mode].batch_size)

            # print(str(self.datasets[mode]) + f' loss: {running_loss}')

        if with_pbar:
            pbar.close()

    def train(self, num_epoch=100):
        vw = ValueWatcher()
        for i in range(num_epoch):
            self.train_epoch()
            if self.datasets['train'].batch_mode == 'fixed' and self.datasets['train'].current >= self.datasets['train'].dataset_size:
                self.datasets['train'].reset(with_shuffle=True)
            elif self.datasets['train'].batch_mode == 'full':
                self.datasets['train'].reset(with_shuffle=True)
            rets = self.inference('dev')

            vw.update(rets['pearson'][0])
            if vw.is_max():
                trainer.save_model()

    def inference(self, mode='dev'):
        running_loss = 0.0
        results = {}
        pearsonrs, spearmanrs = [], []

        # batch loop
        while not self.datasets[mode].is_batch_end():
            sys_scores, gs_scores = [], []
            sentences1, sentences2, scores = self.datasets[mode].get_batch()

            # get vector representation for each embedding
            batch_embeddings = []
            batch_tokens = []
            with torch.no_grad():
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = {}
                    for model_name in self.model_names:
                        rets = self.source[model_name].get_word_embeddings(sent1, sent2)
                        if False:
                            print('\t'.join([' '.join(items) for items in rets['tokens']]))
                        embeddings[model_name] = rets['embeddings']
                    batch_embeddings.append(embeddings)  ## batch, embedding type, sentence source, sentence length, hidden size
                    batch_tokens.append([sent1.split(' '), sent2.split(' ')])

                # get attention output
                gs, sys, loss = self.batch_step(batch_embeddings, scores, with_calc_similality=True)
                sys_scores.extend(sys)
                gs_scores.extend(gs)
                running_loss += loss
            pearsonrs.append(pearsonr(sys_scores, gs_scores)[0])
            spearmanrs.append(spearmanr(sys_scores, gs_scores)[0])

        avg_pearsonr = np.average(pearsonrs)
        avg_spearmanr = np.average(spearmanrs)

        results = {'pearson': avg_pearsonr,
                   'spearman': avg_spearmanr,
                   'nsamples': len(sys_scores),
                   'dev_loss': running_loss}

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"]) :.2f}']
        results['prints'] = print_contents

        print(f'[{mode}] ' + str(self.datasets[mode]) + f' loss: {running_loss}')
        print(' '.join(print_contents))

        self.datasets[mode].reset()

        return results

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def get_round_score(self, score):
        return Decimal(str(score * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    def append_information_file(self, results):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with information_file.open('a') as f:
            f.write('\n'.join(results))

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

    def set_tag(self, tag):
        raise NotImplementedError

