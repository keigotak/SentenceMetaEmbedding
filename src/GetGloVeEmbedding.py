from pathlib import Path

import numpy as np
import torch

import gensim

from AbstractGetSentenceEmbedding import *


class GloVeModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.embeddings = {}
        with Path('../models/glove.42B.300d.txt').open('r') as f:
             for line in f.readlines():
                  items = line.strip().split(' ')
                  self.embeddings[items[0]] = list(map(float, items[1:]))
        self.embedding_dim = 300
        self.vocab = set(self.embeddings.keys())

    def get_word_embedding(self, batch_sentences):
        rets = []
        for sentence in batch_sentences:
            embedding = [self.embeddings[word] if word in self.embeddings.keys() else [0.0] * self.embedding_dim for word in sentence]
            rets.append(embedding)
        return rets

    def get_word_embeddings(self, sent1, sent2):
        ids, tokens, embedding = [], [], []
        for sent in [sent1, sent2]:
            rets = self.get_word_embedding([sent.split(' ')])
            # ids.extend(rets['ids'])
            # tokens.extend(rets['tokens'])
            embedding.extend(rets)

        return {'ids': None, 'tokens': None, 'embeddings': embedding}

    def state_dict(self):
        return self.embedding.state_dict()


class GetGloVeSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_names = ['glove']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_save_embeddings = True
        self.sentence_pooling_method = 'avg'

    def get_model(self):
        if self.model is None:
            self.model = GloVeModel()
        return self.model

    def batcher(self, params, batch):
        sentence_embeddings = []
        word_embeddings = self.model.get_word_embedding(batch)   # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, word_embeddings):
            if self.sentence_pooling_method == 'avg':
                sentence_embedding = torch.mean(torch.FloatTensor(sentence_embedding).requires_grad_(False), dim=0)
            elif self.sentence_pooling_method == 'max':
                sentence_embedding, _ = torch.max(torch.FloatTensor(sentence_embedding).requires_grad_(False), dim=0)
            sentence_embeddings.append(sentence_embedding.tolist())
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding

        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetGloVeSentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''
                                glove-avg      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              45.34              50.80              45.00              50.03
                               STS13-all              47.44              49.26              43.73              44.72
                               STS14-all              49.41              51.67              49.12              51.45
                               STS15-all              55.64              58.81              51.95              55.40
                               STS16-all              44.11              52.28              44.01              52.10
                        STSBenchmark-all              63.16              60.61                  -                  -

                                glove-max      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              50.53              54.41              50.61              53.49
                               STS13-all              51.04              51.06              40.08              40.38
                               STS14-all              55.96              56.14              55.78              55.92
                               STS15-all              60.60              63.14              58.06              61.24
                               STS16-all              58.44              63.47              58.60              63.54
                        STSBenchmark-all              68.23              67.56                  -                  -

'''
