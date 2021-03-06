import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from AbstractGetSentenceEmbedding import *


class GetMetaWordEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        super().__init__()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.model_dims = {'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.source = {model: GetSentenceBertWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.model_dims[model] for model in self.model_names])
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat'   # avg, concat
        self.sentence_pooling_method = 'avg'    # avg, max

    def get_model(self):
        return self.source

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentence_embeddings = []
        for sentence in batch:
            word_embeddings = {}
            for model_name in self.model_names:
                embedding = self.source[model_name].get_word_embedding(' '.join(sentence))
                word_embeddings[model_name] = embedding['embeddings'][0]

            if self.source_pooling_method == 'avg':
                pooled_word_embeddings = []
                for j in range(len(word_embeddings[self.model_names[0]][1:-1])):
                    pooled_word_embedding = []
                    for model_name in self.model_names:
                        pooled_word_embedding.append(torch.FloatTensor(word_embeddings[model_name][j]).requires_grad_(False))
                    pooled_word_embeddings.append(torch.mean(torch.stack(pooled_word_embedding), dim=0))
            elif self.source_pooling_method == 'concat':
                pooled_word_embeddings = []
                for j in range(len(word_embeddings[self.model_names[0]][1:-1])):
                    pooled_word_embedding = []
                    for model_name in self.model_names:
                        pooled_word_embedding.append(torch.FloatTensor(word_embeddings[model_name][j]).requires_grad_(False))
                    pooled_word_embeddings.append(torch.cat(pooled_word_embedding, dim=0))

            if self.sentence_pooling_method == 'avg':
                pooled_sentence_embedding = torch.mean(torch.stack(pooled_word_embeddings), dim=0)
            elif self.sentence_pooling_method == 'max':
                pooled_sentence_embedding, _ = torch.max(torch.stack(pooled_word_embeddings), dim=0)

            sentence_embeddings.append(pooled_sentence_embedding.tolist())

        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetMetaWordEmbedding()
    cls.set_model(f"{'_'.join(cls.model_names)}_{cls.source_pooling_method}_{cls.sentence_pooling_method}")
    rets = cls.single_eval(cls.model_name)
    if cls.with_reset_output_file:
        cls.with_reset_output_file = False


'''


'''
