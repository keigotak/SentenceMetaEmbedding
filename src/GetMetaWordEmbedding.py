import pickle
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AbstractGetSentenceEmbedding import *


class GetMetaWordEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.total_dim = sum([self.source[model].model.embeddings.word_embeddings.embedding_dim for model in self.model_names])
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'avg'
        self.sentence_pooling_method = 'avg'

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
