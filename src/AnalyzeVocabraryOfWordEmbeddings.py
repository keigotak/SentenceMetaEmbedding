import pickle
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AbstractGetSentenceEmbedding import *


class AnalyzeVocabraryOfWordEmbeddings(AbstractGetSentenceEmbedding):
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['bert-base-uncased', 'roberta-base']
        self.source = {model: GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.tokens = {model: set() for model in self.model_names}

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
                self.tokens[model_name] |= set(embedding['tokens'][0])

            pooled_sentence_embedding = torch.mean(torch.FloatTensor(word_embeddings[model_name]), dim=0)
            sentence_embeddings.append(pooled_sentence_embedding.tolist())

        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = AnalyzeVocabraryOfWordEmbeddings()
    cls.output_file_name = 'xxx.txt'
    cls.set_model(f"{'_'.join(cls.model_names)}")
    rets = cls.single_eval(cls.model_name)
    if cls.with_reset_output_file:
        cls.with_reset_output_file = False


'''


'''
