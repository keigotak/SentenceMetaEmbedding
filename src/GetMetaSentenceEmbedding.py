import pickle
import os
import numpy as np
from AbstractGetSentenceEmbedding import *


class GetMetaSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['avg', 'concat']
        # self.src_model_names = ['bert-large-nli-stsb-mean-tokens', 'use']
        # self.src_model_names = ['stsb-mpnet-base-v2', 'bert-large-nli-stsb-mean-tokens',
        #                     'roberta-large-nli-stsb-mean-tokens']
        self.src_model_names = ['stsb-roberta-large', 'stsb-bert-large', 'stsb-distilbert-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.model = {}
        self.with_save_embeddings = True

    def get_model(self):
        for src_model in self.src_model_names:
            with open(f'../models/sentence_embeddings_{src_model}.pkl', 'rb') as f:
                self.model[src_model] = pickle.load(f)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        max_dimention = 1024
        multiple_sentence_embeddings = []

        for sentence in sentences:
            sentence_embeddings = []
            for src_model in self.src_model_names:
                embedding = self.model[src_model][sentence]
                if type(embedding) is not list:
                    embedding = embedding.tolist()
                if self.model_name == 'avg':
                    if len(embedding) < max_dimention:
                        embedding += [0.0] * (max_dimention - len(embedding))
                sentence_embeddings.append(embedding)

            if self.model_name == 'avg':
                self.embeddings[model_name][sentence] = np.array(sentence_embeddings).mean(axis=0)
                multiple_sentence_embeddings.append(np.array(sentence_embeddings).mean(axis=0).tolist())
            elif self.model_name == 'concat':
                concat_embedding = sum(sentence_embeddings, [])
                # for i, sentence_embedding in zip(range(len(sentence_embeddings)), sentence_embeddings):
                #     if i == 0:
                #         concat_embedding = sentence_embedding
                #     else:
                #         concat_embedding += sentence_embedding
                self.embeddings[model_name][sentence] = np.array(concat_embedding)
                multiple_sentence_embeddings.append(concat_embedding)

        return np.array(multiple_sentence_embeddings)


if __name__ == '__main__':
    cls = GetMetaSentenceEmbedding()
    for model_name in cls.model_names:
        cls.set_model(model_name)
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''
                                     avg      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.96              79.01              77.88              75.53
                               STS13-all              90.33              89.33              84.21              83.39
                               STS14-all              93.10              91.66              93.99              92.76
                               STS15-all              89.12              89.16              87.45              87.39
                               STS16-all              85.71              86.20              85.41              85.90
                        STSBenchmark-all              85.46              85.88                  -                  -
                                  concat      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              81.01              79.15              77.84              75.67
                               STS13-all              90.52              89.51              84.48              83.60
                               STS14-all              93.23              91.76              94.11              92.85
                               STS15-all              89.44              89.45              87.71              87.62
                               STS16-all              86.04              86.46              85.74              86.17
                        STSBenchmark-all              85.49              85.84                  -                  -


'''
