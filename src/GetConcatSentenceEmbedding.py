import pickle
from pathlib import Path

import numpy as np
import torch.cuda

from AbstractGetSentenceEmbedding import *

from HelperFunctions import get_now
from GetGloVeEmbedding import GetGloVeSentenceEmbedding


class GetConcatSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        super().__init__()
        self.model_names = ['concat']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.output_file_name = 'concat.txt'
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.tag = get_now()

        self.indexer = None
        self.model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
                            "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl",
                            "glove"]
        self.model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
                            "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl"]

        self.model = {}
        for model_pkl in self.model_pkls:
            if model_pkl == 'glove':
                self.model[model_pkl] = GetGloVeSentenceEmbedding()
                self.model[model_pkl].get_model()
                continue
            with Path(model_pkl).open('rb') as f:
                embedding = pickle.load(f)
            self.model[model_pkl] = embedding

    def get_model(self):
        return self.model

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = []
        for sentence in sentences:
            vectors = []
            for model_type in self.model_pkls:
                if model_type == 'glove':
                    vectors.append(
                        self.model[model_type].batcher(params=None, batch=[sentence.split(' ')]).tolist()[0])
                else:
                    if type(self.model[model_type][sentence]) is not list:
                        vectors.append(self.model[model_type][sentence].tolist())
                    else:
                        vectors.append(self.model[model_type][sentence])
            sentence_embedding = [item for vector in vectors for item in vector] # concatenate embeddings
            sentence_embeddings.append(sentence_embedding.copy())  # get sentence embeddings
            self.embeddings[model_name][sentence] = sentence_embedding
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetConcatSentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''
roberta-large-nli-stsb-mean-tokens,bert-large-nli-stsb-mean-tokens,glove
                                  concat      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.81              79.23              77.66              75.86
                               STS13-all              90.25              89.27              84.08              83.19
                               STS14-all              93.13              91.66              94.01              92.74
                               STS15-all              89.46              89.51              87.69              87.64
                               STS16-all              85.78              86.20              85.51              85.93
                        STSBenchmark-all              84.68              85.44                  -                  -

bert-large-nli-stsb-mean-tokens,glove
                                  concat      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.73              78.96              78.07              75.81
                               STS13-all              89.61              88.82              82.88              82.32
                               STS14-all              92.33              90.93              93.39              92.23
                               STS15-all              88.31              88.24              86.36              86.12
                               STS16-all              83.21              83.95              82.98              83.73
                        STSBenchmark-all              83.70              84.03                  -                  -


roberta-large-nli-stsb-mean-tokens,glove
                                  concat      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.46              77.34              74.89              73.81
                               STS13-all              88.54              87.48              81.87              80.72
                               STS14-all              91.96              90.38              92.86              91.43
                               STS15-all              88.01              88.09              86.01              86.00
                               STS16-all              84.75              85.16              84.42              84.82
                        STSBenchmark-all              82.84              83.57                  -                  -


roberta-large-nli-stsb-mean-tokens,bert-large-nli-stsb-mean-tokens
                                  concat      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.78              79.22              77.63              75.85
                               STS13-all              90.25              89.25              84.06              83.16
                               STS14-all              93.14              91.65              94.02              92.73
                               STS15-all              89.46              89.51              87.68              87.65
                               STS16-all              85.76              86.19              85.48              85.91
                        STSBenchmark-all              84.44              85.39                  -                  -



'''



