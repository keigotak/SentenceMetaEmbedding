import pickle
import os
import numpy as np
from AbstractGetSentenceEmbedding import *


class GetMetaSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        super().__init__()
        self.sentence_pooling_method = 'max'
        self.model_names = ['concat'] # 'avg',
        # self.src_model_names = ['bert-large-nli-stsb-mean-tokens', 'use']
        self.src_model_names = ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
        # self.src_model_names = ['stsb-roberta-large', 'stsb-bert-large', 'stsb-distilbert-base']
        # self.src_model_names = ['stsb-bert-large', 'stsb-distilbert-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.model = {}
        self.with_save_embeddings = True

    def get_model(self):
        for src_model in self.src_model_names:
            # with open(f'../models/sentence_embeddings_{src_model}_max.pkl', 'rb') as f:
            #     self.model[src_model] = pickle.load(f)
            with open(f'./{src_model}_avg.pt', 'rb') as f:
                self.model[src_model] = torch.load(f)

        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        max_dimention = 1024
        multiple_sentence_embeddings = []

        for sentence in sentences:
            if '�' in sentence:
                sentence = sentence.replace('� ', '')
            if 'o ̯ reĝ' in sentence:
                sentence = sentence.replace('o ̯ reĝ', '')
            sentence_embeddings = []
            for src_model in self.src_model_names:
                embedding = torch.as_tensor(self.model[src_model][sentence]['embeddings'], dtype=torch.float)
                # if self.model_name == 'avg':
                #     if len(embedding) < max_dimention:
                #         embedding += [0.0] * (max_dimention - len(embedding))
                sentence_embeddings.append(embedding)

            if self.model_name == 'avg':
                sentence_embeddings = torch.mean(torch.stack(sentence_embeddings), dim=0)
                sentence_embeddings = sentence_embeddings.squeeze(0)
            elif self.model_name == 'concat':
                sentence_embeddings = torch.cat(sentence_embeddings, dim=2)
                sentence_embeddings = sentence_embeddings.squeeze(0)

            if self.sentence_pooling_method == 'avg':
                sentence_embeddings = torch.mean(sentence_embeddings, dim=0)
                self.embeddings[model_name][sentence] = sentence_embeddings
                multiple_sentence_embeddings.append(sentence_embeddings.tolist())
            elif self.sentence_pooling_method == 'max':
                sentence_embeddings, _ = torch.max(sentence_embeddings, dim=0)
                self.embeddings[model_name][sentence] = sentence_embeddings
                multiple_sentence_embeddings.append(sentence_embeddings.tolist())



                # concat_embedding = sum(sentence_embeddings, [])
                # for i, sentence_embedding in zip(range(len(sentence_embeddings)), sentence_embeddings):
                #     if i == 0:
                #         concat_embedding = sentence_embedding
                #     else:
                #         concat_embedding += sentence_embedding
                # self.embeddings[model_name][sentence] = np.array(concat_embedding)
                # multiple_sentence_embeddings.append(concat_embedding)

        return np.array(multiple_sentence_embeddings)


if __name__ == '__main__':
    cls = GetMetaSentenceEmbedding()
    for model_name in cls.model_names:
        cls.set_model(model_name)
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''
['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']
                                    avg      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.52              78.82              77.40              75.44
                               STS13-all              89.89              88.90              83.54              82.68
                               STS14-all              92.85              91.41              93.73              92.49
                               STS15-all              89.09              89.16              87.39              87.36
                               STS16-all              85.55              86.05              85.25              85.75
                        STSBenchmark-all              84.47              85.07                  -                  -
                                  concat      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.68              78.90              77.53              75.51
                               STS13-all              90.11              89.12              83.89              83.07
                               STS14-all              92.98              91.51              93.86              92.59
                               STS15-all              89.25              89.31              87.49              87.46
                               STS16-all              85.74              86.17              85.45              85.87
                        STSBenchmark-all              84.84              85.50                  -                  -


['stsb-bert-large', 'stsb-distilbert-base']
                                     avg      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.59              78.54              77.61              74.93
                               STS13-all              90.36              89.49              84.21              83.63
                               STS14-all              92.72              91.25              93.74              92.52
                               STS15-all              88.47              88.45              86.63              86.50
                               STS16-all              83.97              84.63              83.69              84.33
                        STSBenchmark-all              84.21              84.47                  -                  -
                                  concat      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.92              78.77              77.99              75.18
                               STS13-all              90.38              89.55              84.07              83.56
                               STS14-all              92.86              91.39              93.88              92.66
                               STS15-all              88.64              88.59              86.81              86.63
                               STS16-all              84.27              84.88              84.00              84.60
                        STSBenchmark-all              84.69              85.14                  -                  -



['stsb-roberta-large', 'stsb-bert-large', 'stsb-distilbert-base']
                                     avg      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.63              79.11              77.50              75.76
                               STS13-all              90.03              89.09              83.69              82.89
                               STS14-all              93.01              91.55              93.89              92.64
                               STS15-all              89.27              89.32              87.57              87.52
                               STS16-all              85.57              86.05              85.28              85.75
                        STSBenchmark-all              84.60              85.18                  -                  -
                                  concat      pearson-wmean     spearman-wmean       pearson-mean      spearman-mean
                               STS12-all              80.78              79.22              77.63              75.85
                               STS13-all              90.25              89.25              84.06              83.16
                               STS14-all              93.14              91.65              94.02              92.73
                               STS15-all              89.46              89.51              87.68              87.65
                               STS16-all              85.76              86.19              85.48              85.91
                        STSBenchmark-all              84.44              85.39                  -                  -





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
