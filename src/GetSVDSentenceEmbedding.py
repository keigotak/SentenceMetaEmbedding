import numpy as np
from AbstractGetSentenceEmbedding import *


class GetSVDSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        super().__init__()
        self.model_names = ['svd']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.output_file_name = 'svd.txt'
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.tag = '04072021215506466248'

        self.indexer = None

    def get_model(self):
        if self.tag is None:
            with open('../models/sts_svd.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('../models/sts_svd_sentence_indexer.pkl', 'rb') as f:
                self.indexer = pickle.load(f)
        else:
            with open(f'../models/svd_{self.tag}.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(f'../models/svd_{self.tag}_sentence_indexer.pkl', 'rb') as f:
                self.indexer = pickle.load(f)

        return self.model

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = []
        for sentence in sentences:
            indexes = self.indexer[sentence]
            sentence_embedding = self.model[indexes].tolist()
            sentence_embeddings.append(sentence_embedding)  # get sentence embeddings
            self.embeddings[model_name][sentence] = sentence_embedding
        return np.array(sentence_embeddings)



if __name__ == '__main__':
    cls = GetSVDSentenceEmbedding()
    for model_name in cls.model_names:
        rets = cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False
        print('\n'.join(rets['text']))



'''
                                     svd      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.78              79.21              77.63              75.84
                               STS13-all              90.25              89.25              84.06              83.16
                               STS14-all              93.14              91.65              94.02              92.73
                               STS15-all              89.46              89.51              87.68              87.65
                               STS16-all              85.76              86.19              85.48              85.91
                        STSBenchmark-all              84.91              85.77                  -                  -

'''
