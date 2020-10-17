from gcca.gcca import GCCA
import numpy as np
from AbstructGetSentenceEmbedding import *


class GetGCCASentenceEmbedding(AbstructGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['gcca']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.output_file_name = 'results.gcca.201017.txt'
        self.with_reset_output_file = False
        self.with_save_embeddings = False

        self.input_model_names = ['bert-large-nli-stsb-mean-tokens',
                                  'distilbert-base-nli-stsb-mean-tokens',
                                  'roberta-base-nli-stsb-mean-tokens',
                                  'roberta-large-nli-stsb-mean-tokens']
        self.sentence_embeddings = {}
        for model_name in self.input_model_names:
            with open(f'../models/sentence_embeddings_{model_name}.pkl', 'rb') as f:
                self.sentence_embeddings[model_name] = pickle.load(f)

    def get_model(self):
        if self.model is None:
            self.model = GCCA()
            self.model.load_params("../models/sts_gcca.h5")
        return self.model

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words

        # gcca_embeddings = []
        alist, blist, clist, dlist = [], [], [], []
        for sentence in sentences:
            alist.append(self.sentence_embeddings[self.input_model_names[0]][sentence].tolist())
            blist.append(self.sentence_embeddings[self.input_model_names[1]][sentence].tolist())
            clist.append(self.sentence_embeddings[self.input_model_names[2]][sentence].tolist())
            dlist.append(self.sentence_embeddings[self.input_model_names[3]][sentence].tolist())

        alist, blist, clist, dlist = np.array(alist), np.array(blist), np.array(clist), np.array(dlist)

        gcca_embeddings = self.model.transform(alist, blist, clist, dlist)

        return gcca_embeddings[0]


if __name__ == '__main__':
    cls = GetGCCASentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False



'''


'''
