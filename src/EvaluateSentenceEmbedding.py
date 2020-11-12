import pickle
import numpy as np

from AbstructGetSentenceEmbedding import *


class EvaluateSentenceEmbedding(AbstructGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        super().__init__()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens',
                            'roberta-base-nli-stsb-mean-tokens',
                            'bert-large-nli-stsb-mean-tokens',
                            'distilbert-base-nli-stsb-mean-tokens',
                            'use']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = True
        self.with_save_embeddings = False

    def get_model(self):
        with open(f'../models/sentence_embeddings_{self.model_name}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = [self.model[sentence] for sentence in sentences]  # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = EvaluateSentenceEmbedding()
    for model_name in cls.model_names:
        print(model_name)
        cls.set_model(model_name)
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False



'''


'''
