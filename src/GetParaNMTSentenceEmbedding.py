import _pickle as cPickle
from pathlib import Path
from para_nmt_50m.eval import get_model
from AbstructGetSentenceEmbedding import *


class GetParaNMTSentenceEmbedding(AbstructGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        super().__init__()
        self.vocab = None
        self.model_names = ['para-nmt-50M']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_save_embeddings = True

    def get_model(self):
        if self.model is None:
            # with Path('../models/para-nmt-50m/bilstmavg-4096-40.pickle').open('rb') as f:
            #     weights = cPickle.load(f, encoding='latin1')
            #     self.model = models(weights, params)
            self.model, self.vocab = get_model()
        return self.model

    def batcher(self, params, batch):
        if self.model is None:
            self.get_model()

        sentences = [[self.vocab[word] if word in self.vocab.keys() else self.vocab["unk"] for word in sent] for sent in batch]  # To reconstruct sentence from list of words
        sentence_indexes, masks = self.model.prepare_data(sentences)
        sentence_embeddings = self.model.predict(sentence_indexes)
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetParaNMTSentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''


'''
