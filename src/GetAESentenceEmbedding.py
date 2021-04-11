from pathlib import Path
import numpy as np
from AbstractGetSentenceEmbedding import *
from HelperFunctions import get_now


class GetAESentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['ae']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.output_file_name = 'ae.txt'
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.tag = 'bertl+use+1024' # 'bertl+robertal+512'

        self.sent_to_id, self.id_to_sent = {}, {}
        with Path('./ae/sentence_embeddings_indexer.txt').open('r') as f:
            texts = f.readlines()
        for text in texts:
            sentID, sent = text.strip().split('\t')
            self.sent_to_id[sent] = sentID
            self.id_to_sent[sentID] = sent
        self.model = {}

    def get_model(self):
        if self.tag is None:
            assert 'Set model tag'
        else:
            with open(f'./ae/AutoencodedMetaEmbedding-master/aeme/{self.tag}', 'r') as f:
                texts = f.readlines()
            for text in texts:
                items = text.strip().split(' ')
                self.model[items[0]] = list(map(float, items[1:]))
        return self.model

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = []
        for sentence in sentences:
            indexes = self.sent_to_id[sentence]
            sentence_embedding = self.model[indexes]
            sentence_embeddings.append(sentence_embedding)  # get sentence embeddings
            self.embeddings[model_name][sentence] = sentence_embedding
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetAESentenceEmbedding()
    for model_name in cls.model_names:
        rets = cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False
        print('\n'.join(rets['text']))



'''


'''
