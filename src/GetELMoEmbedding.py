from pathlib import Path

import numpy as np
import torch

from elmoformanylangs import Embedder

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.elmo import Elmo, batch_to_ids

from AbstructGetSentenceEmbedding import *


class ElmoModel:
    def __init__(self, device='cpu', elmo_with="allennlp"):
        self.device = device
        self.elmo_with = elmo_with
        self.elmo_embeddings = None
        if self.elmo_with == "allennlp":
            if Path('../models/sentence_embeddings_elmo_allennlp.pkl').exists():
                with Path('../models/sentence_embeddings_elmo_allennlp.pkl').open('wb') as f:
                    self.elmo_embeddings = pickle.load(f)
            root_path = Path('../models/elmo')
            option_file = root_path / 'config.json'
            weight_file = root_path / 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            self.num_output_representations = 2
            self.embedding = Elmo(option_file, weight_file, num_output_representations=self.num_output_representations).to(device)
            self.embedding_dim = 1024 * self.num_output_representations
            self.vocab = Vocabulary()
        else:
            root_path = Path('../models/elmo')
            self.embedding = Embedder(root_path)
            if str(device) == 'cpu':
                self.embedding.use_cuda = False
                self.embedding.model.use_cuda = False
                self.embedding.model.encoder.use_cuda = False
                self.embedding.model.token_embedder.use_cuda = False
                self.embedding.model.to(device)
            else:
                self.embedding.use_cuda = True
                self.embedding.model.use_cuda = True
                self.embedding.model.encoder.use_cuda = True
                self.embedding.model.token_embedder.use_cuda = True
                self.embedding.model.to(device)
            self.embedding_dim = self.embedding.config['encoder']['projection_dim'] * 2

    def get_word_embedding(self, batch_sentences):
        rets = []
        if self.elmo_with == "allennlp":
            for sentence in batch_sentences:
                indexes = batch_to_ids(sentence)
                embedding = self.embedding(torch.LongTensor(indexes).to(self.device))
                rets.append(torch.mean(torch.mean(torch.cat((embedding['elmo_representations'][0], embedding['elmo_representations'][1]), dim=2), dim=0), dim=0).tolist())
        else:
            rets = self.embedding.sents2elmo(batch_sentences)
            rets = torch.tensor(rets)
        rets = np.array(rets)
        return rets

    def state_dict(self):
        return self.embedding.state_dict()


class GetELMoSentenceEmbedding(AbstructGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['elmo-allennlp']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_save_embeddings = True

    def get_model(self):
        if self.model is None:
            self.model = ElmoModel(elmo_with='allennlp')
        return self.model

    def batcher(self, params, batch):
        sentence_embeddings = self.model.get_word_embedding(batch)   # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding
        return sentence_embeddings


if __name__ == '__main__':
    cls = GetELMoSentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''


'''
