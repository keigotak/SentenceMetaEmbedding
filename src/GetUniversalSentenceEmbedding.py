import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from AbstractGetSentenceEmbedding import *


class GetUniversalSentenceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        self.model_names = ['use']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_save_embeddings = True

    def get_model(self):
        if self.model is None:
            self.model = hub.Module("../models/universal-sentence-encoder-large_3")
        return self.model

    def get_params(self):
        params_senteval = {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
        return params_senteval

    def batcher(self, params, batch):
        if self.model is None:
            self.get_model()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        with tf.Session(config=config) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentence_embeddings = session.run(self.model(sentences)).tolist()  # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetUniversalSentenceEmbedding()
    for model_name in cls.model_names:
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False


'''


'''
