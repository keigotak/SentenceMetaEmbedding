import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from AbstructGetSentenceEmbedding import *


class GetHuggingfaceWordEmbedding:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_ids(self, sent):
        ids_sent = self.tokenizer(sent, return_tensors="pt")
        ids_sent.data['input_ids'] = torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + sent.split(' ') + [self.tokenizer.sep_token])).unsqueeze(0)
        ids_sent.data['token_type_ids'] = torch.zeros_like(ids_sent.data['input_ids'])
        ids_sent.data['attention_mask'] = torch.ones_like(ids_sent.data['input_ids'])
        return ids_sent

    def get_word_embeddings(self, sent1, sent2):
        ids_sent1 = self.get_ids(sent1)
        ids_sent2 = self.get_ids(sent2)
        ids = [ids_sent1, ids_sent2]

        tokens_sent1 = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
        tokens_sent2 = self.tokenizer.convert_ids_to_tokens(ids_sent2.data['input_ids'][0])
        tokens = [tokens_sent1, tokens_sent2]

        emb_sent1 = self.model(**ids_sent1)
        emb_sent2 = self.model(**ids_sent2)
        embedding = [emb_sent1[0].squeeze(0).tolist(), emb_sent2[0].squeeze(0).tolist()]

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def get_word_embedding(self, sentence):
        ids_sent1 = self.get_ids(sentence)
        ids = [ids_sent1]

        tokens_sent1 = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
        tokens = [tokens_sent1]

        emb_sent1 = self.model(**ids_sent1)
        embedding = [emb_sent1[0].squeeze(0).tolist()]

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]


class GetHaggingfaceEmbedding(AbstructGetSentenceEmbedding):
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        super().__init__()
        self.model_names = ['bert-base-uncased']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.mode = 'mean'

    def get_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def batcher(self, params, batch):
        # sentence_embeddings = [self.model(self.tokenizer(sentence, return_tensors="pt")) for sentence in batch]
        sentence_embeddings = []
        for sentence in batch:
            indexes = self.tokenizer(' '.join(sentence), return_tensors="pt")
            sentence_embedding = self.model(**indexes)[0].squeeze(0)
            if self.mode == 'mean':
                sentence_embedding = sentence_embedding[1:-1]
                sentence_embedding = torch.mean(sentence_embedding, dim=0)
            elif self.mode == 'cls':
                sentence_embedding = sentence_embedding[0]
            sentence_embeddings.append(sentence_embedding.tolist())  # get sentence embeddings
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding.tolist()
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    cls = GetHaggingfaceEmbedding()
    for model_name in cls.model_names:
        print(model_name)
        cls.set_model(model_name)
        cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False



'''


'''
