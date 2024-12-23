import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from AbstractGetSentenceEmbedding import *
from HelperFunctions import get_device


class GetHuggingfaceWordEmbedding:
    def __init__(self, model_name, device='cpu'):
        self.device = get_device(device)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenization_mode = 'subword'
        self.subword_pooling_method = 'avg'

    def get_ids(self, sent):
        ids_sent = self.tokenizer(sent, return_tensors="pt")
        if self.tokenization_mode == 'original':
            ids_sent.data['input_ids'] = torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + sent.split(' ') + [self.tokenizer.sep_token])).unsqueeze(0)
            ids_sent.data['token_type_ids'] = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
            ids_sent.data['attention_mask'] = torch.ones_like(ids_sent.data['input_ids'])
        return ids_sent

    def process_subword(self, sent, embeddings):
        words = sent.split(" ")
        subword_list, subword_tokens = [], []
        sequence_index = 0

        # detect subwords
        for i, word in enumerate(words):
            if i != 0 and self.model_name in {'roberta-base', 'roberta-large'}:
                word = ' ' + word
            token = self.tokenizer(word, return_tensors="pt")
            subword_list.extend([sequence_index] * len(token.data['input_ids'][0][1:-1]))
            subword_tokens.append(self.tokenizer.convert_ids_to_tokens(token.data['input_ids'][0][1:-1]))
            sequence_index += 1
        subword_list = [-1] + subword_list + [sequence_index]

        # aggregate subwords embeddings
        subword_aggregated_embeddings = []
        for i in range(-1, sequence_index + 1):
            subword_positions = [j for j, x in enumerate(subword_list) if x == i]
            # if the word are subworded
            if len(subword_positions) > 1:
                subword_embeddings = []
                for subword_position in subword_positions:
                    subword_embeddings.append(embeddings[subword_position])
                # subword pooling
                if self.subword_pooling_method == 'avg':
                    pooled_subword_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'max':
                    pooled_subword_embedding, _ = torch.max(torch.stack(subword_embeddings), dim=0)
                subword_aggregated_embeddings.append(pooled_subword_embedding)
            else:
                subword_aggregated_embeddings.append(embeddings[subword_positions[0]])
        return torch.stack(subword_aggregated_embeddings, dim=0)

    def get_word_embeddings(self, sent1, sent2):
        ids, tokens, embedding = [], [], []
        for sent in [sent1, sent2]:
            rets = self.get_word_embedding(sent)
            ids.extend(rets['ids'])
            tokens.extend(rets['tokens'])
            embedding.extend(rets['embeddings'])

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def get_word_embedding(self, sentence):
        ids_sent1 = self.get_ids(sentence)
        ids = [ids_sent1]

        if str(self.device) != 'cpu':
            ids_sent1.data['input_ids'] = ids_sent1.data['input_ids'].to(self.device)
            ids_sent1.data['attention_mask'] = ids_sent1.data['attention_mask'].to(self.device)
            if 'token_type_ids' in ids_sent1.data.keys():
                ids_sent1.data['token_type_ids'] = ids_sent1.data['token_type_ids'].to(self.device)

        tokens_sent1 = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
        tokens = [tokens_sent1]

        emb_sent1 = self.model(**ids_sent1)
        if self.tokenization_mode == 'subword':
            emb_sent1 = self.process_subword(sentence, emb_sent1.squeeze(0))
        embedding = [emb_sent1.squeeze(0).tolist()]

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]


class GetHuggingfaceEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = get_device(device)
        # self.model_names = ['bert-large-uncased', 'roberta-large'] # bert-base-uncased, roberta-base, bert-large-uncased, roberta-large
        self.model_names = ['facebook/bart-large'] # bert-base-uncased, roberta-base, bert-large-uncased, roberta-large, gpt2
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.source_pooling_method = 'avg'

    def get_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def batcher(self, params, batch):
        sentence_embeddings = []
        for sentence in batch:
            indexes = self.tokenizer(' '.join(sentence), return_tensors="pt")
            if str(self.device) != 'cpu':
                indexes.data['input_ids'] = indexes.data['input_ids'].to(self.device)
                indexes.data['attention_mask'] = indexes.data['attention_mask'].to(self.device)
                if 'token_type_ids' in indexes.data.keys():
                    indexes.data['token_type_ids'] = indexes.data['token_type_ids'].to(self.device)

            sentence_embedding = self.model(**indexes)[0].squeeze(0)
            if self.source_pooling_method == 'avg':
                sentence_embedding = sentence_embedding[1:-1]
                sentence_embedding = torch.mean(sentence_embedding, dim=0)
            elif self.source_pooling_method == 'max':
                sentence_embedding = sentence_embedding[1:-1]
                sentence_embedding, _ = torch.max(sentence_embedding, dim=0)
            elif self.source_pooling_method == 'cls':
                sentence_embedding = sentence_embedding[0]
            sentence_embeddings.append(sentence_embedding.tolist())  # get sentence embeddings
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding.tolist()
        return np.array(sentence_embeddings)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='select device')
    args = parser.parse_args()

    if args.device != 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    cls = GetHuggingfaceEmbedding(device=args.device)
    for model_name in cls.model_names:
        print(f'{model_name}-{cls.source_pooling_method}')
        cls.set_model(model_name)
        rets = cls.single_eval(model_name)
        if cls.with_reset_output_file:
            cls.with_reset_output_file = False



'''


'''
