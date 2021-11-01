import os
import torch
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from AbstractGetSentenceEmbedding import *
from HelperFunctions import get_device, get_now


class GetSentenceBertWordEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self, model_name, device='cpu'):
        super().__init__()
        self.device = get_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        # self.model.device = self.device
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.tokenization_mode = 'subword'
        self.subword_pooling_method = 'avg'
        self.sentence_pooling_method = 'avg'
        self.embeddings = {self.model_name: {}}

        if model_name == 'gpt2':
            self.tokenixer

        self.tag = get_now()
        self.information_file = f'../results/sberts/info-{self.tag}.txt'

        # self.embeddings_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-embeddings.txt')
        # self.indexer_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-indexer.txt')
        #
        # self.cached_embeddings = {}
        # if self.embeddings_path.exists():
        #     with self.embeddings_path.open('r') as f:
        #         for t in f.readlines():
        #             lines = t.strip().split('\t')
        #             self.cached_embeddings[lines[0]] = [list(map(float, l.split(' '))) for l in lines[1:]] # key is sentID
        # else:
        #     self.embeddings_path.touch()
        #
        # self.sent_to_id = {}
        # if self.indexer_path.exists():
        #     with self.indexer_path.open('r') as f:
        #         for t in f.readlines():
        #             sentID, sentence = t.strip().split('\t')
        #             self.sent_to_id[sentence] = sentID
        # else:
        #     self.indexer_path.touch()
        #
        # self.sentence_id = len(self.sent_to_id)


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
            if i != 0 and self.model_name in {'roberta-large-nli-stsb-mean-tokens'}:
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
                    # print(f'{subword_position}, {embeddings.shape[0]}')
                    if subword_position >= embeddings.shape[0]:
                        subword_embeddings.append(torch.FloatTensor([0.0] * embeddings[0].shape[0]).requires_grad_(False))
                    else:
                        if type(embeddings[subword_positions[0]]) == np.ndarray:
                            subword_embeddings.append(torch.FloatTensor(embeddings[subword_position]).requires_grad_(False))
                        else:
                            subword_embeddings.append(embeddings[subword_position].requires_grad_(False))
                # subword pooling
                if self.subword_pooling_method == 'avg':
                    pooled_subword_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'max':
                    pooled_subword_embedding, _ = torch.max(torch.stack(subword_embeddings), dim=0)
                subword_aggregated_embeddings.append(pooled_subword_embedding)
            else:
                if len(subword_positions) == 0:
                    if type(embeddings[subword_positions]) == np.ndarray:
                        subword_aggregated_embeddings.append(torch.zeros_like(torch.FloatTensor(embeddings[0])).requires_grad_(False))
                    else:
                        subword_aggregated_embeddings.append(torch.zeros_like(embeddings[0]).requires_grad_(False))
                else:
                    if subword_positions[0] >= embeddings.shape[0]:
                        subword_embeddings.append(torch.FloatTensor([0.0] * embeddings[0].shape[0]).requires_grad_(False))
                    else:
                        if type(embeddings[subword_positions[0]]) == np.ndarray:
                            subword_aggregated_embeddings.append(torch.FloatTensor(embeddings[subword_positions[0]]).requires_grad_(False))
                        else:
                            subword_aggregated_embeddings.append(embeddings[subword_positions[0]].requires_grad_(False))
        return torch.stack(subword_aggregated_embeddings, dim=0)

    def get_word_embedding(self, sentence, with_process_subwords=True):
        # if sentence in self.sent_to_id.keys():
        #     embedding = [self.cached_embeddings[self.sent_to_id[sentence]]]
        #     tokens = [sentence.split(' ')]
        #     return {'ids': None, 'tokens': tokens, 'embeddings': embedding}

        ids_sent1 = self.get_ids(sentence)
        ids = [ids_sent1]

        tokens_sent1 = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
        tokens = [tokens_sent1[1: -1]]

        emb_sent1 = self.model.encode(sentence, output_value='token_embeddings')
        if with_process_subwords:
            if self.tokenization_mode == 'subword':
                emb_sent1 = self.process_subword(sentence, emb_sent1)
        else:
            emb_sent1 = torch.FloatTensor(emb_sent1)
        embedding = [emb_sent1.squeeze(0).tolist()[1: -1]]

        # with self.embeddings_path.open('a') as f:
        #     line = f'SID{self.sentence_id}\t' + '\t'.join([' '.join([str(e) for e in es]) for es in embedding[0]]) + '\n'
        #     f.write(line)
        #
        # with self.indexer_path.open('a') as f:
        #     line = f'SID{self.sentence_id}\t{sentence}\n'
        #     f.write(line)
        # self.sentence_id += 1

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def get_word_embeddings(self, sent1, sent2):
        ids, tokens, embedding = [], [], []
        for sent in [sent1, sent2]:
            rets = self.get_word_embedding(sent)
            ids.extend(rets['ids'])
            tokens.extend(rets['tokens'])
            embedding.extend(rets['embeddings'])

        # ids_sent1 = self.get_ids(sent1)
        # ids_sent2 = self.get_ids(sent2)
        # ids = [ids_sent1, ids_sent2]
        #
        # tokens_sent1 = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
        # tokens_sent2 = self.tokenizer.convert_ids_to_tokens(ids_sent2.data['input_ids'][0])
        # tokens = [tokens_sent1, tokens_sent2]
        #
        # emb_sent1 = self.model.encode(sent1, output_value='token_embeddings')
        # emb_sent2 = self.model.encode(sent2, output_value='token_embeddings')
        # if self.tokenization_mode == 'subword':
        #     emb_sent1 = self.process_subword(sent1, emb_sent1)
        #     emb_sent2 = self.process_subword(sent2, emb_sent2)
        # embedding = [emb_sent1.squeeze(0).tolist(), emb_sent2.squeeze(0).tolist()]

        return {'ids': ids, 'tokens': tokens, 'embeddings': embedding}

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]

    def get_model(self):
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words

        sentence_embeddings = []
        for sentence in sentences:
            sentence_embedding = self.get_word_embedding(sentence)['embeddings'][0]
            if self.sentence_pooling_method == 'avg':
                # sentence_embedding = sentence_embedding[1:-1]
                sentence_embedding = torch.mean(torch.FloatTensor(sentence_embedding).requires_grad_(False), dim=0).tolist()
            elif self.sentence_pooling_method == 'max':
                # sentence_embedding = sentence_embedding[1:-1]
                sentence_embedding, _ = torch.max(torch.FloatTensor(sentence_embedding).requires_grad_(False), dim=0)
                sentence_embedding = sentence_embedding.tolist()
            sentence_embeddings.append(sentence_embedding)  # get token embeddings
            self.embeddings[model_name][' '.join(sentence)] = sentence_embeddings
        return np.array(sentence_embeddings)

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_name)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/sberts-{self.tag}.pkl'
        self.information_file = f'../results/sberts/info-{self.tag}.txt'


class GetSentenceBertEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        super().__init__()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens',
                            'roberta-base-nli-stsb-mean-tokens',
                            'bert-large-nli-stsb-mean-tokens',
                            'distilbert-base-nli-stsb-mean-tokens']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False

    def get_model(self):
        self.model = SentenceTransformer(self.model_name)
        return self.model

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
        sentence_embeddings = self.model.encode(sentences)  # get sentence embeddings
        for sentence, sentence_embedding in zip(batch, sentence_embeddings):
            self.embeddings[model_name][' '.join(sentence)] = sentence_embedding.tolist()
        return sentence_embeddings


if __name__ == '__main__':
    is_pooled = False
    if is_pooled:
        cls = GetSentenceBertEmbedding()
        for model_name in cls.model_names:
            print(model_name)
            cls.set_model(model_name)
            cls.single_eval(model_name)
            if cls.with_reset_output_file:
                cls.with_reset_output_file = False
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default='cpu', help='select device')
        args = parser.parse_args()

        if args.device != 'cpu':
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        # for model_name in ['bert-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']:
        # for model_name in ['gpt2', 'facebook/bart-base']:
        for model_name in ['stsb-xlm-r-multilingual']: # ['xlm-r-100langs-bert-base-nli-stsb-mean-tokens']:
        # for model_name in ['stsb-mpnet-base-v2']:
            cls = GetSentenceBertWordEmbedding(model_name=model_name, device=args.device)
            cls.set_tag(get_now())
            print(f'{model_name}, {cls.sentence_pooling_method}')
            cls.set_model(model_name)
            cls.single_eval(model_name)
            if cls.with_reset_output_file:
                cls.with_reset_output_file = False





'''
self.sentence_pooling_method = 'avg'
                      stsb-mpnet-base-v2      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              79.36              77.15              76.13              73.77
                               STS13-all              90.02              88.83              84.00              83.41
                               STS14-all              92.35              89.97              92.94              90.70
                               STS15-all              89.36              89.45              88.14              88.12
                               STS16-all              85.86              86.92              85.53              86.62
                        STSBenchmark-all              87.85              88.22                  -                  -

self.sentence_pooling_method = 'max'
                      stsb-mpnet-base-v2      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.07              76.70              75.58              73.93
                               STS13-all              87.45              86.27              79.03              78.24
                               STS14-all              90.39              88.63              90.71              89.13
                               STS15-all              86.02              86.79              83.99              84.72
                               STS16-all              84.38              85.96              84.12              85.74
                        STSBenchmark-all              85.72              85.88                  -                  -

self.sentence_pooling_method = 'avg'
xlm-r-100langs-bert-base-nli-stsb-mean-tokens      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.95              74.88              73.92              71.54
                               STS13-all              85.74              85.21              79.21              78.65
                               STS14-all              89.44              87.27              90.02              88.08
                               STS15-all              86.19              86.35              84.17              84.15
                               STS16-all              80.60              82.20              80.20              81.81
                        STSBenchmark-all              82.03              82.23                  -                  -

self.sentence_pooling_method = 'max'
xlm-r-100langs-bert-base-nli-stsb-mean-tokens      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              76.50              75.03              73.91              71.94
                               STS13-all              84.81              84.57              77.40              77.35
                               STS14-all              88.43              86.78              88.78              87.42
                               STS15-all              85.03              85.76              82.86              83.47
                               STS16-all              79.82              81.43              79.49              81.13
                        STSBenchmark-all              81.63              81.77                  -                  -

'''
