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

        self.embeddings_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-embeddings.txt')
        self.indexer_path = Path(f'./sentence_embeddings/{model_name}-{self.tokenization_mode}-{self.subword_pooling_method}-indexer.txt')

        self.cached_embeddings = {}
        if self.embeddings_path.exists():
            with self.embeddings_path.open('r') as f:
                for t in f.readlines():
                    lines = t.strip().split('\t')
                    self.cached_embeddings[lines[0]] = [list(map(float, l.split(' '))) for l in lines[1:]] # key is sentID
        else:
            self.embeddings_path.touch()

        self.sent_to_id = {}
        if self.indexer_path.exists():
            with self.indexer_path.open('r') as f:
                for t in f.readlines():
                    sentID, sentence = t.strip().split('\t')
                    self.sent_to_id[sentence] = sentID
        else:
            self.indexer_path.touch()

        self.sentence_id = len(self.sent_to_id)
        self.with_save_embeddings = True

    def train(self):
        self.model.train()

    def get_ids(self, sent):
        ids_sent = self.model.tokenize(sent)
        if self.tokenization_mode == 'original':
            ids_sent.data['input_ids'] = torch.LongTensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + sent.split(' ') + [self.tokenizer.sep_token])).unsqueeze(0)
            ids_sent.data['token_type_ids'] = self.tokenizer.convert_ids_to_tokens(ids_sent1.data['input_ids'][0])
            ids_sent.data['attention_mask'] = torch.ones_like(ids_sent.data['input_ids'])
        return ids_sent

    def process_subword(self, subword_list, embeddings):
        # aggregate subwords embeddings
        subword_aggregated_embeddings = []
        for i in range(-1, max(subword_list) + 1):
            subword_positions = [j for j, x in enumerate(subword_list) if x == i]
            # if the word are subworded
            if len(subword_positions) > 1:
                subword_embeddings = []
                for subword_position in subword_positions:
                    if type(embeddings[subword_positions[0]]) == np.ndarray:
                        subword_embeddings.append(torch.FloatTensor(embeddings[subword_position]).requires_grad_(False))
                    else:
                        subword_embeddings.append(embeddings[subword_position].requires_grad_(False))
                # subword pooling
                if self.subword_pooling_method == 'avg':
                    pooled_subword_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'max':
                    pooled_subword_embedding, _ = torch.max(torch.stack(subword_embeddings), dim=0)
                elif self.subword_pooling_method == 'head':
                    pooled_subword_embedding = subword_embeddings[0]
                subword_aggregated_embeddings.append(pooled_subword_embedding)
            else:
                if len(subword_positions) == 0:
                    if type(embeddings[subword_positions]) == np.ndarray:
                        subword_aggregated_embeddings.append(torch.zeros_like(torch.FloatTensor(embeddings[0])).requires_grad_(False))
                    else:
                        subword_aggregated_embeddings.append(torch.zeros_like(embeddings[0]).requires_grad_(False))
                else:
                    if type(embeddings[subword_positions[0]]) == np.ndarray:
                        subword_aggregated_embeddings.append(torch.FloatTensor(embeddings[subword_positions[0]]).requires_grad_(False))
                    else:
                        subword_aggregated_embeddings.append(embeddings[subword_positions[0]].requires_grad_(False))
        return torch.stack(subword_aggregated_embeddings, dim=0)

    def get_word_embedding(self, sentence, with_process_subwords=True):
        # sentence = sentence.replace('`', '')
        # sentence = sentence.replace("'", "")
        if '�' in sentence:
            sentence = sentence.replace('� ', '')
        if 'o ̯ reĝ' in sentence:
            sentence = sentence.replace('o ̯ reĝ', '')
        enc = self.model.tokenizer(sentence).encodings[0]

        indexes, subwords, subword_ids = [], [], []
        index, subword, subword_id = [], [], []

        for i, (o1, o2) in enumerate(zip(enc.offsets, enc.offsets[1:])):
            if o1 == (0, 0):
                continue
            if o1 == o2:
                index.append(enc.ids[i])
                subword.append(enc.tokens[i])
                subword_id.append(enc.ids[i])
            else:
                if o1[1] == o2[0]:
                    index.append(enc.ids[i])
                    subword.append(enc.tokens[i])
                    subword_id.append(enc.ids[i])
                else:
                    index.append(enc.ids[i])
                    subword.append(enc.tokens[i])
                    subword_id.append(enc.ids[i])

                    indexes.append(index)
                    subwords.append(subword)
                    subword_ids.append(subword_id)

                    index, subword, subword_id = [], [], []
        flatten_indexes = [[j] * len(subwords[j]) for j in range(len(subwords))]
        flatten_indexes = [-1] + [i for index in flatten_indexes for i in index] + [len(subwords)]

        # word_ids = enc.word_ids
        # indexes, subwords = [], []
        # for i in range(word_ids[-2] + 1):
        #     index, subword = [], []
        #     for t, w in zip(enc.tokens, word_ids):
        #         if w == i:
        #             index.append(w)
        #             subword.append(t)
        #     indexes.append(index)
        #     subwords.append(subword)

        emb_sent1 = self.model.encode(sentence, output_value='token_embeddings')
        emb_sent1 = self.model.forward({'input_ids': torch.LongTensor([enc.ids]).to(self.device), 'attention_mask': torch.LongTensor([enc.attention_mask]).to(self.device)})
        emb_sent1 = emb_sent1['token_embeddings'].squeeze(0).cpu().detach().numpy()

        if with_process_subwords:
            if self.tokenization_mode == 'subword':
                emb_sent1 = self.process_subword(flatten_indexes, emb_sent1)
        else:
            emb_sent1 = torch.FloatTensor(emb_sent1)
        embedding = [emb_sent1.squeeze(0).tolist()[1: -1]] # 1, length, embedding_dim

        return {'ids': indexes, 'tokens': subwords, 'embeddings': embedding}

    def get_word_embeddings(self, sent1, sent2):
        ids, tokens, embedding = [], [], []
        for sent in [sent1, sent2]:
            rets = self.get_word_embedding(sent)
            ids.extend(rets['ids'])
            tokens.extend(rets['tokens'])
            embedding.extend(rets['embeddings'])

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
            self.embeddings[model_name][sentence] = sentence_embedding.copy()
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        super().__init__()
        # self.model_names = ['roberta-large-nli-stsb-mean-tokens',
        #                     'roberta-base-nli-stsb-mean-tokens',
        #                     'bert-large-nli-stsb-mean-tokens',
        #                     'distilbert-base-nli-stsb-mean-tokens']
        self.model_names = ['stsb-bert-large', 'stsb-roberta-large', 'stsb-distilbert-base']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = True

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
        # for model_name in ['stsb-xlm-r-multilingual']: # ['xlm-r-100langs-bert-base-nli-stsb-mean-tokens']:
        # for model_name in ['stsb-mpnet-base-v2']:
        for model_name in ['stsb-bert-large', 'stsb-roberta-large', 'stsb-distilbert-base']:
            cls = GetSentenceBertWordEmbedding(model_name=model_name, device=args.device)
            cls.set_tag(get_now())
            print(f'{model_name}, {cls.sentence_pooling_method}')
            cls.set_model(model_name)
            cls.single_eval(model_name)
            if cls.with_reset_output_file:
                cls.with_reset_output_file = False





'''
STS12-16を全てまとめて学習する

単語で位置がどう変わるか
文を３つAB高い，ACは低い　Aは基準で-0.5, 0.0, 0.5の3点がある．
メタembeddingはソースに引っ張られるのでそれを見せられるようにする．

there is a certain physical distance within which some other entity can participate in an event ( typically perception or manipulation ) with the participant. alternatively , the event may be indicated metonymically by a instrument . [ note the connection with * distance , sufficiency , and capability. words in this frame can generally be paraphrased " close enough to be able to " . ]
[[-0.01814483 -0.12344421  0.04544263 ...  0.06656019  0.02028461,  -0.02380267], [ 0.08801881 -0.13987686 -0.10606434 ...  0.02950472  0.04954498,  -0.02610875], [ 0.02853095 -0.32427612 -0.06939633 ...  0.08875471  0.02078654,  -0.0742402 ], ..., [ 0.03648302 -0.11451291  0.07402899 ... -0.00294275 -0.02224921,  -0.05778958], [ 0.09217613  0.02251625  0.0576703  ...  0.02815251 -0.08181024,  -0.06141173], [ 0.02454806 -0.08518206  0.04093451 ...  0.04033475 -0.06351527,  -0.03479177]]

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


stsb-bert-large, avg
                         stsb-bert-large      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.74              78.78              78.08              75.59
                               STS13-all              89.76              88.89              83.01              82.42
                               STS14-all              92.37              90.93              93.42              92.23
                               STS15-all              88.29              88.22              86.36              86.12
                               STS16-all              83.24              84.02              83.00              83.78
                        STSBenchmark-all              83.66              84.05                  -                  -
stsb-roberta-large, avg
                      stsb-roberta-large      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              78.32              76.94              74.81              73.42
                               STS13-all              88.30              87.15              81.53              80.23
                               STS14-all              91.72              90.18              92.63              91.23
                               STS15-all              87.72              87.88              85.74              85.81
                               STS16-all              84.55              84.93              84.21              84.58
                        STSBenchmark-all              82.46              83.26                  -                  -
stsb-distilbert-base, avg
                    stsb-distilbert-base      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              77.69              76.26              74.02              71.96
                               STS13-all              88.48              87.82              81.39              80.84
                               STS14-all              91.49              89.97              92.70              91.43
                               STS15-all              86.34              86.48              84.13              84.18
                               STS16-all              82.59              83.32              82.26              82.96
                        STSBenchmark-all              83.40              83.67                  -                  -
'''
