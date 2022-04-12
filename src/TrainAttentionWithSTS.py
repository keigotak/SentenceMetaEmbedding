from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

import dask
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

# from STSDataset import STSDataset
from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
# from AttentionModel import MultiheadSelfAttentionModel, AttentionModel
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device, get_metrics


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        set_seed(0)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, tensor, key_padding_mask=None):
        if key_padding_mask is not None:
            return self.attention(query=tensor, key=tensor, value=tensor, key_padding_mask=key_padding_mask)
        else:
            return self.attention(query=tensor, key=tensor, value=tensor)


class TrainAttentionWithSTS(AbstractTrainer):
    def __init__(self, device='cpu'):
        self.device = get_device(device)
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.source = self.get_source_embeddings() # {model: GetSentenceBertWordEmbedding(model, device=self.device) if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']) else GetHuggingfaceWordEmbedding(model) for model in self.model_names}
        self.embedding_dims = [self.model_dims[model] for model in self.model_names]
        self.attention_head_num = 1
        self.dropout_ratio = 0.2
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat'      # avg, concat
        self.sentence_pooling_method = 'avg'    # avg, concat, max
        self.attention = Attention(embed_dim=max(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio).to(self.device)
        self.attention.train()

        self.learning_ratio = 2e-5
        self.gradient_clip = 0.0
        self.weight_decay = 1e-2
        self.parameters = self.attention.parameters()

        for k, v in self.attention.named_parameters():
            print("{}, {}, {}".format(v.requires_grad, v.size(), k))

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.save_model_path = f'../models/attention-{self.tag}.pkl'
        self.information_file = f'../results/attention/info-{self.tag}.txt'
        self.loss_mode = 'rscore' # rscore, cos

    def get_source_embeddings(self):
        sources = {}
        for model in self.model_names:
            if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'stsb-mpnet-base-v2', 'stsb-roberta-large', 'stsb-mpnet-base-v2', 'stsb-bert-large', 'stsb-distilbert-base']):
                sources[model] = GetSentenceBertWordEmbedding(model, device=self.device)
            elif model == 'glove':
                sources[model] = GloVeModel()
            elif model == 'use':
                sources[model] = GetUniversalSentenceEmbedding()
            else:
                sources[model] = GetHuggingfaceWordEmbedding(model, device=self.device)
        return sources

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            if not self.attention.training:
                self.attention.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            if self.attention.training:
                self.attention.eval()

        # 入力のbatchを一気に処理できるように変換．
        # batch_size * model_name * similar_sentence * embedding_dim -> model_name * similar_sentence * batch_size * sentence_length * embedding_dim
        gs_scores, sys_scores = [], []
        padded_sequences, padding_masks = self.modify_batch_embeddings_to_easy_to_compute(batch_embeddings)

        # similar_sentence 毎にバッチ処理（長さが同じなので，embedding の種類をまとめて一気に）
        sentence_embeddings, attention_weights, normalized_outputs = [], [], []
        for i in range(2):  # for input sentences, sentence1 and sentence2
            sentence_embedding, attention_weight = self.step({model_name: padded_sequences[model_name][i] for model_name in self.model_names},
                                                             padding_mask={model_name: padding_masks[model_name][i] for model_name in self.model_names})
            sentence_embeddings.append(sentence_embedding)
            # attention_weights.append(attention_weight)

        if self.loss_mode == 'cos':
            loss = 1. - self.cos1(sentence_embeddings[0], sentence_embeddings[1])
            loss = torch.mean(loss)
        elif self.loss_mode == 'word':
            # dimensions: sentence, source, words, hidden
            loss1, loss2, loss3 = [], [], []
            for word_embedding in word_embeddings:
                combinations = set()
                for iw1 in range(len(self.model_names)):
                    for iw2 in range(len(self.model_names)):
                        if iw1 == iw2 or (iw1, iw2) in combinations:
                            continue
                        combinations.add((iw1, iw2))
                        combinations.add((iw2, iw1))

                        words1 = word_embedding[self.model_names[iw1]]
                        words2 = word_embedding[self.model_names[iw2]]
                        distance = torch.cdist(words1.contiguous(), words2.contiguous())
                        cosine_similarity = self.cosine_similarity(words1, words2)

                        placeholder = torch.eye(distance.shape[1]).unsqueeze(0).repeat(distance.shape[0], 1, 1)
                        loss1.append(distance[placeholder == 1.])
                        loss2.append(-(self.alpha.weight * distance[placeholder == 0.]).squeeze())

                        # sentence_length = words1.shape[1]
                        # for i in range(sentence_length):
                        #     for j in range(sentence_length):
                        #         if i == j: # 同じ文の同じ単語を比較　同じ単語は近い位置に
                        #             # loss.append((1. - self.cos1(words1[:, i], words2[:, j])))
                        #             loss1.append(torch.norm(words1[:, i] - words2[:, j], dim=1))
                        #         else: # 同じ文の違う単語を比較　違う単語は遠くに
                        #             # loss.append((1. + self.cos1(words1[:, i], words2[:, j])))
                        #             loss2.append((-self.alpha.weight * torch.norm(words1[:, i] - words2[:, j], dim=1)).squeeze(0))

                        # 違う文の比較　
                        # loss.append((1. + self.cos1(sentence_embeddings[0], sentence_embeddings[1])))
                        # loss3.append((-self.beta.weight * (torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1))).squeeze(0))
                        loss3.append(-(self.beta.weight * torch.dist(sentence_embeddings[0].unsqueeze(1), sentence_embeddings[1].unsqueeze(1))).squeeze())

            embedding_loss = [(self.lam.weight * torch.norm(self.va.projection_matrices[model_name].weight.T @ self.va.projection_matrices[model_name].weight - torch.eye(self.va.embedding_dims[model_name], device=self.device))).squeeze() for model_name in self.model_names]

            # loss の和
            if self.lam == 0.0:
                loss = torch.mean(torch.stack(loss1)) + torch.mean(torch.stack(loss2)) + torch.mean(torch.stack(loss3))
            else:
                # loss = torch.mean(torch.abs(torch.stack(loss))) + torch.mean(torch.stack(embedding_loss))
                # loss = torch.abs(torch.mean(torch.stack(loss1))) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))
                # loss = torch.abs(torch.mean(torch.stack(loss1))) + torch.abs(torch.mean(torch.stack(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))
                loss = torch.abs(torch.mean(torch.cat(loss1))) + torch.abs(torch.mean(torch.cat(loss2))) + torch.abs(torch.mean(torch.stack(loss3))) + torch.abs(torch.mean(torch.stack(embedding_loss)))

        elif self.loss_mode == 'rscore':
            loss = torch.abs(torch.norm(sentence_embeddings[0] - sentence_embeddings[1], dim=1)) - torch.as_tensor(scores, dtype=torch.float, device=self.device)
            loss = torch.mean(loss)

        if with_calc_similality:
            sys_score = [self.similarity(e1, e2) for e1, e2 in zip(sentence_embeddings[0].tolist(), sentence_embeddings[1].tolist())]
            sys_scores.extend(sys_score)
            gs_scores.extend(scores)

        running_loss += loss.item()
        print(running_loss)

        if with_training:
            loss.backward()
            if self.gradient_clip > 0.:
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
            self.optimizer.step()
            del loss

        return gs_scores, sys_scores, running_loss

    def step(self, feature, padding_mask=None):
        attention_outputs, attention_weights, weighted_attention_outputs = [], [], []
        sentence_length = feature[self.model_names[0]].shape[1]
        for i in range(sentence_length):
            x = torch.cat([torch.narrow(feature[model_name], dim=1, start=i, length=1) for model_name in self.model_names], dim=1).transpose_(0, 1).requires_grad_(True)
            x, attention_weight = self.attention(tensor=x)
            x = x.transpose_(0, 1)

            if self.source_pooling_method == 'avg':
                attention_outputs.append(torch.mean(x, dim=1).squeeze())
            elif self.source_pooling_method == 'concat':
                attention_outputs.append(torch.flatten(x, start_dim=1))
            attention_weights.append(attention_weight)
        attention_output = torch.stack(attention_outputs).transpose(0, 1)

        if self.sentence_pooling_method == 'avg':
            x = torch.mean(attention_output, dim=1)
        elif self.sentence_pooling_method == 'concat':
            x = torch.flatten(attention_output, start_dim=1)
        elif self.sentence_pooling_method == 'max':
            x, _ = torch.max(attention_output, dim=1)

        return x, attention_weight

    def save_model(self):
        torch.save(self.attention.state_dict(), self.save_model_path)
        self.save_information_file()

    def load_model(self):
        if not os.path.exists(self.save_model_path):
            pass
        else:
            self.attention.load_state_dict(torch.load(self.save_model_path, map_location='cuda:0'))
            self.attention.to(self.device)

    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'attention_head_num: {self.attention_head_num}\n')
            f.write(f'attention_dropout_ratio: {self.dropout_ratio}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'loss_mode: {self.loss_mode}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/attention-{self.tag}.pkl'
        self.information_file = f'../results/attention/info-{self.tag}.txt'

    def update_hyper_parameters(self, hyper_params):
        self.dropout_ratio = hyper_params['dropout_ratio']
        self.attention = Attention(embed_dim=max(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio).to(self.device)

        self.source_pooling_method = hyper_params['source_pooling_method']
        self.sentence_pooling_method = hyper_params['sentence_pooling_method']
        # if self.source_pooling_method == 'avg':
        #     self.attention = nn.MultiheadAttention(embed_dim=max(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio).to(self.device)
        # elif self.source_pooling_method == 'concat':
        #     self.attention = nn.MultiheadAttention(embed_dim=sum(self.embedding_dims), num_heads=self.attention_head_num, dropout=self.dropout_ratio).to(self.device)

        self.learning_ratio = hyper_params['learning_ratio']
        self.gradient_clip = hyper_params['gradient_clip']
        self.weight_decay = hyper_params['weight_decay']
        self.parameters = self.attention.parameters()

        super().__init__()

        self.batch_size = hyper_params['batch_size']
        self.datasets_stsb['train'].batch_size = self.batch_size


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='select device')
    args = parser.parse_args()

    if args.device != 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with_senteval = True
    if with_senteval:
        dp = DataPooler()
        es_metrics = 'pearson'
        if es_metrics == 'dev_loss':
            vw = ValueWatcher(mode='minimize')
        else:
            vw = ValueWatcher()
        model = TrainAttentionWithSTS(args.device)

        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            model.train_epoch()
            model.datasets_stsb['train'].reset(with_shuffle=True)
            rets = model.inference(mode='dev')
            if es_metrics == 'pearson':
                vw.update(rets[es_metrics])
            else:
                vw.update(rets[es_metrics])
            if vw.is_updated():
                model.save_model()
                dp.set('best-epoch', vw.epoch)
                dp.set('best-score', vw.max_score)
            dp.set(f'scores', rets)
        print(f'dev best scores: {model.get_round_score(dp.get("best-score")[-1]) :.2f}')
        print(model.information_file)
        model.append_information_file([f'es_metrics: {es_metrics}'])
        model.append_information_file([f'dev best scores: {model.get_round_score(dp.get("best-score")[-1])}'])

        model.load_model()
        rets = model.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        model.append_information_file(rets["prints"])
        for mode in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            dev_rets = model.inference_sts(mode=mode)
            metrics = get_metrics(dev_rets['sys_scores'], dev_rets['gold_scores'], dev_rets['tags'])
            dev_rets['prints'] = dev_rets['prints'] + [f'{k}: {v}' for k, v in metrics.items()]
            model.append_information_file(dev_rets['prints'])
        # rets = cls.single_eval(model_tag[0])
        print(model.information_file)
    else:
        trainer = TrainAttentionWithSTSBenchmark(device=args.device)
        trainer.train()
