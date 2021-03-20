from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr, pearsonr
from senteval.utils import cosine

from STSDataset import STSDataset
from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AbstractGetSentenceEmbedding import *
from AbstractTrainer import *
from ValueWatcher import ValueWatcher
from DataPooler import DataPooler
from HelperFunctions import set_seed, get_now, get_device


class TrainVectorAttentionWithSTSBenchmark(AbstractTrainer):
    def __init__(self, device='cpu'):
        self.device = get_device(device)
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.source = {model: GetSentenceBertWordEmbedding(model, device=self.device) if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']) else GetHuggingfaceWordEmbedding(model, device=self.device) for model in self.model_names}
        self.embedding_dims = {model: self.model_dims[model] for model in self.model_names}
        self.total_dim = sum(self.embedding_dims.values())

        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'concat' # avg, concat
        self.sentence_pooling_method = 'avg' # avg, max

        self.projection_matrices = {model: torch.FloatTensor(self.embedding_dims[model], self.embedding_dims[model]).uniform_().detach_().to(self.device).requires_grad_(False) for model in self.model_names}
        for model in self.model_names:
            for i, t in enumerate(self.projection_matrices[model]):
                t = t / torch.sum(t)
                self.projection_matrices[model][i] = t
        self.projection_matrices = {model: self.projection_matrices[model].requires_grad_(True) for model in self.model_names}

        self.vector_attention = {model: torch.FloatTensor(self.embedding_dims[model]).uniform_().detach_().to(self.device).requires_grad_(False) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model] / sum(self.vector_attention[model]) for model in self.model_names}
        self.vector_attention = {model: self.vector_attention[model].requires_grad_(True) for model in self.model_names}

        self.learning_ratio = 0.01
        self.gradient_clip = 0.2
        self.weight_decay = 1e-4
        self.with_vector_attention = False
        if self.with_vector_attention:
            self.parameters = list(self.vector_attention.values()) + list(self.projection_matrices.values())
        else:
            self.parameters = list(self.projection_matrices.values())

        super().__init__()

        self.batch_size = 128
        self.datasets['train'].batch_size = self.batch_size
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'

    def batch_step(self, batch_embeddings, scores, with_training=False, with_calc_similality=False):
        running_loss = 0.0
        if with_training:
            self.optimizer.zero_grad()

        gs_scores, sys_scores = [], []
        losses = []
        for embeddings, score in zip(batch_embeddings, scores):
            sentence_embeddings = []
            for i in range(2):  # for input sentences, sentence1 and sentence2
                pooled_sentence_embedding = self.step({model_name: torch.FloatTensor(embeddings[model_name][i]) for model_name in self.model_names})
                sentence_embeddings.append(pooled_sentence_embedding)

            # loss = (torch.dot(*sentence_embeddings) - score) ** 2
            cosine_similarity = self.cos(sentence_embeddings[0], sentence_embeddings[1])
            loss = torch.square(cosine_similarity - (2 * score - 1))
            losses.append(loss)

            if with_calc_similality:
                sys_score = self.similarity(sentence_embeddings[0].tolist(), sentence_embeddings[1].tolist())
                sys_scores.append(sys_score)
                gs_scores.append(score)

            running_loss += loss.item()

        if with_training:
            loss = torch.mean(torch.stack(losses))
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip)
            self.optimizer.step()

        return gs_scores, sys_scores, running_loss

    def step(self, feature):
        projected_embeddings = {
            model_name: torch.einsum('pq, rs->ps', feature[model_name].to(self.device), self.projection_matrices[model_name]) for
            model_name in self.model_names}
        if self.with_vector_attention:
            word_embeddings = {
                model_name: torch.einsum('pq, r->pr', projected_embeddings[model_name], self.vector_attention[model_name]) for
                model_name in self.model_names}
        else:
            word_embeddings = projected_embeddings

        # multiple source embedding and vector attention
        if self.source_pooling_method == 'avg':
            pooled_word_embeddings = []
            for j in range(word_embeddings[self.model_names[0]].shape[0]):
                pooled_word_embedding = []
                for model_name in self.model_names:
                    pooled_word_embedding.append(word_embeddings[model_name][j])
                pooled_word_embeddings.append(torch.mean(torch.stack(pooled_word_embedding), dim=0))
        elif self.source_pooling_method == 'concat':
            pooled_word_embeddings = []
            for j in range(word_embeddings[self.model_names[0]].shape[0]):
                pooled_word_embedding = []
                for model_name in self.model_names:
                    pooled_word_embedding.append(word_embeddings[model_name][j])
                pooled_word_embeddings.append(torch.cat(pooled_word_embedding, dim=0))

        # aggregate word embeddings to sentence embedding
        if self.sentence_pooling_method == 'avg':
            pooled_sentence_embedding = torch.mean(torch.stack(pooled_word_embeddings), dim=0)
        elif self.sentence_pooling_method == 'max':
            pooled_sentence_embedding, _ = torch.max(torch.stack(pooled_word_embeddings), dim=0)

        # L2 normを計算してここで割る
        # torch.dev(pooled_sentence_embedding, torch.norm(pooled_sentence_embedding))
        # pooled_sentence_embedding = pooled_sentence_embedding / torch.sum(pooled_sentence_embedding)

        return pooled_sentence_embedding

    def get_save_path(self, tag):
        return f'../models/vec_attention-{self.tag}-{tag}.pkl'

    def save_model(self):
        if self.with_vector_attention:
            torch.save(self.vector_attention, self.get_save_path('vector'))
        torch.save(self.projection_matrices, self.get_save_path('projection_matrices'))
        self.save_information_file()

    def load_model(self):
        if self.with_vector_attention:
            if not os.path.exists(self.get_save_path('vector')):
                pass
            else:
                self.vector_attention = torch.load(self.get_save_path('vector'))

        if not os.path.exists(self.get_save_path('projection_matrices')):
            pass
        else:
            self.projection_matrices = torch.load(self.get_save_path('projection_matrices'))


    def save_information_file(self):
        super().save_information_file()

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')
            f.write(f'learning_ratio: {self.learning_ratio}\n')
            f.write(f'gradient_clip: {self.gradient_clip}\n')
            f.write(f'weight_decay: {self.weight_decay}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'with_vector_attention: {self.with_vector_attention}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.information_file = f'../results/vec_attention/info-{self.tag}.txt'

    def update_hyper_parameters(self, hyper_params):
        self.source_pooling_method = hyper_params['source_pooling_method']
        self.sentence_pooling_method = hyper_params['sentence_pooling_method']

        self.learning_ratio = hyper_params['learning_ratio']
        self.gradient_clip = hyper_params['gradient_clip']
        self.weight_decay = hyper_params['weight_decay']
        self.with_vector_attention = hyper_params['with_vector_attention']
        if self.with_vector_attention:
            self.parameters = list(self.vector_attention.values()) + list(self.projection_matrices.values())
        else:
            self.parameters = list(self.projection_matrices.values())

        super().__init__()

        self.batch_size = hyper_params['batch_size']
        self.datasets['train'].batch_size = self.batch_size


class EvaluateVectorAttentionModel(AbstractGetSentenceEmbedding):
    def __init__(self, device):
        super().__init__()
        self.tag = get_now()
        self.model_names = ['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.model = TrainVectorAttentionWithSTSBenchmark(device=device)
        self.model.model_names = self.model_names
        self.model_tag = [f'vec_attention-{self.tag}']
        self.output_file_name = 'vec_attention.txt'

    def get_model(self):
        return self.model

    def load_model(self):
        self.model.load_model()

    def batcher(self, params, batch):
        sentence_embeddings = []
        with torch.no_grad():
            for sentence in batch:
                embeddings = {}
                for model_name in self.model_names:
                    rets = self.model.source[model_name].get_word_embedding(' '.join(sentence))
                    embeddings[model_name] = rets['embeddings'][0]

                sentence_embedding = self.model.step({model_name: torch.FloatTensor(embeddings[model_name]) for model_name in
                               self.model_names})
                sentence_embeddings.append(sentence_embedding.tolist())

        return np.array(sentence_embeddings)

    def set_tag(self, tag):
        self.model_tag[0] = f'{self.model_tag[0]}-{tag}'
        self.tag = tag

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
        es_metrics = 'dev_loss'
        if es_metrics == 'dev_loss':
            vw = ValueWatcher(mode='minimize')
        else:
            vw = ValueWatcher()
        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = TrainVectorAttentionWithSTSBenchmark(args.device)

        trainer.model_names = cls.model_names
        trainer.set_tag(cls.tag)

        while not vw.is_over():
            print(f'epoch: {vw.epoch}')
            trainer.train_epoch()
            trainer.datasets['train'].reset(with_shuffle=True)
            rets = trainer.inference(mode='dev')
            if es_metrics == 'pearson':
                vw.update(rets[es_metrics][0])
            else:
                vw.update(rets[es_metrics])
            if vw.is_updated():
                trainer.save_model()
                dp.set('best-epoch', vw.epoch)
                dp.set('best-score', vw.max_score)
            dp.set(f'scores', rets)
        print(f'dev best scores: {trainer.get_round_score(dp.get("best-score")[-1]) :.2f}')

        trainer.load_model()
        rets = trainer.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        cls.model = trainer
        rets = cls.single_eval(cls.model_tag[0])
        trainer.append_information_file([f'es_metrics: {es_metrics}'])
        trainer.append_information_file(rets['text'])
    else:
        cls = EvaluateVectorAttentionModel(device=args.device)
        trainer = TrainVectorAttentionWithSTSBenchmark(args.device)
        tag = '03062021183728375245'
        trainer.set_tag(tag)
        cls.set_tag(tag)
        trainer.load_model()
        rets = trainer.inference(mode='test')
        print(f'test best scores: ' + ' '.join(rets['prints']))
        cls.model = trainer
        model_tag = cls.model_tag[0]
        if cls.tag != trainer.tag:
            model_tag = f'{model_tag}-{trainer.tag}'
        rets = cls.single_eval(model_tag)
