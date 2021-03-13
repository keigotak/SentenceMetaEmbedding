import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from GetSentenceBertEmbedding import GetSentenceBertWordEmbedding
from GetHuggingfaceEmbedding import GetHuggingfaceWordEmbedding
from AbstractGetSentenceEmbedding import *
from HelperFunctions import get_now, get_device


class GetMetaWordEmbedding(AbstractGetSentenceEmbedding):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = get_device(device)
        self.model_names = ['bert-large-uncased', 'roberta-large'] # 'bert-large-uncased', 'roberta-large', 'roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens'
        self.model_dims = {'bert-large-uncased': 1024, 'roberta-large': 1024, 'roberta-large-nli-stsb-mean-tokens': 1024, 'bert-large-nli-stsb-mean-tokens': 1024}
        self.source = {model: GetSentenceBertWordEmbedding(model, device=self.device) if model in set(['roberta-large-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens']) else GetHuggingfaceWordEmbedding(model, device=self.device) for model in self.model_names}
        self.total_dim = sum([self.model_dims[model] for model in self.model_names])
        self.tokenization_mode = self.source[self.model_names[0]].tokenization_mode
        self.subword_pooling_method = self.source[self.model_names[0]].subword_pooling_method
        self.source_pooling_method = 'avg'   # avg, concat
        self.sentence_pooling_method = 'avg'    # avg, max

        self.tag = get_now()
        self.information_file = f'../results/meta/info-{self.tag}.txt'

    def get_model(self):
        return self.source

    def set_model(self, model_name):
        self.model_name = model_name

    def batcher(self, params, batch):
        sentence_embeddings = []
        for sentence in batch:
            word_embeddings = {}
            for model_name in self.model_names:
                embedding = self.source[model_name].get_word_embedding(' '.join(sentence))
                word_embeddings[model_name] = embedding['embeddings'][0]

            if self.source_pooling_method == 'avg':
                pooled_word_embeddings = []
                for j in range(len(word_embeddings[self.model_names[0]][1:-1])):
                    pooled_word_embedding = []
                    for model_name in self.model_names:
                        pooled_word_embedding.append(torch.FloatTensor(word_embeddings[model_name][j]).requires_grad_(False))
                    pooled_word_embeddings.append(torch.mean(torch.stack(pooled_word_embedding), dim=0))
            elif self.source_pooling_method == 'concat':
                pooled_word_embeddings = []
                for j in range(len(word_embeddings[self.model_names[0]][1:-1])):
                    pooled_word_embedding = []
                    for model_name in self.model_names:
                        pooled_word_embedding.append(torch.FloatTensor(word_embeddings[model_name][j]).requires_grad_(False))
                    pooled_word_embeddings.append(torch.cat(pooled_word_embedding, dim=0))

            if self.sentence_pooling_method == 'avg':
                pooled_sentence_embedding = torch.mean(torch.stack(pooled_word_embeddings), dim=0)
            elif self.sentence_pooling_method == 'max':
                pooled_sentence_embedding, _ = torch.max(torch.stack(pooled_word_embeddings), dim=0)

            sentence_embeddings.append(pooled_sentence_embedding.tolist())

        return np.array(sentence_embeddings)

    def save_information_file(self):
        information_file = Path(self.information_file)
        if not information_file.parent.exists():
            information_file.parent.mkdir(exist_ok=True)

        with Path(self.information_file).open('w') as f:
            f.write(f'source: {",".join(self.model_names)}\n')
            f.write(f'tokenization_mode: {self.tokenization_mode}\n')
            f.write(f'subword_pooling_method: {self.subword_pooling_method}\n')
            f.write(f'source_pooling_method: {self.source_pooling_method}\n')
            f.write(f'sentence_pooling_method: {self.sentence_pooling_method}\n')

    def set_tag(self, tag):
        self.tag = tag
        self.save_model_path = f'../models/meta-{self.tag}.pkl'
        self.information_file = f'../results/meta/info-{self.tag}.txt'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='select device')
    args = parser.parse_args()

    if args.device != 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    cls = GetMetaWordEmbedding(device=args.device)
    cls.set_model(f"{'_'.join(cls.model_names)}_{cls.source_pooling_method}_{cls.sentence_pooling_method}")
    rets = cls.single_eval(cls.model_name)
    if cls.with_reset_output_file:
        cls.with_reset_output_file = False


'''


'''
