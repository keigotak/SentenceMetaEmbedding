import sys
import random
from pathlib import Path


class STSDataset:
    random.seed(91)

    def __init__(self, mode='train'):
        self.current = 0
        self.batch_size = 4

        self.mode = mode
        self.path = None
        if mode in ['train', 'dev', 'test']:
            self.path = Path(f'../data/sts-{mode}.csv')

        if self.path is None:
            sys.exit('Please set dataset type.')

        self.texts = None
        with self.path.open('r') as f:
            self.texts = [self.get_data_dict(*line.strip().split('\t')) for line in f.readlines()]
        self.dataset_size = len(self.texts)

    @staticmethod
    def get_data_dict(genre, filename, year, index, score, sentence1, sentence2):
        return {'genre': genre,
                'filename': filename,
                'year': year,
                'index': int(index),
                'score': float(score),
                'sentence1': sentence1,
                'sentence2': sentence2}

    def shuffle(self):
        random.shuffle(self.texts)

    def get_batch(self):
        if self.current + self.batch_size < len(self.texts):
            batch = self.texts[self.current: self.current + self.batch_size]
            self.current += self.batch_size
        else:
            batch = self.texts[self.current:]
            self.current = len(self.texts)

        sentences1 = [b['sentence1'] for b in batch]
        sentences2 = [b['sentence2'] for b in batch]
        scores = [b['score']/5.0 for b in batch]
        return sentences1, sentences2, scores

    def reset(self, with_shuffle=False):
        self.current = 0
        if with_shuffle:
            self.shuffle()

    def is_batch_end(self):
        if self.current == len(self.texts):
            return True
        return False

    def __str__(self):
        return f'{self.current} / {self.dataset_size}'


if __name__ == '__main__':
    s = STSDataset()
    while not s.is_batch_end():
        print(s.get_batch())

