import sys
import random
from pathlib import Path


class STSDataset:
    def __init__(self, mode='train'):
        self.current = 0
        self.batch_size = 1

        self.mode = mode
        self.path = None
        if mode in ['train', 'dev', 'test']:
            self.path = Path(f'../data/sts-{mode}.csv')

        if self.path is None:
            sys.exit('Please set dataset type.')

        self.texts = None
        with self.path.open('r', encoding='utf-8') as f:
            self.texts = [self.get_data_dict(*line.strip().split('\t')) for line in f.readlines()]
        self.dataset_size = len(self.texts)
        self.batch_mode = 'full' # full, fixed

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
        if self.batch_mode == 'full':
            if self.current == self.dataset_size:
                return True
        elif self.batch_mode == 'fixed':
            if self.current >= int(self.dataset_size / 10 + 0.5):
                return True
        return False

    def __str__(self):
        return f'{self.current} / {self.dataset_size}'


class MergedSTSDataset(STSDataset):
    def __init__(self, mode='train'):
        self.current = 0
        self.batch_size = 128

        self.mode = mode
        self.path = None
        if mode in ['train', 'dev', 'test']:
            self.path = Path(f'../data/sts-{mode}.csv')

        if self.path is None:
            sys.exit('Please set dataset type.')

        self.texts = None
        with self.path.open('r', encoding='utf-8') as f:
            self.texts = [self.get_data_dict(*line.strip().split('\t')) for line in f.readlines()]
        unique_sentences = set([t['sentence1'] for t in self.texts]) | set([t['sentence2'] for t in self.texts])

        ignore_sentences = set(['the president has less of capacities than it does not appear to with it , and it particuli � rement is particuli � rement eclipsed by the guide supr � me the ayatollah ali khamenei .',
                                'the english word " right " comes from proto-indo-european word o ̯ reĝtos which meant " correct " and had cognates o ̯ reĝr " directive , order " , o ̯ reĝs " king , ruler " , o ̯ reĝti " guides , directs " , o ̯ reĝi ̯ om " kingdom " .',
                                'jackie battley and newt gingrich marriage profile -- marriage of newt gingrich and jackie battley newt gingrich  s second wife dishes hard to esquire : his money woes , his philandering , his meltdown | tpmmuckraker newt gingrich  s second wife dishes hard to esquire : his money woes , his philandering , his meltdown | tpmmuckraker newt gingrich - marital affairs in 1980 , newt began a relationship with a woman he met at a political fundraiser , marianne .',
                                'the farsi word " جادو " / dʒɒ : du : / is thought to be cognate with sanskrit " यातु " / ja : tu / , with a similar meaning .'
                               ])

        for d in ['STS12-en-test', 'STS13-en-test', 'STS14-en-test', 'STS15-en-test', 'STS16-en-test']:
            if '12' in d:
                tags = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
                # STS.input.MSRpar.txt
                # STS.input.MSRvid.txt
                # STS.input.SMTeuroparl.txt
                # STS.input.surprise.OnWN.txt
                # STS.input.surprise.SMTnews.txt
            elif '13' in d:
                tags = ['FNWN', 'headlines', 'OnWN']
                # STS.input.FNWN.txt
                # STS.input.headlines.txt
                # STS.input.OnWN.txt
            elif '14' in d:
                tags = ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news']
                # STS.input.deft-forum.txt
                # STS.input.deft-news.txt
                # STS.input.headlines.txt
                # STS.input.images.txt
                # STS.input.OnWN.txt
                # STS.input.tweet-news.txt
            elif '15' in d:
                tags = ['answers-forums', 'answers-students', 'belief', 'headlines', 'images']
                # STS.input.answers-forums.txt
                # STS.input.answers-students.txt
                # STS.input.belief.txt
                # STS.input.headlines.txt
                # STS.input.images.txt
            elif '16' in d:
                tags = ['answer-answer', 'headlines', 'plagiarism', 'postediting', 'question-question']
                # STS.input.answer-answer.txt
                # STS.input.headlines.txt
                # STS.input.plagiarism.txt
                # STS.input.postediting.txt
                # STS.input.question-question.txt
            for tag in tags:
                with Path(f'/home/keigo/SentEval/data/downstream/STS/{d}/STS.input.{tag}.txt').open('r') as f:
                    texts = f.readlines()
                for text in texts:
                    sentence1, sentence2 = text.strip().split('\t')
                    if sentence1 in ignore_sentences or sentence2 in ignore_sentences:
                        continue
                    self.texts.append({'sentence1': sentence1, 'sentence2': sentence2, 'score': 0.0})

        self.dataset_size = len(self.texts)
        self.batch_mode = 'full' # full, fixed

        # print(self.texts)



if __name__ == '__main__':
    # s = STSDataset()
    # while not s.is_batch_end():
    #     print(s.get_batch())
    s = MergedSTSDataset()


