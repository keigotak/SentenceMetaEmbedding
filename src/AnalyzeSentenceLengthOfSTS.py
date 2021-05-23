from STSDataset import STSDataset

class AnalyzeSentenceLengthOfSTS:
    def __init__(self):
        self.modes = ['train', 'dev', 'test']
        self.dataset = {mode: STSDataset(mode=mode) for mode in self.modes}
        self.max_sentence_length = 256

    def analyze(self):
        length = {mode: [0] * self.max_sentence_length for mode in self.modes}
        for mode in self.modes:
            while not self.dataset[mode].is_batch_end():
                ss1, ss2, scores = self.dataset[mode].get_batch()
                for s1, s2 in zip(ss1, ss2):
                    length[mode][len(s1.split(' '))] += 1
                    length[mode][len(s2.split(' '))] += 1

        print('\t' + '\t'.join(self.modes))
        for i in range(self.max_sentence_length):
            print(f'{i}\t' + '\t'.join([str(length[mode][i]) for mode in self.modes]))


if __name__ == '__main__':
    cls = AnalyzeSentenceLengthOfSTS()
    cls.analyze()
