import numpy as np
from AbstractTrainer import *
from AbstractGetSentenceEmbedding import *
from HelperFunctions import set_seed, get_metrics
from STSDataset import STSBenchmarkDataset, STSDataset

class GetSVDSentenceEmbedding(AbstractTrainer):
    def __init__(self):
        # super().__init__()
        set_seed(0)
        self.dataset_type = 'normal'
        self.datasets_stsb = {mode: STSBenchmarkDataset(mode=mode) for mode in ['train', 'dev', 'test']}
        self.datasets_sts = {mode: STSDataset(mode=mode) for mode in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']}
        self.model_names = ['svd']
        self.embeddings = {model_name: {} for model_name in self.model_names}
        self.output_file_name = 'svd.txt'
        self.with_reset_output_file = False
        self.with_save_embeddings = False
        self.tag = '20220319212359823547' # '01122022184319086520' # '01112022204815661173' # '01082022070340899941' # '01032022161750908240' # '04072021215506466248'
        self.indexer, self.model = None, None
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=2)
        self.information_file = f'../results/svd/info-{self.tag}.txt'

    def get_model(self):
        if self.tag is None:
            with open('../models/sts_svd.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('../models/sts_svd_sentence_indexer.pkl', 'rb') as f:
                self.indexer = pickle.load(f)
        else:
            with open(f'../models/svd_{self.tag}.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(f'../models/svd_{self.tag}_sentence_indexer.pkl', 'rb') as f:
                self.indexer = pickle.load(f)
        return self.model

    def batch_step(self, sentences, with_training=False, with_calc_similality=False):
        sentence_embeddings = []
        for sentence in sentences:
            if '�' in sentence:
                sentence = sentence.replace('� ', '')
            if 'o ̯ reĝ' in sentence:
                sentence = sentence.replace('o ̯ reĝ', '')

            indexes = self.indexer[sentence]
            sentence_embedding = self.model[indexes].tolist()
            sentence_embeddings.append(sentence_embedding)  # get token embeddings
            self.embeddings[self.model_names[0]][sentence] = sentence_embedding
        return sentence_embeddings

    def load_model(self):
        self.get_model()

    def set_tag(self, tag):
        self.tag = tag

    def inference(self, mode='dev', with_test_mode=False):
        running_loss = 0.0
        results = {}
        pearson_rs, spearman_rhos = [], []

        # batch loop
        sys_scores, gs_scores = [], []
        with torch.inference_mode():
            while not self.datasets_stsb[mode].is_batch_end(with_test_mode=with_test_mode):
                sentences1, sentences2, scores = self.datasets_stsb[mode].get_batch()

                # get vector representation for each embedding
                batch_embeddings = []
                batch_tokens = []
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = self.batch_step([sent1, sent2])
                    sys_score = self.similarity(embeddings[0], embeddings[1])
                    sys_scores.append(sys_score)
                gs_scores.extend(scores)
                # running_loss += loss

        pearson_rs = pearsonr(sys_scores, gs_scores)[0]
        spearman_rhos = spearmanr(sys_scores, gs_scores)[0]

        avg_pearson_r = np.average(pearson_rs)
        avg_spearman_rho = np.average(spearman_rhos)

        results = {'pearson': avg_pearson_r,
                   'spearman': avg_spearman_rho,
                   'nsamples': len(sys_scores),
                   'sys_scores': sys_scores,
                   'gold_scores': gs_scores
                   }

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"]) :.2f}']
        results['prints'] = print_contents

        self.datasets_stsb[mode].reset()

        return results

    def inference_sts(self, mode='STS12', with_test_mode=False):
        running_loss = 0.0
        results = {}
        pearson_rs, spearman_rhos = [], []

        # batch loop
        sys_scores, gs_scores, tag_sequence = [], [], []
        with torch.inference_mode():
            while not self.datasets_sts[mode].is_batch_end(with_test_mode=with_test_mode):
                sentences1, sentences2, scores, tags = self.datasets_sts[mode].get_batch()

                # get vector representation for each embedding
                batch_embeddings = []
                batch_tokens = []
                for sent1, sent2 in zip(sentences1, sentences2):
                    embeddings = self.batch_step([sent1, sent2])
                    sys_score = self.similarity(embeddings[0], embeddings[1])
                    sys_scores.append(sys_score)
                gs_scores.extend(scores)
                tag_sequence.extend(tags)
                # running_loss += loss
        rets = get_metrics(sys_scores, gs_scores, tags)

        pearson_rs = pearsonr(sys_scores, gs_scores)[0]
        spearman_rhos = spearmanr(sys_scores, gs_scores)[0]

        avg_pearson_r = np.average(pearson_rs)
        avg_spearman_rho = np.average(spearman_rhos)

        results = {'pearson': avg_pearson_r,
                   'spearman': avg_spearman_rho,
                   'nsamples': len(sys_scores),
                   'sys_scores': sys_scores,
                   'gold_scores': gs_scores,
                   'tags': tag_sequence
                   }

        print_contents = [f'STSBenchmark-{mode}',
                          f'pearson: {self.get_round_score(results["pearson"]) :.2f}',
                          f'spearman: {self.get_round_score(results["spearman"]) :.2f}']
        results['prints'] = print_contents

        self.datasets_sts[mode].reset()

        return results


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model = GetSVDSentenceEmbedding()
    model.load_model()
    rets = model.inference(mode='dev')
    model.append_information_file(rets["prints"])
    rets = model.inference(mode='test')
    model.append_information_file(rets["prints"])
    for mode in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        dev_rets = model.inference_sts(mode=mode)
        metrics = get_metrics(dev_rets['sys_scores'], dev_rets['gold_scores'], dev_rets['tags'])
        dev_rets['prints'] = dev_rets['prints'] + [f'{k}: {v}' for k, v in metrics.items()]
        model.append_information_file(dev_rets['prints'])
    # rets = cls.single_eval(model_tag[0])
    print(model.information_file)



'''
                                     svd      pearson-wmean     spearman-wmean        pearso-mean     spearman-wmean
                               STS12-all              80.78              79.21              77.63              75.84
                               STS13-all              90.25              89.25              84.06              83.16
                               STS14-all              93.14              91.65              94.02              92.73
                               STS15-all              89.46              89.51              87.68              87.65
                               STS16-all              85.76              86.19              85.48              85.91
                        STSBenchmark-all              84.91              85.77                  -                  -

'''
