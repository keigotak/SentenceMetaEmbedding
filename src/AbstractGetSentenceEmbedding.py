import os
import pickle
from decimal import Decimal, ROUND_HALF_UP

import torch

import senteval


class AbstractGetSentenceEmbedding:
    def __init__(self):
        self.model_names = None
        self.embeddings = None
        self.model = None
        self.output_file_name = 'results.txt'
        self.with_detailed_log = False
        self.with_reset_output_file = False
        self.with_save_embeddings = False

    def get_model(self):
        pass

    def batcher(self, params, batch):
        return

    def eval(self):
        for model_name in self.model_names:
            self.single_eval(model_name=model_name)

    def get_params(self):
        # return {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'usepytorch': True, 'batch_size': 10000}
        return {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'batch_size': 10000}

    def single_eval(self, model_name):
        self.model = self.get_model()
        params = self.get_params()
        params['encoder'] = self.model

        se = senteval.engine.SE(params, self.batcher)
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
        # transfer_tasks = ['STS12']
        # transfer_tasks = ['STSBenchmark']
        results = se.eval(transfer_tasks)

        print_header = [model_name, 'pearson-r', 'peason-p_val', 'spearman-r', 'spearman-p_val', 'n_samples']
        print_contents = [print_header]

        print_all_header = [model_name, 'pearson-wmean', 'spearman-wmean', 'pearso-mean', 'spearman-wmean']
        print_all_contents = [print_all_header]

        for task in results:
            if task == 'STSBenchmark':
                print_all_contents.append([f'{task}-all',
                                           f'{Decimal(str(results[task]["pearson"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                           f'{Decimal(str(results[task]["spearman"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                           '-',
                                           '-'])
            else:
                for category in results[task].keys():
                    if category == 'all':
                        print_all_contents.append([f'{task}-{category}',
                                                   f'{Decimal(str(results[task][category]["pearson"]["wmean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["spearman"]["wmean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["pearson"]["mean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{Decimal(str(results[task][category]["spearman"]["mean"] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}'])
                    else:
                        if self.with_detailed_log:
                            print_contents.append([f'{task}-{category}',
                                                   f'{Decimal(str(results[task][category]["pearson"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                                   f'{results[task][category]["pearson"][1]}',
                                                   f'{Decimal(str(results[task][category]["spearman"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP):.2f}',
                                                   f'{results[task][category]["spearman"][1]}',
                                                   f'{results[task][category]["nsamples"]}'])

        if self.with_reset_output_file:
            if os.path.exists(f'../results/{self.output_file_name}'):
                os.remove(f'../results/{self.output_file_name}')

        texts = []
        with open(f'../results/{self.output_file_name}', 'a') as f:
            for print_all_content in print_all_contents:
                text = ' '.join(['{: >40}'] + ['{: >18}'] * (len(print_all_header) - 1)).format(*print_all_content)
                print(text, file=f)
                print(text)
                texts.append(text)

            if self.with_detailed_log:
                print('', file=f)

                for print_content in print_contents:
                    print(' '.join(['{: >40}'] + ['{: >25}'] * (len(print_header) - 2) + ['{: >10}']).format(
                        *print_content), file=f)

            print('', file=f)
            print('', file=f)

        if self.with_save_embeddings:
            with open(f'../models/sentence_embeddings_{model_name}.pkl', 'wb') as f:
                pickle.dump(self.embeddings[model_name], f)

        return {'text': texts, 'pearson': results[task]["pearson"], 'spearman': results[task]["spearman"]}

    def modify_batch_sentences_for_senteval(self, batch_words):
        padded_sequences, padding_masks = {}, {}
        # for words in batch_words:
        #     items = []
        #     # words = ['for', 'the', 'third', 'time', 'in', 'four', 'years', 'wildfires', 'closed', 'mesa', 'verde', 'national', 'park', ',', 'the', 'country', '’', 's', 'only', 'park', 'dedicated', 'to', 'ancient', 'ruins', '.']
        #     # words = ['"', 'i', '´', 'm', 'very', 'proud', 'of', 'the', 'citizens', 'of', 'this', 'state', ',', '"', 'gov.', 'john', 'baldacci', 'said', 'after', 'votes', 'from', 'tuesday', '´', 's', 'referendum', 'were', 'counted', '.']
        #     # words = ['on', 'saturday', ',', 'a', '149mph', 'serve', 'against', 'agassi', 'equalled', 'rusedski', "'s", 'world', 'record', '.']
        #     # words = ['the', 'settlement', 'includes', '$', '4.1', 'million', 'in', 'attorneys', "'", 'fees', 'and', 'expenses', '.']
        #     # words = ['the', 'president', 'has', 'less', 'of', 'capacities', 'than', 'it', 'does', 'not', 'appear', 'to', 'with', 'it', ',', 'and', 'it', 'particuli', '�', 'rement', 'is', 'particuli', '�', 'rement', 'eclipsed', 'by', 'the', 'guide', 'supr', '�', 'me', 'the', 'ayatollah', 'ali', 'khamenei', '.']
        #     # words = ['the', 'old', 'version', 'of', 'the', 'european', 'reaction', '(', 'what', 'the', 'psychologists', 'call', '“', 'the', 'desire', 'of', 'the', 'dollar', '”', ')', 'will', 'become', 'only', 'more', 'penetrating', '.']
        #     words = ['there', 'is', 'a', 'certain', 'physical', 'distance', 'within', 'which', 'some', 'other', 'entity', 'can', 'participate', 'in', 'an', 'event', '(', 'typically', 'perception', 'or', 'manipulation', ')', 'with', 'the', 'participant.', 'alternatively', ',', 'the', 'event', 'may', 'be', 'indicated', 'metonymically', 'by', 'a', 'instrument', '.', '[', 'note', 'the', 'connection', 'with', '*', 'distance', ',', 'sufficiency', ',', 'and', 'capability.', 'words', 'in', 'this', 'frame', 'can', 'generally', 'be', 'paraphrased', '"', 'close', 'enough', 'to', 'be', 'able', 'to', '"', '.', ']']
        #
        #     words = [w for w in words if w != '�']
        #     for model_name in self.model_names:
        #         item = self.model.source[model_name].get_word_embedding(' '.join(words))
        #         items.append(item['embeddings'][0])
        #         assert len(item['ids']) == len(item['embeddings'][0])
        #     print(f'{len(items[0])}, {len(items[1])}, {len(items[2])}')

        for model_name in self.model_names:
            items = []
            if model_name == 'glove':
                items = self.model.source[model_name].get_word_embedding(batch_words)
                items = [torch.FloatTensor(item) for item in items]
            else:
                for words in batch_words:
                    words = [w for w in words if w != '�']
                    # print(words)
                    item = self.model.source[model_name].get_word_embedding(' '.join(words))
                    items.append(torch.FloatTensor(item['embeddings'][0]))
            padded_sequences[model_name] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True).to(self.device)
            # padded_sequences[model_name] = torch.nn.utils.rnn.pad_sequence([torch.FloatTensor(items['embeddings'][0][1:-1])
            #                                      for items in [self.model.source[model_name].get_word_embedding(' '.join(words))
            #                                                    for words in batch_words]], batch_first=True).to(self.device)

            max_sentence_length = max([len(words) for words in batch_words])
            padding_masks[model_name] = torch.BoolTensor([[[False] * len(words) + [True] * (max_sentence_length - len(words)) ] for words in batch_words]).squeeze(1).to(self.device)

        max_sentence_length = [padded_sequences[k].shape[1] for k in padded_sequences.keys()]
        padded_sequences[model_name] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True).to(self.device)
        return padded_sequences, padding_masks


def prepare(params, samples):
    return



