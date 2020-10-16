import senteval
from sentence_transformers import SentenceTransformer
import pickle
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np

import torch
from typing import Dict, List
from overrides import overrides

from elmoformanylangs import Embedder

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.elmo import Elmo, batch_to_ids


# def _make_bos_eos(
#     character: int,
#     padding_character: int,
#     beginning_of_word_character: int,
#     end_of_word_character: int,
#     max_word_length: int,
# ):
#     char_ids = [padding_character] * max_word_length
#     char_ids[0] = beginning_of_word_character
#     char_ids[1] = character
#     char_ids[2] = end_of_word_character
#     return char_ids


# class CustomELMoCharacterMapper:
#     """
#     Maps individual tokens to sequences of character ids, compatible with ELMo.
#     To be consistent with previously trained models, we include it here as special of existing
#     character indexers.
#     We allow to add optional additional special tokens with designated
#     character ids with ``tokens_to_add``.
#     """
#     max_word_length = 50
#
#     def __init__(self, tokens_to_add: Dict[str, int] = None) -> None:
#         self.tokens_to_add = tokens_to_add or {}
#
#         # setting special token
#         self.beginning_of_sentence_character = self.tokens_to_add['<bos>']  # <begin sentence>
#         self.end_of_sentence_character = self.tokens_to_add['<eos>']  # <end sentence>
#         self.beginning_of_word_character = self.tokens_to_add['<bow>']  # <begin word>
#         self.end_of_word_character = self.tokens_to_add['<eow>']  # <end word>
#         self.padding_character = self.tokens_to_add['<pad>']  # <padding>
#         self.oov_character = self.tokens_to_add['<oov>']
#
#         self.max_word_length = 50
#
#         # char ids 0-255 come from utf-8 encoding bytes
#         # assign 256-300 to special chars
#
#         self.beginning_of_sentence_characters = _make_bos_eos(
#             self.beginning_of_sentence_character,
#             self.padding_character,
#             self.beginning_of_word_character,
#             self.end_of_word_character,
#             self.max_word_length,
#         )
#         self.end_of_sentence_characters = _make_bos_eos(
#             self.end_of_sentence_character,
#             self.padding_character,
#             self.beginning_of_word_character,
#             self.end_of_word_character,
#             self.max_word_length,
#         )
#
#         self.bos_token = "<bos>"
#         self.eos_token = "<eos>"
#
#     def convert_word_to_char_ids(self, word: str) -> List[int]:
#         if word in self.tokens_to_add:
#             char_ids = [self.padding_character] * self.max_word_length
#             char_ids[0] = self.beginning_of_word_character
#             char_ids[1] = self.tokens_to_add[word]
#             char_ids[2] = self.end_of_word_character
#         elif word == self.bos_token:
#             char_ids = self.beginning_of_sentence_characters
#         elif word == self.eos_token:
#             char_ids = self.end_of_sentence_characters
#         else:
#             word = word[: (self.max_word_length - 2)]
#             char_ids = [self.padding_character] * self.max_word_length
#             char_ids[0] = self.beginning_of_word_character
#             for k, char in enumerate(word, start=1):
#                 char_ids[k] = self.tokens_to_add[char] if char in self.tokens_to_add else self.oov_character
#             char_ids[len(word) + 1] = self.end_of_word_character
#
#         # +1 one for masking
#         # return [c + 1 for c in char_ids]
#         return char_ids
#
#     def __eq__(self, other) -> bool:
#         if isinstance(self, other.__class__):
#             return self.__dict__ == other.__dict__
#         return NotImplemented
#
#
# @TokenIndexer.register("custom_elmo_characters")
# class CustomELMoTokenCharactersIndexer(TokenIndexer[List[int]]):
#     """
#     Convert a token to an array of character ids to compute ELMo representations.
#     Parameters
#     ----------
#     namespace : ``str``, optional (default=``elmo_characters``)
#     tokens_to_add : ``Dict[str, int]``, optional (default=``None``)
#         If not None, then provides a mapping of special tokens to character
#         ids. When using pre-trained models, then the character id must be
#         less then 261, and we recommend using un-used ids (e.g. 1-32).
#     token_min_padding_length : ``int``, optional (default=``0``)
#         See :class:`TokenIndexer`.
#     """
#
#     def __init__(
#         self,
#         namespace: str = "elmo_characters",
#         tokens_to_add: Dict[str, int] = None,
#         token_min_padding_length: int = 0,
#     ) -> None:
#         super().__init__(token_min_padding_length)
#         self._namespace = namespace
#         self._mapper = CustomELMoCharacterMapper(tokens_to_add)
#
#     @overrides
#     def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
#         pass
#
#     @overrides
#     def tokens_to_indices(
#         self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
#     ) -> Dict[str, List[List[int]]]:
#         # TODO(brendanr): Retain the token to index mappings in the vocabulary and remove this
#
#         # https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/wordpiece_indexer.py#L113
#
#         texts = [token for token in tokens]
#
#         if any(text is None for text in texts):
#             raise ConfigurationError(
#                 "ELMoTokenCharactersIndexer needs a tokenizer " "that retains text"
#             )
#         return {index_name: [self._mapper.convert_word_to_char_ids(text) for text in texts]}
#
#     @overrides
#     def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
#         return {}
#
#     @staticmethod
#     def _default_value_for_padding():
#         return [0] * CustomELMoCharacterMapper.max_word_length
#
#     @overrides
#     def as_padded_tensor(
#         self,
#         tokens: Dict[str, List[List[int]]],
#         desired_num_tokens: Dict[str, int],
#         padding_lengths: Dict[str, int],
#     ) -> Dict[str, torch.Tensor]:
#         return {
#             key: torch.LongTensor(
#                 pad_sequence_to_length(
#                     val, desired_num_tokens[key], default_value=self._default_value_for_padding
#                 )
#             )
#             for key, val in tokens.items()
#         }


class ElmoModel:
    def __init__(self, device='cpu', elmo_with="allennlp"):
        self.device = device
        self.elmo_with = elmo_with
        self.elmo_embeddings = None
        if self.elmo_with == "allennlp":
            if Path('../models/sentence_embeddings_elmo_allennlp.pkl').exists():
                with Path('../models/sentence_embeddings_elmo_allennlp.pkl').open('wb') as f:
                    self.elmo_embeddings = pickle.load(f)
            root_path = Path('../models/elmo')
            option_file = root_path / 'config.json'
            weight_file = root_path / 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            self.num_output_representations = 2
            self.embedding = Elmo(option_file, weight_file, num_output_representations=self.num_output_representations).to(device)
            self.embedding_dim = 1024 * self.num_output_representations
            self.vocab = Vocabulary()
        else:
            root_path = Path('../models/elmo')
            self.embedding = Embedder(root_path)
            if str(device) == 'cpu':
                self.embedding.use_cuda = False
                self.embedding.model.use_cuda = False
                self.embedding.model.encoder.use_cuda = False
                self.embedding.model.token_embedder.use_cuda = False
                self.embedding.model.to(device)
            else:
                self.embedding.use_cuda = True
                self.embedding.model.use_cuda = True
                self.embedding.model.encoder.use_cuda = True
                self.embedding.model.token_embedder.use_cuda = True
                self.embedding.model.to(device)
            self.embedding_dim = self.embedding.config['encoder']['projection_dim'] * 2

    def get_word_embedding(self, batch_sentences):
        rets = []
        if self.elmo_with == "allennlp":
            for sentence in batch_sentences:
                indexes = batch_to_ids(sentence)
                embedding = self.embedding(torch.LongTensor(indexes).to(self.device))
                rets.append(torch.mean(torch.mean(torch.cat((embedding['elmo_representations'][0], embedding['elmo_representations'][1]), dim=2), dim=0), dim=0).tolist())
        else:
            rets = self.embedding.sents2elmo(batch_sentences)
            rets = torch.tensor(rets)
        rets = np.array(rets)
        return rets

    def state_dict(self):
        return self.embedding.state_dict()


model_names = ['elmo-allennlp']

embeddings = {model_name: {} for model_name in model_names}


def prepare(params, samples):
    return


def batcher(params, batch):
    # sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
    sentence_embeddings = model.get_word_embedding(batch)   # get sentence embeddings
    for sentence, sentence_embedding in zip(batch, sentence_embeddings):
        embeddings[model_name][' '.join(sentence)] = sentence_embedding
    return sentence_embeddings


for model_name in model_names:
    model = ElmoModel(elmo_with='allennlp')
    params = {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'usepytorch': True}
    params['encoder'] = model

    se = senteval.engine.SE(params, batcher)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
    results = se.eval(transfer_tasks)

    print_header = [model_name, 'pearson-r', 'peason-p_val', 'spearman-r', 'spearman-p_val', 'n_samples']
    print_contents = [print_header]

    print_all_header = [model_name, 'pearson-wmean', 'spearman-wmean', 'pearso-mean', 'spearman-wmean']
    print_all_contents = [print_all_header]

    for task in results:
        print_content = []
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
                    print_contents.append([f'{task}-{category}',
                                           f'{Decimal(str(results[task][category]["pearson"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP) :.2f}',
                                           f'{results[task][category]["pearson"][1]}',
                                           f'{Decimal(str(results[task][category]["spearman"][0] * 100)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP):.2f}',
                                           f'{results[task][category]["spearman"][1]}',
                                           f'{results[task][category]["nsamples"]}'])

    with open("../results/results.single.201008.txt", "a") as f:
        for print_all_content in print_all_contents:
            print(" ".join(["{: >40}"] + ["{: >18}"] * (len(print_all_header) - 1)).format(*print_all_content), file=f)

        print("", file=f)

        for print_content in print_contents:
            print(" ".join(["{: >40}"] + ["{: >25}"] * (len(print_header) - 2) + ["{: >10}"]).format(*print_content), file=f)

        print("", file=f)
        print("", file=f)

    with_save_embeddings = True
    if with_save_embeddings:
        with open(f'../models/sentence_embeddings_{model_name}.pkl', 'wb') as f:
            pickle.dump(embeddings[model_name], f)


'''


'''
