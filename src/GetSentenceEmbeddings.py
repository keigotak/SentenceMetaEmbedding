import senteval
from sentence_transformers import SentenceTransformer
import pickle
from decimal import Decimal, ROUND_HALF_UP


model_names = ['roberta-large-nli-stsb-mean-tokens',
               'roberta-base-nli-stsb-mean-tokens',
               'bert-large-nli-stsb-mean-tokens',
               'distilbert-base-nli-stsb-mean-tokens']

embeddings = {model_name: {} for model_name in model_names}


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words
    sentence_embeddings = model.encode(sentences)   # get sentence embeddings
    for sentence, sentence_embedding in zip(batch, sentence_embeddings):
        embeddings[model_name][' '.join(sentence)] = sentence_embedding
    return sentence_embeddings


for model_name in model_names:
    model = SentenceTransformer(model_name)
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

    for print_all_content in print_all_contents:
        print(" ".join(["{: >40}"] + ["{: >18}"] * (len(print_all_header) - 1)).format(*print_all_content))

    print("")

    for print_content in print_contents:
        print(" ".join(["{: >40}"] + ["{: >25}"] * (len(print_header) - 2) + ["{: >10}"]).format(*print_content))

    with_save_embeddings = False
    if with_save_embeddings:
        with open(f'../models/sentence_embeddings_{model_name}.pkl', 'wb') as f:
            pickle.dump(embeddings[model_name], f)


'''


'''
