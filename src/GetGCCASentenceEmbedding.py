import senteval
import pickle
from decimal import Decimal, ROUND_HALF_UP
from gcca.gcca import GCCA
import numpy as np


model_names = ['bert-large-nli-stsb-mean-tokens',
               'distilbert-base-nli-stsb-mean-tokens',
               'roberta-base-nli-stsb-mean-tokens',
               'roberta-large-nli-stsb-mean-tokens']

gcca_model = GCCA()
gcca_model.load_params("../models/sts_gcca.h5")
sentence_embeddings = {}
for model_name in model_names:
    with open(f'../models/sentence_embeddings_{model_name}.pkl', 'rb') as f:
        sentence_embeddings[model_name] = pickle.load(f)


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]  # To reconstruct sentence from list of words

    # gcca_embeddings = []
    alist, blist, clist, dlist = [], [], [], []
    for sentence in sentences:
        alist.append(sentence_embeddings[model_names[0]][sentence].tolist())
        blist.append(sentence_embeddings[model_names[1]][sentence].tolist())
        clist.append(sentence_embeddings[model_names[2]][sentence].tolist())
        dlist.append(sentence_embeddings[model_names[3]][sentence].tolist())

    alist, blist, clist, dlist = np.array(alist), np.array(blist), np.array(clist), np.array(dlist)

    gcca_embeddings = gcca_model.transform(alist, blist, clist, dlist)

    return gcca_embeddings[3]


model_name = 'gcca'
params = {'task_path': '/clwork/keigo/SentenceMetaEmbedding/data', 'usepytorch': True}
params['encoder'] = gcca_model

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

with open("../results/results.gcca.201012.txt", "a") as f:
    for print_all_content in print_all_contents:
        print(" ".join(["{: >40}"] + ["{: >18}"] * (len(print_all_header) - 1)).format(*print_all_content), file=f)

    print("", file=f)

    for print_content in print_contents:
        print(" ".join(["{: >40}"] + ["{: >25}"] * (len(print_header) - 2) + ["{: >10}"]).format(*print_content), file=f)

    print("", file=f)
    print("", file=f)



'''


'''
