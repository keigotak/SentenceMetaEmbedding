import pickle
from pathlib import Path

model_names = ['roberta-large-nli-stsb-mean-tokens',
               'roberta-base-nli-stsb-mean-tokens',
               'bert-large-nli-stsb-mean-tokens',
               'distilbert-base-nli-stsb-mean-tokens',
               'use']

sentence_indexer = {}

for model_name in model_names:
    sentence_id = 0
    with open(f'../models/sentence_embeddings_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    with Path(f'./ae/sentence_embeddings_{model_name}.txt').open('w') as f_model:
        # f_model.write(f'{len(model)} {len(model[list(model.keys())[0]])}\n')

        for key in model.keys():
            if key not in sentence_indexer.keys():
                sentence_indexer[key] = sentence_id

                with Path(f'./ae/sentence_embeddings_indexer.txt').open('a') as f_indexer:
                    f_indexer.write(f'SID{sentence_id}\t{key}\n')

                f_model.write(f'SID{sentence_id} {" ".join(map(str, model[key]))}\n')
                sentence_id += 1
            else:
                f_model.write(f'SID{sentence_indexer[key]} {" ".join(map(str, model[key]))}\n')
