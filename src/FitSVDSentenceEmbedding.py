import pickle
from pathlib import Path
import torch

from HelperFunctions import get_now
from SVD import SVD


# model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_distilbert-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_use.pkl"]
# model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl"]
# model_pkls = ["../models/sentence_embeddings_stsb-bert-large.pkl",
#               "../models/sentence_embeddings_stsb-distilbert-base.pkl",
#               "../models/sentence_embeddings_stsb-roberta-large.pkl"]
# model_pkls = ["../models/sentence_embeddings_stsb-bert-large.pkl",
#               "../models/sentence_embeddings_stsb-distilbert-base.pkl"]
model_pkls = [
    './stsb-bert-large_avg.pt',
    './stsb-distilbert-base_avg.pt',
    './stsb-mpnet-base-v2_avg.pt',
    './stsb-roberta-large_avg.pt'
]

embeddings = {}
for model_pkl in model_pkls:
    with Path(model_pkl).open('rb') as f:
        embeddings[model_pkl] = torch.load(f)

sentences = embeddings[model_pkl].keys()
embedding_dims = [len(list(embeddings[model_pkl].values())[0]) for model_pkl in model_pkls]
# print(embedding_dims)

# vectors = [torch.zeros([len(embeddings[model_pkls[0]]), embedding_dim], dtype=torch.float64) for embedding_dim in embedding_dims]
vectors = [[] for _ in range(len(model_pkls))]
key_to_index = {key: i for i, key in enumerate(embeddings.keys())}
sentence_to_index = {}
for i, sentence in enumerate(sentences):
    vector = []
    for model_type in embeddings.keys():
        tmp = embeddings[model_type][sentence]['embeddings']
        # print(torch.DoubleTensor(tmp).shape)
        # print(torch.mean(torch.DoubleTensor(tmp), dim=1).shape)
        # vectors[key_to_index[model_type]][i] = torch.mean(torch.DoubleTensor(tmp), dim=1).squeeze(0)
        vectors[key_to_index[model_type]].append(torch.mean(torch.DoubleTensor(tmp), dim=1).squeeze(0).tolist())
        # if type(embeddings[model_type][sentence]['embeddings']) is not list:
        #     vectors[key_to_index[model_type]].append(np.mean(embeddings[model_type][sentence]['embeddings'], axis=1)[0].tolist())
        # else:
        #     vectors[key_to_index[model_type]].append(np.mean(embeddings[model_type][sentence]['embeddings'], axis=1)[0])
    sentence_to_index[sentence] = i

del embeddings, f

tag = get_now()
print(tag)

svd = SVD(tag=tag)
svd.prepare(vectors)
svd.fit(vectors)
svd.save_model()
svd.load_model()
svd_vectors = svd.transform(vectors)
with Path(f'../models/svd_{tag}.pkl').open('wb') as f:
    pickle.dump(svd_vectors, f)
with Path(f'../models/svd_{tag}_sentence_indexer.pkl').open('wb') as f:
    pickle.dump(sentence_to_index, f)


