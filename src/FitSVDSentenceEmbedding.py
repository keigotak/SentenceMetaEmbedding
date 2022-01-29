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
model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
              "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl"]
# model_pkls = ["../models/sentence_embeddings_stsb-bert-large.pkl",
#               "../models/sentence_embeddings_stsb-distilbert-base.pkl",
#               "../models/sentence_embeddings_stsb-roberta-large.pkl"]
# model_pkls = ["../models/sentence_embeddings_stsb-bert-large.pkl",
#               "../models/sentence_embeddings_stsb-distilbert-base.pkl"]

embeddings = {}
for model_pkl in model_pkls:
    with Path(model_pkl).open('rb') as f:
        embedding = pickle.load(f)
    embeddings[model_pkl] = embedding

sentences = embedding.keys()
embedding_dims = [len(list(embeddings[model_pkl].values())[0]) for model_pkl in model_pkls]

vectors = [torch.zeros([len(embeddings[model_pkls[0]]), embedding_dim], dtype=torch.float64) for embedding_dim in embedding_dims]
# vectors = [torch.zeros([4, embedding_dim], dtype=torch.float64) for embedding_dim in embedding_dims]

# print([v.shape for v in vectors])
# vectors = [[] for _ in range(len(model_pkls))]
key_to_index = {key: i for i, key in enumerate(embeddings.keys())}
sentence_to_index = {}
for i, sentence in enumerate(sentences):
    for model_type in embeddings.keys():
        tmp = embeddings[model_type][sentence]
        vectors[key_to_index[model_type]][i] = torch.DoubleTensor(tmp)
        # if type(embeddings[model_type][sentence]) is not list:
        #     vectors[key_to_index[model_type]].append(embeddings[model_type][sentence].tolist())
        # else:
        #     vectors[key_to_index[model_type]].append(embeddings[model_type][sentence])
    sentence_to_index[sentence] = i
    # if i >= 3:
    #     break

del embedding, embeddings, f

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


