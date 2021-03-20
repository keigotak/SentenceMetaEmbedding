import pickle
from pathlib import Path

from HelperFunctions import get_now
from GCCA import GCCA


# model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_distilbert-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl"]
model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
              "../models/sentence_embeddings_use.pkl"]

embeddings = {}
for model_pkl in model_pkls:
    with Path(model_pkl).open('rb') as f:
        embedding = pickle.load(f)
    embeddings[model_pkl] = embedding

sentences = embedding.keys()

vectors = [[] for _ in range(len(model_pkls))]
key_to_index = {key: i for i, key in enumerate(embeddings.keys())}
sentence_to_index = {}
for i, sentence in enumerate(sentences):
    vector = []
    for model_type in embeddings.keys():
        tmp = embeddings[model_type][sentence]
        if type(embeddings[model_type][sentence]) is not list:
            vectors[key_to_index[model_type]].append(embeddings[model_type][sentence].tolist())
        else:
            vectors[key_to_index[model_type]].append(embeddings[model_type][sentence])
    sentence_to_index[sentence] = i

#
# a = np.array(vectors[0])
# b = np.array(vectors[1])
# c = np.array(vectors[2])
# d = np.array(vectors[3])
#
# # create instance of GCCA
# gcca = GCCA()
# # calculate GCCA
# gcca.fit(a, b, c, d)
# # transform
# gcca.transform(a, b, c, d)
# # save
# gcca.save_params("../models/sts_gcca.h5")
# # load
# gcca.load_params("../models/sts_gcca.h5")
# # plot
# gcca.plot_result()

# a = np.array([1., 0.])
# b = np.array([0., 1.])

tag = get_now()

gcca = GCCA(tag=tag)
gcca.prepare(vectors)
gcca.fit(vectors)
gcca.save_model()
gcca_vectors = gcca.transform(vectors)
with Path(f'../models/gcca_{tag}.pkl').open('wb') as f:
    pickle.dump(gcca_vectors, f)
with Path(f'../models/gcca_{tag}_sentence_indexer.pkl').open('wb') as f:
    pickle.dump(sentence_to_index, f)


