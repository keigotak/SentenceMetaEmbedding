# https://github.com/rjadr/GCCA
import pickle
from pathlib import Path
from gcca.gcca import GCCA
import logging
import numpy as np

# set log level
# logging.root.setLevel(level=logging.INFO)
#
# model_pkls = ["../models/sentence_embeddings_bert-large-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_distilbert-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-base-nli-stsb-mean-tokens.pkl",
#               "../models/sentence_embeddings_roberta-large-nli-stsb-mean-tokens.pkl"]
#
# embeddings = {}
# for model_pkl in model_pkls:
#     with Path(model_pkl).open('rb') as f:
#         embedding = pickle.load(f)
#     embeddings[model_pkl] = embedding
#
# sentences = embedding.keys()
#
# vectors = [[] for _ in range(len(model_pkls))]
# key_to_index = {key: i for i, key in enumerate(embeddings.keys())}
# for sentence in sentences:
#     vector = []
#     for model_type in embeddings.keys():
#         vectors[key_to_index[model_type]].append(embeddings[model_type][sentence].tolist())
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

a = np.array([1., 0.])
b = np.array([0., 1.])
gcca = GCCA()
gcca.fit(a, b)
gcca.transform(a, b)
gcca.plot_result()


