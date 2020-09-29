import senteval
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
params = {'task_path': '/home/keigo/SentEval/data', 'usepytorch': True, 'kfold': 5}


def batcher(params, batch):
    sentences = [' '.join(b) for b in batch]
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


se = senteval.engine.SE(params, batcher)
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
results = se.eval(transfer_tasks)

