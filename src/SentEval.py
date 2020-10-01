import senteval
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
params = {'task_path': '/home/keigo/SentEval/data', 'usepytorch': True}
params['encoder'] = model


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch] # To reconstruct sentence from list of words
    sentence_embeddings = model.encode(sentences) # get sentence embeddings
    return sentence_embeddings


se = senteval.engine.SE(params, batcher)
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
# transfer_tasks = ['STS12']
results = se.eval(transfer_tasks)
for key in results:
    print(f'{key}: {results[key]}')


'''
* roberta-large-nli-stsb-mean-tokens
STS12: {'MSRpar': {'pearson': (0.8973911271090574, 5.315560886276807e-268), 'spearman': SpearmanrResult(correlation=0.881623508075305, pvalue=3.943850201678884e-246), 'nsamples': 750}, 'MSRvid': {'pearson': (0.9639753236498194, 0.0), 'spearman': SpearmanrResult(correlation=0.9633191861749041, pvalue=0.0), 'nsamples': 750}, 'SMTeuroparl': {'pearson': (0.4884410104404851, 6.789098462619042e-29), 'spearman': SpearmanrResult(correlation=0.5561839848565967, pvalue=1.2480790452965277e-38), 'nsamples': 459}, 'surprise.OnWN': {'pearson': (0.7446848285720652, 1.7426761993477275e-133), 'spearman': SpearmanrResult(correlation=0.7107226892966618, pvalue=2.2559503030582753e-116), 'nsamples': 750}, 'surprise.SMTnews': {'pearson': (0.646842044559346, 1.1674265715488382e-48), 'spearman': SpearmanrResult(correlation=0.5765179167583552, pvalue=1.0216173549559633e-36), 'nsamples': 399}, 'all': {'pearson': {'mean': 0.7482668668661546, 'wmean': 0.7840485389541725}, 'spearman': {'mean': 0.7376734570323645, 'wmean': 0.772866195462006}}}
STS13: {'FNWN': {'pearson': (0.6184379142188333, 2.496520970119092e-21), 'spearman': SpearmanrResult(correlation=0.6088247998217468, pvalue=1.487353062969793e-20), 'nsamples': 189}, 'headlines': {'pearson': (0.9564516143192449, 0.0), 'spearman': SpearmanrResult(correlation=0.9569321731787693, pvalue=0.0), 'nsamples': 750}, 'OnWN': {'pearson': (0.8799552184606487, 7.656491255110629e-183), 'spearman': SpearmanrResult(correlation=0.8534655040406516, pvalue=2.3901801861172458e-160), 'nsamples': 561}, 'all': {'pearson': {'mean': 0.8182815823329089, 'wmean': 0.885252236055478}, 'spearman': {'mean': 0.806407492347056, 'wmean': 0.8743741098781284}}}
STS14: {'deft-forum': {'pearson': (0.9705204682870285, 5.5941923248452966e-279), 'spearman': SpearmanrResult(correlation=0.9722865757268025, pvalue=6.66953430508889e-285), 'nsamples': 450}, 'deft-news': {'pearson': (0.9754975862017035, 5.285528428819551e-198), 'spearman': SpearmanrResult(correlation=0.963724164433573, pvalue=5.399981994600007e-173), 'nsamples': 300}, 'headlines': {'pearson': (0.9479710244198108, 0.0), 'spearman': SpearmanrResult(correlation=0.9522455097970969, pvalue=0.0), 'nsamples': 750}, 'images': {'pearson': (0.956732678730368, 0.0), 'spearman': SpearmanrResult(correlation=0.9379467968146444, pvalue=0.0), 'nsamples': 750}, 'OnWN': {'pearson': (0.899474576338094, 3.722487155883676e-271), 'spearman': SpearmanrResult(correlation=0.8755722224959163, pvalue=1.491079969278812e-238), 'nsamples': 750}, 'tweet-news': {'pearson': (0.8216065118500188, 9.454883307533972e-185), 'spearman': SpearmanrResult(correlation=0.7837137282345785, pvalue=7.358014013913027e-157), 'nsamples': 750}, 'all': {'pearson': {'mean': 0.9286338076378372, 'wmean': 0.919659221358238}, 'spearman': {'mean': 0.9142481662504354, 'wmean': 0.9036679737103493}}}
STS15: {'answers-forums': {'pearson': (0.7426104182022147, 6.355775335757287e-67), 'spearman': SpearmanrResult(correlation=0.7372141270510836, pvalue=1.7205186166559247e-65), 'nsamples': 375}, 'answers-students': {'pearson': (0.8171325178916032, 3.9985068716900657e-181), 'spearman': SpearmanrResult(correlation=0.8228322500763348, pvalue=9.212778137843523e-186), 'nsamples': 750}, 'belief': {'pearson': (0.817343969607753, 2.4131563374512546e-91), 'spearman': SpearmanrResult(correlation=0.8154056000919012, pvalue=1.4196555304584117e-90), 'nsamples': 375}, 'headlines': {'pearson': (0.9598933425651655, 0.0), 'spearman': SpearmanrResult(correlation=0.9606857289994915, pvalue=0.0), 'nsamples': 750}, 'images': {'pearson': (0.9631024534187883, 0.0), 'spearman': SpearmanrResult(correlation=0.9637175945621475, pvalue=0.0), 'nsamples': 750}, 'all': {'pearson': {'mean': 0.860016540337105, 'wmean': 0.8800263769451352}, 'spearman': {'mean': 0.8599710601561916, 'wmean': 0.8808863593023665}}}
STS16: {'answer-answer': {'pearson': (0.8060611905376801, 2.44100897759137e-59), 'spearman': SpearmanrResult(correlation=0.8016022145987944, pvalue=3.1518507871827125e-58), 'nsamples': 254}, 'headlines': {'pearson': (0.9468547394367105, 1.1382342386112322e-123), 'spearman': SpearmanrResult(correlation=0.9504968297047918, pvalue=2.2239778887238582e-127), 'nsamples': 249}, 'plagiarism': {'pearson': (0.851639070554015, 6.658540574834816e-66), 'spearman': SpearmanrResult(correlation=0.857061369868074, pvalue=1.324506734338106e-67), 'nsamples': 230}, 'postediting': {'pearson': (0.8806044352861144, 1.8614554606802694e-80), 'spearman': SpearmanrResult(correlation=0.8988682979989215, pvalue=1.1084268093520465e-88), 'nsamples': 244}, 'question-question': {'pearson': (0.7340955482365981, 1.1997087110180896e-36), 'spearman': SpearmanrResult(correlation=0.7318654980214938, pvalue=2.4999753998060168e-36), 'nsamples': 209}, 'all': {'pearson': {'mean': 0.8438509968102237, 'wmean': 0.8471136682419865}, 'spearman': {'mean': 0.847978842038415, 'wmean': 0.8513394114439065}}}
STSBenchmark: {'devpearson': 0.8391156420465512, 'pearson': 0.8268143088392823, 'spearman': 0.8343727676572694, 'mse': 0.8275295085212143, 'yhat': array([1.77155968, 1.92480455, 2.022122  , ..., 4.41291753, 4.05387761,
       4.02281437]), 'ndev': 1500, 'ntest': 1379}
'''
