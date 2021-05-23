import torch
import optuna

from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from TrainAttentionWithSTSBenchmark import TrainAttentionWithSTSBenchmark, EvaluateAttentionModel
from TrainVectorAttentionWithSTSBenchmark import TrainVectorAttentionWithSTSBenchmark, EvaluateVectorAttentionModel
from TrainSeq2seqWithSTSBenchmark import TrainSeq2seqWithSTSBenchmark, EvaluateSeq2seqModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='select device')
parser.add_argument('--model', type=str, default='att')
args = parser.parse_args()

if args.device != 'cpu':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def objective_a(trial):
    dp = DataPooler()
    # es_metrics = trial.suggest_categorical('es_metrics', ['dev_loss', 'pearson'])
    es_metrics = 'pearson'
    if es_metrics == 'dev_loss':
        vw = ValueWatcher(mode='minimize')
    else:
        vw = ValueWatcher(threshold=1)
    cls = EvaluateAttentionModel(device=args.device)

    hyper_params = {}
    hyper_params['dropout_ratio'] = trial.suggest_float('dropout_ratio', 0.0, 1.0)
    hyper_params['source_pooling_method'] = trial.suggest_categorical('source_pooling_method', ['avg', 'concat'])
    hyper_params['sentence_pooling_method'] = trial.suggest_categorical('sentence_pooling_method', ['avg', 'max'])
    hyper_params['learning_ratio'] = trial.suggest_float('learning_ratio', 1e-5, 1e-1, log=True)
    hyper_params['gradient_clip'] = trial.suggest_float('gradient_clip', 1e-1, 10.0, log=True)
    hyper_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    hyper_params['batch_size'] = trial.suggest_int('batch_size', 128, 512, step=2)
    cls.model.update_hyper_parameters(hyper_params)

    while not vw.is_over():
        print(f'epoch: {vw.epoch}')
        cls.model.train_epoch()
        cls.model.datasets['train'].reset(with_shuffle=True)
        rets = cls.model.inference(mode='dev')
        vw.update(rets[es_metrics])
        if vw.is_updated():
            cls.model.save_model()
            dp.set('best-epoch', vw.epoch)
            dp.set('best-score', vw.max_score)
        dp.set(f'scores', rets)
    print(f'dev best scores: {cls.model.get_round_score(dp.get_best("best-score")) :.2f}')

    cls.model.load_model()
    test_rets = cls.model.inference(mode='test')
    print(f'test best scores: ' + ' '.join(test_rets['prints']))
    # cls.model = trainer
    eval_rets = cls.single_eval(cls.model_tag[0])
    cls.model.append_information_file([f'es_metrics: {es_metrics}'])
    cls.model.append_information_file(eval_rets['text'])

    eval_rets['best_epoch'] = dp.get_best('best-epoch')
    eval_rets['dev_pearson'] = dp.get_best("best-score")
    eval_rets['test_pearson'] = test_rets['pearson']
    cls.save_summary_writer(eval_rets)

    del cls
    torch.cuda.empty_cache()

    return eval_rets['pearson']


def objective_va(trial):
    dp = DataPooler()
    # es_metrics = trial.suggest_categorical('es_metrics', ['dev_loss', 'pearson'])
    es_metrics = 'pearson'
    if es_metrics == 'dev_loss':
        vw = ValueWatcher(mode='minimize')
    else:
        vw = ValueWatcher()
    cls = EvaluateVectorAttentionModel(device=args.device)

    hyper_params = {}
    hyper_params['source_pooling_method'] = trial.suggest_categorical('source_pooling_method', ['avg', 'concat'])
    hyper_params['sentence_pooling_method'] = trial.suggest_categorical('sentence_pooling_method', ['avg', 'max'])
    hyper_params['learning_ratio'] = trial.suggest_float('learning_ratio', 1e-5, 1e-1, log=True)
    hyper_params['gradient_clip'] = trial.suggest_float('gradient_clip', 1e-2, 100, log=True)
    hyper_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    hyper_params['with_vector_attention'] = trial.suggest_categorical('with_vector_attention', [True, False])
    hyper_params['loss_mode'] = trial.suggest_categorical('loss_mode', ['cos', 'word'])
    hyper_params['batch_size'] = trial.suggest_int('batch_size', 128, 512, step=2)
    cls.model.update_hyper_parameters(hyper_params)

    while not vw.is_over():
        print(f'epoch: {vw.epoch}')
        cls.model.train_epoch()
        cls.model.datasets['train'].reset(with_shuffle=True)
        rets = cls.model.inference(mode='dev')
        vw.update(rets[es_metrics])
        if vw.is_updated():
            cls.model.save_model()
            dp.set('best-epoch', vw.epoch)
            dp.set('best-score', vw.max_score)
        dp.set(f'scores', rets)
    print(f'dev best scores: {cls.model.get_round_score(dp.get_best("best-score")) :.2f}')

    cls.model.load_model()
    test_rets = cls.model.inference(mode='test')
    print(f'test best scores: ' + ' '.join(test_rets['prints']))
    # cls.model = trainer
    eval_rets = cls.single_eval(cls.model_tag[0])
    cls.model.append_information_file([f'es_metrics: {es_metrics}'])
    cls.model.append_information_file(eval_rets['text'])

    eval_rets['best_epoch'] = dp.get_best('best-epoch')
    eval_rets['dev_pearson'] = dp.get_best("best-score")
    eval_rets['test_pearson'] = test_rets['pearson']
    cls.save_summary_writer(eval_rets)

    del cls
    torch.cuda.empty_cache()

    return eval_rets['pearson']


def objective_s2s(trial):
    dp = DataPooler()
    # es_metrics = trial.suggest_categorical('es_metrics', ['dev_loss', 'pearson'])
    es_metrics = 'pearson'
    if es_metrics == 'dev_loss':
        vw = ValueWatcher(mode='minimize')
    else:
        vw = ValueWatcher()
    cls = EvaluateSeq2seqModel(device=args.device)

    hyper_params = {}
    hyper_params['meta_embedding_dim'] = trial.suggest_int('batch_size', 128, 2048, step=2)
    hyper_params['activation'] = trial.suggest_categorical('activation', ['none', 'relu'])
    hyper_params['attention_dropout_ratio'] = trial.suggest_float('attention_dropout_ratio', 0.0, 1.0)
    hyper_params['sentence_pooling_method'] = trial.suggest_categorical('sentence_pooling_method', ['avg', 'max'])
    hyper_params['learning_ratio'] = trial.suggest_float('learning_ratio', 1e-5, 1e-1, log=True)
    hyper_params['gradient_clip'] = trial.suggest_float('gradient_clip', 1e-2, 100, log=True)
    hyper_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    hyper_params['lambda_e'] = trial.suggest_float('lambda_e', 1e-6, 1e-1, log=True)
    hyper_params['lambda_d'] = trial.suggest_float('lambda_d', 1e-6, 1e-1, log=True)
    hyper_params['batch_size'] = trial.suggest_int('batch_size', 16, 256, step=2)
    hyper_params['which_prime_output_to_use_in_testing'] = trial.suggest_categorical('which_prime_output_to_use_in_testing', ['encoder', 'decoder'])
    cls.model.update_hyper_parameters(hyper_params)

    while not vw.is_over():
        print(f'epoch: {vw.epoch}')
        cls.model.train_epoch()
        cls.model.datasets['train'].reset(with_shuffle=True)
        rets = cls.model.inference(mode='dev')
        vw.update(rets[es_metrics])
        if vw.is_updated():
            cls.model.save_model()
            dp.set('best-epoch', vw.epoch)
            dp.set('best-score', vw.max_score)
        dp.set(f'scores', rets)
    print(f'dev best scores: {cls.model.get_round_score(dp.get_best("best-score")) :.2f}')

    cls.model.load_model()
    test_rets = cls.model.inference(mode='test')
    print(f'test best scores: ' + ' '.join(test_rets['prints']))
    # cls.model = trainer
    eval_rets = cls.single_eval(cls.model_tag[0])
    cls.model.append_information_file([f'es_metrics: {es_metrics}'])
    cls.model.append_information_file(eval_rets['text'])

    eval_rets['best_epoch'] = dp.get_best('best-epoch')
    eval_rets['dev_pearson'] = dp.get_best("best-score")
    eval_rets['test_pearson'] = test_rets['pearson']
    cls.save_summary_writer(eval_rets)

    del cls
    torch.cuda.empty_cache()

    return eval_rets['pearson']


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')  # Create a new study.

    if args.model == 'att':
        objective = objective_a
    elif args.model == 'vatt':
        objective = objective_va
    elif args.model == 'seq':
        objective = objective_s2s

    study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.






