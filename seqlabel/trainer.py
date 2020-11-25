# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:基于crf的命名实体标注训练模型。
"""

from tqdm import tqdm
from datetime import datetime
import pickle
import os
from multiprocessing import cpu_count
from multiprocessing import Pool

import sklearn_crfsuite
from sklearn import model_selection
import os
import sys
father_path = os.path.abspath('..')
sys.path.append(father_path)
from seqlabel.metrics import model_metrics
from seqlabel.feature import FeatureExtracter
from seqlabel.utils import check_path
from seqlabel.data import load_ner_datas,datas2XY
from copy import deepcopy


def train_model(X_train, Y_train, opt):
    print('\nmodel parms:\n')
    print('algorithm', opt.algorithm)
    print('c1', opt.c1)
    print('c2', opt.c2)
    print('max_iterations', opt.max_iterations)
    print('all_possible_transitions', opt.all_possible_transitions)
    crf = sklearn_crfsuite.CRF(
        algorithm=opt.algorithm,
        c1=opt.c1,
        c2=opt.c2,
        max_iterations=opt.max_iterations,
        all_possible_transitions=opt.all_possible_transitions,
        verbose=opt.verbose,
    )

    crf.fit(X_train, Y_train)

    return crf


def singe_train(X, Y, opt,X_test=None, Y_test=None, **kwargs):
    model_idx = str(kwargs.get('model_idx', ''))
    model_name = kwargs.get('model_name', 'crf') + '_${}_%s.model'.format(model_idx)

    save_objects = {}
    crf = train_model(X, Y, opt)

    print()
    print(model_idx, ' start eval')
    mts_train = model_metrics(crf, [X, Y])
    save_objects['mts_train'] = mts_train

    print(model_idx, ' train_dbqa.sh metrics:')
    print(mts_train)

    test = X_test and Y_test

    f1 = 'train_f1_%s' % mts_train.loc['TOTAL', 'f1']

    if test:
        mts_test = model_metrics(crf, [X_test, Y_test])
        save_objects['mts_test'] = mts_test
        print(model_idx, ' script metrics:')
        print(mts_test)
        f1 = 'test_f1_%s' % mts_test.loc['TOTAL', 'f1']
    save_objects['model'] = crf

    save_objects['name'] = model_name % f1

    return save_objects


def k_fold_single_train(kfold_data_idx):
    start_time = datetime.now()
    train_idxs = kfold_data_idx['train_idxs']
    test_idxs = kfold_data_idx['test_idxs']
    n = kfold_data_idx['n']
    opt = kfold_data_idx['opt']

    X_train, Y_train = [global_X[i] for i in train_idxs], [global_Y[i] for i in train_idxs]
    X_dev, Y_dev = [global_X[i] for i in test_idxs], [global_Y[i] for i in test_idxs]
    save_objects = singe_train(X_train, Y_train, opt, X_dev, Y_dev,
                               model_name=opt.model_name, model_idx=n)
    save_objects['name'] = save_objects['name']

    test = global_X_test and global_Y_test
    if test:
        mts_test = model_metrics(save_objects['model'], [global_X_test, global_Y_test])
        save_objects['mts_test_other'] = mts_test
        print(n, '-mts_test_other metrics:')
        print(mts_test)

    now_time = datetime.now()
    print(n, '-cost time:', now_time - start_time, '\n')
    return save_objects


def k_fold_train(X, Y, opt,X_test=None, Y_test=None):
    mark_start_time = datetime.now()

    global global_X, global_Y, global_X_test, global_Y_test
    global_X, global_Y, global_X_test, global_Y_test = X, Y, X_test, Y_test

    print('\nuse %s--fold cross validation' % opt.k_fold)

    kf = model_selection.KFold(n_splits=opt.k_fold, shuffle=True)

    kfold_data_idxs = []
    for n, (train_idxs, test_idxs) in enumerate(tqdm(kf.split(X), desc='%s-fold' % opt.k_fold, total=opt.k_fold)):
        kfold_data_idxs.append({
            'train_idxs': train_idxs,
            'test_idxs': test_idxs,
            'n': n + 1,
            'opt': opt,
        })

    if opt.n_jobs <= 1:
        for kfold_data_idx in kfold_data_idxs:
            yield k_fold_single_train(kfold_data_idx)

    else:
        if cpu_count() < opt.n_jobs:
            import warnings
            warnings.warn('cpu数量小于n_jobs')

        print('开启%s个进程' % min(opt.n_jobs, cpu_count()))
        pool = Pool(opt.n_jobs)
        results_save_objects = pool.map(k_fold_single_train, kfold_data_idxs)
        pool.close()
        pool.join()
        for save_objects in results_save_objects:
            yield save_objects

    #
    now_time = datetime.now()
    print('total cost time:', now_time - mark_start_time)


def train(opt):
    def save(save_object):
        save_object['vocabs'] = save_vocabs
        save_object['opt'] = opt
        save_object['template_lines'] = template_lines
        save_path = os.path.join(opt.save_path, save_object['name'])
        print('save model to %s' % save_path)
        pickle.dump(save_object, open(save_path, 'wb'))

    feature_extracter = FeatureExtracter(opt, is_train=True)
    template_lines = feature_extracter.template_lines
    vocabs = feature_extracter.feature_tool.get('vocabs', None)
    print('load train_dbqa.sh data')
    train_datas = load_ner_datas(opt.train_data_path, opt.data_separator)
    X_train, Y_train = datas2XY(train_datas, feature_extracter)
    print('train_dbqa.sh data : %s' % len(X_train))
    print('this is an Examples in train_dbqa.sh data:')
    for k, i in enumerate(X_train[-2]):
        print(i['Unigram-U02'], i)
        print()
        if k > 5:
            break

    if opt.test_data_path:
        print('load script data')
        test_datas = load_ner_datas(opt.test_data_path, opt.data_separator)
        feature_extracter = FeatureExtracter(opt, pre_vocabs=vocabs, template_lines=template_lines)
        X_test, Y_test = datas2XY(test_datas, feature_extracter)
        print('script data : %s' % len(X_test))
    else:
        X_test, Y_test = None, None

    new_opt = deepcopy(opt)
    new_opt.drop_vocab_pro = 0
    save_vocabs = FeatureExtracter(new_opt, is_train=True).feature_tool.get('vocabs', None)
    check_path(opt.save_path)
    if opt.k_fold <= 1:
        save_object = singe_train(X_train, Y_train, opt=opt,
                                  X_test=X_test, Y_test=Y_test,
                                  model_name=opt.model_name,
                                  )

        save(save_object)


    else:
        save_objects = k_fold_train(X_train, Y_train, opt=opt,
                                    X_test=X_test, Y_test=Y_test)

        for save_object in save_objects:
            save(save_object)
if __name__ == '__main__':
    from opts import argparse,train_opt,crfsuite_opt,feature_opt,general_opt

    parser = argparse.ArgumentParser(
        description='opts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    crfsuite_opt(parser)
    feature_opt(parser)
    train_opt(parser)
    general_opt(parser)
    opt = parser.parse_args()
    test = train(opt)