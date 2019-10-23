# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:该模型是metrics，运用于对数据进行标签。
"""

from collections import defaultdict
import pandas as pd
import numpy as np

def label2entiy_idx(label):
    entiys = defaultdict(list)
    mark_l = []
    last_tag = ''
    for idx, l in enumerate(label):
        if l == 'O':
            if not mark_l:
                continue
            tag_state = ''

        else:
            mark_l.append(str(idx))
            tag_state, tag = l.split('_')
            last_tag = tag

        if tag_state in {'E', 'S'} or l == 'O':
            entiys[last_tag].append((mark_l[0] + '-' + mark_l[-1]))
            mark_l = []

    return entiys


def get_metrics(Y_true, Y_pred, tags=None):
    def check(tag):
        if tag not in entiys_counter:
            entiys_counter[tag] = np.zeros((3))

    entiys_counter = {}

    for n, (l_true, l_pred) in enumerate(zip(Y_true, Y_pred)):
        entiy_idx_true = label2entiy_idx(l_true)
        entiy_idx_pred = label2entiy_idx(l_pred)
        for tag, idxs in entiy_idx_pred.items():
            check(tag)
            entiys_counter[tag][1] += len(idxs)
            for idx in idxs:
                if idx in entiy_idx_true[tag]:
                    entiys_counter[tag][0] += 1

        for tag, idxs in entiy_idx_true.items():
            check(tag)
            entiys_counter[tag][2] += len(idxs)

    socre = {}
    for tag in sorted(entiys_counter.keys()):
        cnt = entiys_counter[tag]
        if cnt[0] == 0:
            precision=recall=f1=0

        else:
            precision, recall = (100 * cnt[0]) / cnt[1:]
            f1 = 2 * precision * recall / (precision + recall)
        socre[tag] = [precision, recall, f1]


    metrics = pd.DataFrame(socre, index=['precision', 'recall', 'f1']).T

    if tags:
        metrics = metrics.loc[tags]

    total = metrics.mean(axis=0)
    total.name = 'TOTAL'
    metrics = metrics.append(total)
    metrics = round(metrics, 2)
    return metrics


def model_metrics(crf_model, data, tags=None):
    #基于模型对数据进行标注
    if tags is None:
        labels = list(crf_model.classes_)
        labels.remove('O')
        tags = list({l[2:] for l in labels})

    tags = sorted(list(tags))

    X, Y = data
    pred = crf_model.predict(X)

    metrics = get_metrics(Y, pred, tags)


    return metrics