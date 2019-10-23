# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:。
"""

import os
from tqdm import tqdm
from collections import defaultdict
import pickle

from nlp.seqlabel.utils import format_result


def load_ner_datas(datas_path, separator=' '):
    print('load data from %s' % datas_path)
    output = os.popen('wc -l ' + datas_path)
    total = int(output.readline().split()[0])
    with open(datas_path, 'r') as r_f:
        sentence = []
        for n, line in enumerate(tqdm(r_f, total=total)):
            if line == '\n':
                if sentence:
                    yield sentence
                sentence = []
                continue
            split = line.strip().split(separator)
            if len(split) == 1:
                split = [' ', split[0]]
            sentence.append(split)


def datas2XY(datas, feature_extracter):
    X, Y = [], []
    for n, sentence_labels in enumerate(datas):
        sentence, labels = '', []
        for char, label in sentence_labels:
            sentence += char
            labels.append(label)
        features = feature_extracter(sentence)

        X.append(features)
        Y.append(labels)
    return X, Y


def get_exits_vocab(ner_data_path, path_or_bool=False):
    if path_or_bool:
        if isinstance(path_or_bool, bool):
            print('load exits vocab form data: \"%s\"' % ner_data_path)
            VOCAB = defaultdict(set)
            datas = load_ner_datas(ner_data_path)
            for data in datas:
                for _, _, tag, word in format_result(data):
                    VOCAB[tag].add(word)
        else:
            print('load exits vocab form file: \"%s\"' % path_or_bool)
            VOCAB = pickle.load(open(path_or_bool, 'rb'))
        print([(tag, len(v)) for tag, v in VOCAB.items()])
    else:
        VOCAB = None
    return VOCAB
