# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:
"""

import os
import re
import json

from math import exp, log
import os
import sys
father_path = os.path.abspath('..')
sys.path.append(father_path)
from nlp_code.utils_nlp.IO import readfile


def check_path(path):
    if not os.path.exists(path):
        print('%s is not exists, create %s' % (path, path))
        os.makedirs(path, exist_ok=True)


def char2num(char):
    if char.isdigit():
        num = int(char)
    else:
        try:
            num = float(char)
        except:
            num = char
    return num



def get_feature_func(feature_templ):
    def func(sequence, idx):
        feature = ''
        sequence_len = len(sequence)

        for shift, feature_line in feature_templ:
            shift_idx = idx + shift
            if shift_idx < 0 or shift_idx >= sequence_len:
                break
            feature += sequence[shift_idx][feature_line]
        return feature

    return func


def load_feature_template(feature_template_path, template_lines=None):
    feature_templates = {}
    prefix = ''
    save_template_lines = []

    if template_lines is None:
        if feature_template_path == 'base':
            feature_template_path = os.path.join(os.path.dirname(__file__), 'files', 'base_template')
        elif feature_template_path == 'w2':
            feature_template_path = os.path.join(os.path.dirname(__file__), 'files', 'w12_template')

        print('load feature template form %s' % feature_template_path)

        lines = readfile(feature_template_path)

    else:
        lines = template_lines

    for line in lines:
        save_template_lines.append(line)
        if not line:
            continue

        if template_lines is None:
            print(line)

        if line.startswith('# '):
            prefix = line[2:] + '-'
            continue
        name, templs = line.split(':')
        templs = templs.split('/')

        feature_templ = []
        for templ in templs:
            shift, feature_line = re.findall('\[(-?\d),(-?\d)\]', templ)[0]
            shift = int(shift)
            feature_line = int(feature_line)
            feature_templ.append((shift, feature_line))
        feature_templates[prefix + name] = feature_templ

    feature_funcs = {}
    for feature_name, feature_templ in feature_templates.items():
        feature_funcs[feature_name] = get_feature_func(feature_templ)

    return feature_funcs, save_template_lines


# def create_tag(word_len, tag, only_tag=False, concat_tag='_'):
#     if only_tag:
#         return [tag] * word_len
#     tag = concat_tag + tag if tag else ''
#     if tag == 'O':
#         return ['O'] * word_len
#     if word_len == 1:
#         tags = ['S' + tag]
#     else:
#         tags = ['B' + tag] + ['I' + tag] * (word_len - 2) + ['E' + tag]
#     return tags


def create_tag(word_len, tag='', only_tag=False, concat_tag='_', tag_type='BIES'):
    def concat(tt, tag):
        return concat_tag.join([tt, tag])

    start_tag = concat('B', tag)
    if tag_type == 'BIO':
        alone_tag = start_tag
        end_tag = middle_tag = concat('I', tag)
    elif tag_type == 'BMES':
        middle_tag = concat('M', tag)
        end_tag = concat('E', tag)
        alone_tag = concat('S', tag)

    elif tag_type == 'BIES':
        middle_tag = concat('I', tag)
        end_tag = concat('E', tag)
        alone_tag = concat('S', tag)

    else:
        raise Exception('tag_type must be BIO or BMES or BIES, but get %s' % tag_type)

    if only_tag:
        return [tag] * word_len

    if tag == 'O':
        return [tag] * word_len

    elif word_len == 1:
        tags = [alone_tag]
    else:
        tags = [start_tag] + [middle_tag] * (word_len - 2) + [end_tag]
    return tags


def format_result(result, pro_dict=None):
    entiys = []
    mark_l = []
    word = ''
    pro = []
    for idx, (c, l) in enumerate(result):
        if l == 'O':
            continue

        mark_l.append(idx)
        word += c
        if pro_dict:
            pro.append(pro_dict[idx][l])

        if l[0] in ['E', 'S']:
            if pro:
                pro = exp(sum([log(p) for p in pro]) / len(pro))
                entiys.append((mark_l[0], mark_l[-1], l[2:], word, pro))
            else:
                entiys.append((mark_l[0], mark_l[-1], l[2:], word))
            mark_l = []
            word = ''
            pro = []
    return entiys


def save_parms(parms, path=''):
    json.dump(parms, open(os.path.join(path, 'parms.json'), "w"))
