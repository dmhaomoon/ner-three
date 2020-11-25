# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:该方法是处理会计数据，得到新的训练语料。
"""
from tqdm import tqdm

import os
import sys
father_path = os.path.abspath('..')
sys.path.append(father_path)
from seqlabel.inference import InferenceModel
from nlp_code.word_cut.Jieba import Jieba
from nlp_code.tools.chinese_process import single_word
jieba1 = Jieba()

model_path = father_path+'/seqlabel/crf_save/crf_$1_test_f1_87.29.model'
test = InferenceModel(model_path)
kuaiji = open(father_path+'/nlp_code/data/ner/question.txt',mode='r',encoding='utf-8')
dict_file = open(father_path+'/nlp_code/data/kuaiji/dict.txt',mode='r',encoding='utf-8')
dict_list = []
for line in dict_file.readlines():
    line = line.replace('\n','')
    dict_list.append(line)

file_kuaiji = open('./crf_save/kuaijiqa_train.data',mode='a',encoding='utf-8')
file_kuaiji1 = open('./crf_save/kuaijiqa_test.data',mode='a',encoding='utf-8')
num = 0
for line in tqdm(kuaiji.readlines()):
    num = num + 1
    line = line.replace('\n','')
    words = jieba1.segment(line)[0]
    single_words = single_word(line)
    shuyu_list = {}
    entity_list = {}
    for word in words:
        if word.word in dict_list:
            begin = word.characterOffsetBegin
            end = word.characterOffsetEnd
            shuyu_list[begin,end] = 'ACCOUNTANT'
    ner_lists = test(line)
    for ner_list in ner_lists:
        begin = ner_list[0]
        end = ner_list[1]+1
        entity_list[begin,end] = ner_list[2]
    result_list = []
    for i in shuyu_list:
        if result_list == []:
            result_list.append(i)
        else:
            bol1 = True
            for j in range(len(result_list)):
                if i[0] < result_list[j][0] and i[1]<result_list[j][1] and i[1]>result_list[j][0]:
                    bol1 = False
                elif i[0]>result_list[j][0] and i[1]<result_list[j][1]:
                    bol1 = False
                elif i[0]>result_list[j][0] and i[0]<result_list[j][1] and i[1] >result_list[j][1]:
                    bol1 = False

                if bol1:
                    if i[0] <=result_list[j][0] and i[1] >=result_list[j][1]:
                        result_list[j] = i
                    else:
                        if i not in result_list:
                            result_list.append(i)
    for i in entity_list:
        if result_list == []:
            result_list.append(i)
        else:
            bol1 = True
            for j in range(len(result_list)):
                if i[0] <result_list[j][0] and i[1]<result_list[j][1] and i[1]>result_list[j][0]:
                    bol1 = False
                elif i[0]>result_list[j][0] and i[1]<result_list[j][1]:
                    bol1 = False
                elif i[0]>result_list[j][0] and i[0]<result_list[j][1] and i[1] >result_list[j][1]:
                    bol1 = False

                if bol1:
                    if i[0] <=result_list[j][0] and i[1] >=result_list[j][1]:
                        result_list[j] = i
                    else:
                        if i not in result_list:
                            result_list.append(i)
    result = []
    for i in result_list:
        if i not in result:
            result.append(i)
    pos_dict = {}
    for i in result:
        a = list(range(i[0],i[1]))
        pos = ''
        try:
            pos = shuyu_list[i]
        except:
            pos = entity_list[i]
        if len(a) == 1:
            pos_word = 'S_'+ pos
            pos_dict[a[0]] = pos_word
        else:
            pos_dict[a[0]] = 'B_'+ pos
            for i in range(len(a)-1):
                pos_dict[a[i+1]] = 'I_' + pos
            pos_dict[a[-1]] = 'E_' + pos
    for i in range(len(single_words)):
        if i in pos_dict:
            if num%10 == 0:
                file_kuaiji1.write(single_words[i]+' '+pos_dict[i]+'\n')
            else:
                file_kuaiji.write(single_words[i]+' '+pos_dict[i]+'\n')
        else:
            if num%10 == 0:
                file_kuaiji1.write(single_words[i] + ' O'+'\n')
            else:
                file_kuaiji.write(single_words[i] + ' O'+'\n')
    if num%10 ==0:
        file_kuaiji1.write('\n')
    else:
        file_kuaiji.write('\n')