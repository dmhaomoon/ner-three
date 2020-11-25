# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:。
"""

import pickle
import os
import sys
father_path = os.path.abspath('..')
sys.path.append(father_path)
from seqlabel.feature import FeatureExtracter
from seqlabel.utils import format_result


class InferenceModel:
    """
    用于读取训练得到的序列标注模型
    Parameters
    ----------
        model_path:  训练得到的模型路径
        **kwargs:   用于修改opt的参数

    Examples
    --------
        model = InferenceModel(model_path)

    >>> model('中国位于亚洲')
    ['B_LOC', 'E_LOC', 'O', 'O', 'B_LOC', 'E_LOC']
    >>> model(['中国位于亚洲', '姚明是一个足球运动员'])
    [['B_LOC', 'E_LOC', 'O', 'O', 'B_LOC', 'E_LOC'],
    ['B_PERSON', 'E_PERSON', 'O', 'B_NUMBER', 'E_NUMBER', 'O', 'O', 'O', 'O', 'O']]

    """

    def __init__(self, model_path, **kwargs):
        self._load(model_path, **kwargs)

    def __call__(self, sentences, marginal=False, out_p=False):
        """
        Parameters
        ----------
            sentences:  string or [string1, string2], 一个或多个句子
            marginal:   bool, 是否输出边缘概率，默认 False

        Return
        ------
            返回标注结果（marginal=False）或每个位置的所有label的概率(marginal=True)
        """

        if isinstance(sentences, str):
            sentences = [sentences]
            single = True
        else:
            single = False

        features = self.sentences2features(sentences)
        annotations = self.predict(features)
        if marginal:
            marginal_annotations = self.predict_marginals(features)

        result = [format_result(zip(sentence, annotation), marginal_annotations[idx] if marginal else None)
                  for idx, (sentence, annotation) in enumerate(zip(sentences, annotations))]

        if single:
            result = result[0]
            if marginal:
                marginal_annotations = marginal_annotations[0]

        if out_p:
            return result, marginal_annotations
        else:
            return result

    def sentences2features(self, sentences):
        """
        提取特征
        Parameters
        ----------
            sentences:  string or [string1, string2], 一个或多个句子

        Return
        ------
            特征提取结果
        """

        features = [self.feature_extracter(sentence) for sentence in sentences]
        return features

    def predict_marginals(self, features):
        """
        输出每个位置的所有label的概率

        Parameters
        ----------
            features: 提取的特征

        Return
        ------
            每个位置的所有label的概率
        """
        annotation_marginal = self.model.predict_marginals(features)
        return annotation_marginal

    def predict(self, features):
        """
        输出每个位置的最佳label

        Parameters
        ----------
            features: 提取的特征

        Return
        ------
            每个位置的最佳label
        """
        annotation = self.model.predict(features)
        return annotation

    def _load(self, model_path, **kwargs):
        models = pickle.load(open(model_path, 'rb'))
        self.vocabs = None
        self.template_lines = None
        for key, obj in models.items():
            setattr(self, key, obj)

        if hasattr(self, 'opt') and hasattr(self, 'model'):
            for key, obj in kwargs.items():
                setattr(self.opt, key, obj)

            self.feature_extracter = FeatureExtracter(self.opt,
                                                      pre_vocabs=self.vocabs,
                                                      template_lines=self.template_lines)
        else:
            raise Exception('模型文件中找不到opt')

if __name__ == '__main__':
    model_path = father_path+'/seqlabel/crf_save_question/crf_$2_test_f1_77.98.model'
    test = InferenceModel(model_path)
    test1 = InferenceModel(father_path+'/seqlabel/crf_save_question/crf_$4_test_f1_82.32.model')
    print(test(['小规模纳税人为什么可以底销项税不能底进项税']))
    # file = open('./question_document.txt',mode='r',encoding='utf-8')
    # file_write = open('./question_entity_same.txt',mode='a',encoding='utf-8')
    # file_write1 = open('./question_entity_different.txt',mode='a',encoding='utf-8')
    # dict_file = open('/data/menghao/yun2space/nlp_code/data_nlp/dict.txt', mode='r', encoding='utf-8')
    # dict_list = []
    # import jieba
    # jieba.load_userdict('/data/menghao/yun2space/nlp_code/data_nlp/dict.txt')
    # for line in dict_file.readlines():
    #     line = line.replace('\n', '')
    #     dict_list.append(line)
    # for line in file.readlines():
    #     line = line.replace('\n','').split('&&')[0]
    #     words = list(jieba.cut(line))
    #     entity_list = []
    #     for word in words:
    #         if word in dict_list:
    #             entity_list.append(word)
    #     if len(test([line])[0]) == len(test1([line])[0]):
    #         bol = True
    #         if len(entity_list) == len(test([line])[0]):
    #             if test([line])[0] != []:
    #                 for i in range(len(test([line])[0])):
    #                     if entity_list[i] != test([line])[0][i][3]:
    #                         bol = False
    #         if bol:
    #             file_write.write(str(line) + '\n')
    #             file_write.write(str(test([line])) + '\n')
    #             file_write.write(str(test1([line])) + '\n')
    #             file_write.write(str(entity_list) + '\n')
    #             file_write.write('\n')
    #         else:
    #             file_write1.write(str(line) + '\n')
    #             file_write1.write(str(test([line])) + '\n')
    #             file_write1.write(str(test1([line])) + '\n')
    #             file_write1.write(str(entity_list) + '\n')
    #             file_write1.write('\n')
    #     else:
    #         file_write1.write(str(line)+'\n')
    #         file_write1.write(str(test([line]))+'\n')
    #         file_write1.write(str(test1([line]))+'\n')
    #         file_write1.write(str(entity_list)+'\n')
    #         file_write1.write('\n')