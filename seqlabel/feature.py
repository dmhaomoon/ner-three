# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:。
"""

from collections import defaultdict, Counter
from ahocorasick import Automaton
from tqdm import tqdm
import random
import os
from nlp.utils_nlp import *


class WordPos:
    def __init__(self,
                 use_word=True,
                 use_pos=True,
                 use_ner=False,
                 nlptool='jieba',
                 stanford_path=None,
                 LTP_DATA_DIR=None,
                 jieba_HMM=True,
                 user_dicts_path=None,
                 add_prefix=False,
                 add_suffix=False,
                 ):
        self.nlptool = nlptool
        self.use_word = use_word
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.jieba_HMM = jieba_HMM
        self.stanford_path = stanford_path
        self.LTP_DATA_DIR = LTP_DATA_DIR
        self.user_dicts_path = user_dicts_path
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix

        self._init_model()

    def _init_model(self):
        print('init the word and pos feature extracter')
        print('use %s tool' % self.nlptool)

        if self.nlptool == 'jieba':
            print('use HMM', self.jieba_HMM)
            from opennlp.Jieba import Jieba
            self.model = Jieba(HMM=self.jieba_HMM)

        elif self.nlptool == 'ltp':
            from opennlp.ltp import LTP
            if not self.LTP_DATA_DIR or not os.path.exists(self.LTP_DATA_DIR):
                raise Exception('LTP_DATA_DIR 不存在')
            self.model = LTP(LTP_DATA_DIR=self.LTP_DATA_DIR, user_dicts_path=self.user_dicts_path)

        elif self.nlptool == 'stanford':
            from opennlp.stanford import StanfordNlp
            self.model = StanfordNlp(self.stanford_path, user_dicts_path=self.user_dicts_path)

        self.tag_types = []
        if self.use_word:
            self.tag_types.append('word')
            self.func_name = 'segment'
        if self.use_pos:
            self.tag_types.append('pos')
            self.func_name = 'pos'
        if self.use_ner and self.nlptool != 'jieba':
            self.tag_types.append('ner')
            self.func_name = 'ner'

        if len(self.tag_types) == 0:
            raise Exception('至少使用一个 use_word / use_pos / use_ner')

    def __call__(self, *args, **kwargs):
        features = self._get_feature(*args, **kwargs)

        return features

    def _get_feature(self, sentence):
        sequences = getattr(self.model, self.func_name)(sentence)
        features = defaultdict(dict)
        for sequence in sequences:
            doc_sid = sequence.doc_sid
            for token in sequence:
                characterOffsetBegin = token.characterOffsetBegin + doc_sid
                characterOffsetEnd = token.characterOffsetEnd + doc_sid
                print(characterOffsetBegin)
                print(characterOffsetEnd)
                cache = {}
                for tag_type in self.tag_types:
                    cache[tag_type] = create_tag(characterOffsetEnd - characterOffsetBegin, getattr(token, tag_type))

                    if self.add_suffix:
                        cache[tag_type] = create_tag(characterOffsetEnd - characterOffsetBegin,
                                                     getattr(token, tag_type), only_tag=True)

                if self.add_prefix:
                    cache['prefix'] = create_tag(characterOffsetEnd - characterOffsetBegin, '')
                print(cache)
                for c_idx, idx in enumerate(range(characterOffsetBegin, characterOffsetEnd)):
                    for tag_type, tags in cache.items():
                        features[idx][tag_type] = tags[c_idx]

        return features


class TrieTree:
    '''
    前缀树类，用于匹配词典
    Parameters
    ----------
    paths:一个或者一组字典文件名(str or list)，文件格式要求每列用制表符隔开：
        第一列为词，
        第二列为词对应的信息，
        第三列为信息附带的数值等，没有则默认为True
        如：　
        中国 LOC 0.8
        美国 国家

    tp:为匹配类型，可选"c, m, mc",默认"mc", 分别对应：
        c:  "BIES + _ +　词"
        m:  "BIES + _"
        mc: "BIES + _","BIES + _ + 词"

    Return
    ------
    defaultdict(in, {idx_0:{feature: value}, idx_1:...})
    返回一个以词id对应特征字典的特征集合


    Examples
    --------
    >>> trietree_c = TrieTree(paths=your_vocab_files, tp='c')
    >>> trietree_c("中国是一个国家")
    defaultdict(in, {0: {'B_LOC': True}, 1: {'E_LOC': True}})

    >>> trietree_m = TrieTree(paths=your_vocab_files, tp='m')
    >>> trietree_m("中国是一个国家")
    defaultdict(in, {0: {'B': True}, 1: {'E': True}})

    >>> trietree_mc = TrieTree(paths=your_vocab_files, tp='mc')
    >>> trietree_mc("中国是一个国家")
    defaultdict(in,
            {0: {'B': True, 'B_LOC': True}, 1: {'E': True, 'E_LOC': True}})

    '''

    def __init__(self, vocab_paths, vocab_match_type='mc', drop_vocab_pro=0, vocab_name_space=False, separator='\t'):
        self.match_cnt = Counter()
        self.user_automaton = {}
        self.keep_vocab_pro = 1 - drop_vocab_pro
        self.vocab_name_space = vocab_name_space
        self.vmp = vocab_match_type
        self.load_vocab(vocab_paths, separator=separator)
        self.cnt = Counter()

        print('trietree:\ntp: %s\n, vocab path:%s' % (self.vmp, str(vocab_paths)))
        if self.keep_vocab_pro < 1:
            print('drop vocab pro', self.keep_vocab_pro)

    def __call__(self, *args, **kwargs):
        vocab_feature = self._vocab_feature(*args, **kwargs)
        return vocab_feature

    def load_vocab(self, paths, add=False, separator='\t'):
        if add and hasattr(self, 'automaton'):
            pass
        else:
            self.automaton = Automaton()

        vocab = defaultdict(list)
        tags = set()
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            name_space = os.path.split(path)[-1]
            print('read %s' % path)
            output = os.popen('wc -l ' + path)
            total = int(output.readline().split()[0])
            with open(path, 'r') as r_f:
                print('vocab file Examples:')
                for n, line in enumerate(r_f):
                    print(line.strip())
                    if n >= 10:
                        break
                r_f.seek(0)
                for line in tqdm(r_f, desc='read file', total=total):
                    if random.random() > self.keep_vocab_pro:
                        continue
                    splits = line.strip().split(separator)
                    try:
                        if len(splits) == 2:
                            word, tag = splits
                            value = True
                        elif len(splits) == 3:
                            word, tag, value = splits
                            value = char2num(value)

                        elif len(splits) == 1:
                            word = splits[0]
                            value = True
                            tag = 'WORD'

                        else:
                            continue

                        if self.vocab_name_space:
                            tag = name_space + '_' + tag
                        vocab[word].append((tag, value))
                        if tag not in tags:
                            tags.add(tag)


                    except Exception as e:
                        print('vocab error: path-%s, line %s' % (path, line), e)
                        continue

        self.tags = tags if not hasattr(self, 'tags') else self.tags | tags

        for word, value in tqdm(vocab.items(), desc='add words'):
            self.automaton.add_word(word, (len(word), word, value))

        print('总共有%s个词' % len(vocab))
        self.automaton.make_automaton()

    def _vocab_feature(self, sentence):
        vocab_feature = defaultdict(dict)
        self.match(sentence, vocab_feature)
        if self.user_automaton:
            self.match(sentence, vocab_feature, base_or_user='user')

        return vocab_feature

    def match(self, sentence, vocab_feature, base_or_user='base'):

        if base_or_user == 'base':
            result = self.automaton.iter(sentence)
        else:
            result = self.user_automaton.iter(sentence)

        for end_idx, (word_len, _, tag_value) in list(result):

            start_idx = end_idx - word_len + 1
            for tag, value in tag_value:
                self.match_cnt[tag] += 1
                if self.vmp == 'c':
                    tagss = [create_tag(word_len, tag)]
                elif self.vmp == 'm':
                    tagss = [create_tag(word_len, '')]
                elif self.vmp == 'mc':
                    tagss = [create_tag(word_len, tag), create_tag(word_len, '')]
                else:
                    tagss = []
                for tags in tagss:
                    for idx, tag in zip(range(start_idx, end_idx + 1), tags):
                        vocab_feature[idx][tag] = value

    def init_user_automaton(self):
        self.user_automaton = Automaton()
        self.user_automaton.make_automaton()

    def add_word(self, word, tag, value, update=True):
        '''
        Parameters
        ----------
        word:  匹配的词
        tag:   词对应的信息
        value: 信息附带的数值

        Examples
        --------
        >>> trietree.add_word('中国', '国家', True)
        >>> trietree.user_automaton.get('中国')
        (2, '中国', [('LOC', True)])
        '''
        have_add = ''
        if self.user_automaton == {}:
            self.init_user_automaton()
        wl, w, tag_values = self.user_automaton.get(word, (len(word), word, []))
        for i, (t, v) in enumerate(tag_values):
            if t == tag:
                tag_values[i] = (tag, value)
                break
        else:
            tag_values.append((tag, value))
        self.user_automaton.add_word(w, (wl, w, tag_values))
        if update:
            self.user_automaton.make_automaton()

    def add_words(self, word_tag_values):
        '''
        do:

        for word, tag, value in word_tag_values:
            self.add_word(word, tag, value, update=False)



        Examples
        --------
        word_tag_values = [('中国', '面积', 9666), ('中国', '人口', 8888)]
        >>> trietree.add_word('中国', '国家', True)
        >>> trietree.user_automaton.get('中国')
        (2, '中国', [('面积', 9666), ('人口', 8888)])

        '''
        for word, tag, value in word_tag_values:
            self.add_word(word, tag, value, update=False)
        self.user_automaton.make_automaton()

    def get(self, key, default=None, vocab='all'):
        '''
        与字典get方法一样

        Parameters
        ----------
        vocab:  用于选择基本词典或者用户自定义词典，base（基本）/user（用户自定义）/all（两个），默认为all
        '''
        if vocab == 'base':
            value = self.automaton.get(key, default)
        elif vocab == 'user':
            value = self.user_automaton.get(key, default)
        else:
            value = {
                'base': self.automaton.get(key, default),
                'user': self.user_automaton.get(key, default)
            }
        return value


class FeatureExtracter:
    '''
    特征提取器，支持模板文件特征和词典特征
    Parameters
    ----------
    feature_template_path： 模板文件，参考CRF++文件
    vocab_paths:           一个或这多个词典文件路径，str/list，文件格式见TrieTree类
    tp:                    匹配类型，可选"c, m, mc",默认"mc", 详细见TrieTree类中参数介绍

    Examples
    --------
    example_template_file:
                        # Unigram
                        base:%x[0,0]




    example_vocab_files:
                        中国 LOC
                        亚洲 LOC


    >>> feature_extractor("中国是一个亚洲国家")
    [{'Unigram-base': '中', 'vocab_B_LOC': True},
     {'Unigram-base': '国', 'vocab_E_LOC': True},
     {'Unigram-base': '是'},
     {'Unigram-base': '一'},
     {'Unigram-base': '个'},
     {'Unigram-base': '亚'},
     {'Unigram-base': '洲'},
     {'Unigram-base': '国'},
     {'Unigram-base': '家'}]
    '''

    def __init__(self, opt, is_train=False, pre_vocabs=None, template_lines=None):

        self.feature_tool = {}

        self.feature_funcs, self.template_lines = load_feature_template(opt.template_path, template_lines=template_lines)

        self._load_vocabs(opt, is_train, pre_vocabs)
        self._load_wordpos(opt)

    def __call__(self, *args, **kwargs):
        features = self._sentence2features(*args, **kwargs)
        return features

    def _print_no_trietree(self):
        print('have no vocab, please set the vocab_paths when init the object')
        return None

    def _load_wordpos(self, opt):
        if opt.use_word or opt.use_pos:
            self.feature_tool['nlptool'] = WordPos(use_word=opt.use_word,
                                                   use_pos=opt.use_pos,
                                                   use_ner=opt.use_ner,
                                                   nlptool=opt.nlptool,
                                                   stanford_path=opt.stanford_path,
                                                   LTP_DATA_DIR=opt.LTP_DATA_DIR,
                                                   jieba_HMM=opt.jieba_HMM,
                                                   user_dicts_path=opt.user_dicts_path,
                                                   add_prefix=opt.add_prefix,
                                                   add_suffix=opt.add_suffix,
                                                   )

    def _load_vocabs(self, opt, is_train, pre_vocabs=None):
        vocab_paths = opt.vocab_path or []
        if opt.vocabs_dir_path:
            temp_vocab_paths = [os.path.join(opt.vocabs_dir_path, i) for i in os.listdir(opt.vocabs_dir_path)]
            vocab_paths.extend(temp_vocab_paths)

        drop_vocab_pro = opt.drop_vocab_pro if is_train else 0
        if vocab_paths or pre_vocabs:
            trietree = pre_vocabs or TrieTree(vocab_paths, vocab_match_type=opt.vocab_match_type,
                                              drop_vocab_pro=drop_vocab_pro, vocab_name_space=opt.vocab_name_space)

            self.__cache__parms = pre_vocabs or (vocab_paths, opt.vocab_match_type, opt.vocab_name_space)
            self.feature_tool['vocabs'] = trietree
            self.vocab = trietree.automaton.items()
            self.user_vocab = trietree.user_automaton.items()
            self.get = trietree.get
            self.add_word = trietree.add_word
            self.add_words = trietree.add_words
        else:
            self.vocab = self.user_vocab = self.get = self.add_word = self.add_words = self._print_no_trietree

    def _sentence2features(self, sentence):

        features = []

        extrant_features = self._extract_all_feature(sentence)

        for idx in range(len(sentence)):
            word_features = self._word2features(sentence, idx)
            if idx in extrant_features:
                for tag, value in extrant_features[idx].items():
                    word_features[tag] = value
            features.append(word_features)
        return features

    def _word2features(self, sentence, idx):
        features = {}
        for feature_name, feature_func in self.feature_funcs.items():
            feature = feature_func(sentence, idx)
            if feature:
                features[feature_name] = feature
        return features

    def _extract_all_feature(self, sentence):
        all_features = defaultdict(dict)
        for name_scope, feature_tool in self.feature_tool.items():
            features = feature_tool(sentence)
            for idx, f_dict in features.items():
                for tag, value in f_dict.items():
                    all_features[idx][name_scope + '_' + tag] = value
        return all_features



    def reload_dict(self):
        self.feature_tool['vocabs'] = trietree

if __name__ == '__main__':
    test = WordPos()
    sentence = '外购货物还有哪些没增值税？'
    print(test._get_feature(sentence))
    dict_path = '/data/menghao/yun2space/nlp_code/data_nlp/ner/server.vocab'
    test = TrieTree(dict_path)
    print(test._vocab_feature(sentence))