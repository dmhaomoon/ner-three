# -*- coding: UTF-8 -*-
"""
Author:杜梦豪
date:2019.8.8
version:1.0
feature:一些参数的配置
"""

import argparse
import os

base_path = os.path.dirname(__file__)
father_path = os.path.abspath('..')

def crfsuite_opt(parser):
    """
    sklearn_crfsuite.CRF 参数
    参数详细说明请查看
    >>> import sklearn_crfsuite
    >>> help(sklearn_crfsuite.CRF)
    """
    group = parser.add_argument_group('sklearn_crfsuite.CRF')
    group.add_argument('-algorithm', type=str, default='lbfgs',
                       help="""str, Training algorithm.\n\n
                       Allowed values:[lbfgs|l2sgd|ap|pa|arow]
                       """)
    group.add_argument('-c1', type=float, default=0,
                       help="""float, The coefficient for L1 regularization.
                            Supported training algorithms: lbfgs
                       """)
    group.add_argument('-c2', type=float, default=1.0,
                       help="""float, The coefficient for L2 regularization.
                            Supported training algorithms: l2sgd, lbfgs
                       """)
    group.add_argument('-max_iterations', type=int, default=None,
                       help="""int, The maximum number of iterations for optimization algorithms.
                            Default value depends on training algorithm:
                            [lbfgs: unlimited | l2sgd: 1000 | ap: 100 | pa: 100 | arow: 100]
                       """)
    group.add_argument('-all_possible_transitions', type=bool, default=True,
                       help="""bool, optional (default=False)
                        Specify whether CRFsuite generates transition features that
                        do not even occur in the training data 
                       """)


def feature_opt(parser):
    group = parser.add_argument_group('Base Feature')
    group.add_argument('-template_path', type=str, default='base',
                       help="""特征模板文件路径,提供两个默认模板标识[base | w2],
                            详细可以查看nlp.seqlabel.files.*_template
                            """)

    group = parser.add_argument_group('Vocab')
    group.add_argument('-vocabs_dir_path', type=str, default=None,
                       help="词典文件夹路径，读取文件夹下的所有词典")

    group.add_argument('-vocab_path', type=str, default=None, nargs='*',
                       help="词典文件路径,如多个词典，则用空格隔开")

    group.add_argument('-vocab_separator', type=str, default='\t',
                       help="词典文件中每行的分割符号")

    group.add_argument('-vocab_match_type', type=str, default='mc',
                       help="""可选"c, m, mc",默认"mc", 分别对应：
                            c:  "BIES + _ +　词"
                            m:  "BIES + _"
                            mc: "BIES + _","BIES + _ + 词"
                       """)

    group.add_argument('-drop_vocab_pro', type=float, default=0,
                       help="训练过程中，以一定概率词典特征")

    group.add_argument('-vocab_name_space', type=bool, default=True,
                       help="以词典文件名字作为特征名字前缀")

    group = parser.add_argument_group('Word and Pos')
    group.add_argument('-use_word', action='store_true',
                       help="使用额外分词信息")

    group.add_argument('-use_pos', action='store_true',
                       help="使用额外词性信息")

    group.add_argument('-use_ner', action='store_true',
                       help="使用额外命名实体识别信息")

    group.add_argument('-add_prefix', action='store_true',
                       help="添加前缀，如B,I,O,E,S")

    group.add_argument('-add_suffix', action='store_true',
                       help="添加后缀")

    group.add_argument('-nlptool', type=str, default='ltp',
                       help="""分词词性工具，可选
                            [ltp: 哈工大 | stanford: 斯坦福 | jieba: 结巴]
                        """)

    group.add_argument('-stanford_path', type=str, default=father_path+'/nlp_code/data/stanford',
                       help="斯坦福nlp服务的 host")


    group.add_argument('-LTP_DATA_DIR', type=str, default=father_path+'/nlp_code/data/ltp_data_v3.4.0',
                       help="哈工大nlp工具的模型文件夹路径")

    group.add_argument('-jieba_HMM', type=bool, default=False,
                       help="是否启用结巴分词HMM模式")

    group.add_argument('-user_dicts_path', type=str, default=None,
                       help="额外nlptool的自定义词典文件")



def train_opt(parser):
    group = parser.add_argument_group('Train')
    group.add_argument('-k_fold', type=int, default=5,
                       help="交叉验证折数")

    group.add_argument('-train_data_path', type=str,default=father_path+'/nlp_code/data/ner/kuaijiqa_train.data',
                       help="训练数据路径")

    group.add_argument('-test_data_path', type=str, default=father_path+'/nlp_code/data/ner/kuaijiqa_test.data',
                       help="测试数据路径")

    group.add_argument('-verbose', action='store_true',
                       help="打印训练过程信息")

    group.add_argument('-save_path', type=str, default='crf_save_question',
                       help="模型保存路径")

    group.add_argument('-model_name', type=str, default='crf',
                       help="模型保存名字前缀")

    group.add_argument('-n_jobs', type=int, default=3,
                       help="使用交叉验证时，开启的进程数量")


def general_opt(parser):
    group = parser.add_argument_group('General')
    group.add_argument('-data_separator', type=str, default=' ',
                       help="数据文件分隔符")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='opts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    crfsuite_opt(parser)
    feature_opt(parser)
    train_opt(parser)
    #
    opt = parser.parse_args()
    print(opt)
    print(opt.c1)
