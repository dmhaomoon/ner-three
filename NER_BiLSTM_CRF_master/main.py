# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

class NER_LSTM_CRF(object):
    def __init__(self,con_key):
        flags = tf.app.flags
        flags.DEFINE_boolean("clean",       True,      "clean train folder")
        flags.DEFINE_boolean("train",       con_key['trian_type'],      "Wither train the model---False")
        # configurations for the model
        flags.DEFINE_boolean("use_start_end_crf",       True,      "whether use start and end status in crf loss")
        flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
        flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
        flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
        flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

        # configurations for training
        flags.DEFINE_float("clip",          5,          "Gradient clip")
        flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
        flags.DEFINE_integer("batch_size",    32,         "batch size")
        flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
        flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
        flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
        flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
        flags.DEFINE_boolean("lower",       True,       "Wither lower case")

        flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
        flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
        flags.DEFINE_string("ckpt_path",    con_key['ckpt_path'],      "Path to save model-ckpt_book")
        flags.DEFINE_string("summary_path", con_key['summary_path'],      "Path to store summaries-summary_book")
        flags.DEFINE_string("log_file",     con_key['log_file'],    "File for log-train_book.log")
        flags.DEFINE_string("map_file",     con_key['map_file'],     "file for maps-maps_book.pkl")
        flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
        flags.DEFINE_string("config_file",  "config_file",  "File for config")
        flags.DEFINE_string("script",       "conlleval",    "evaluation script")
        flags.DEFINE_string("result_path",  con_key['result_path'],       "Path for results-result_book")
        flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
        flags.DEFINE_string("train_file",   os.path.join("data", con_key['train_file']),  "Path for train data-train_book.data")
        flags.DEFINE_string("dev_file",     os.path.join("data", con_key['dev_file']),    "Path for dev data-dev_book.data")
        flags.DEFINE_string("test_file",    os.path.join("data", con_key['test_file']),   "Path for test data-test_book.data")

        self.FLAGS = flags.FLAGS
        assert self.FLAGS.clip < 5.1, "gradient clip should't be too much"
        assert 0 <= self.FLAGS.dropout < 1, "dropout rate between 0 and 1"
        assert self.FLAGS.lr > 0, "learning rate must larger than zero"
        assert self.FLAGS.optimizer in ["adam", "sgd", "adagrad"]


    # config for the model
    def config_model(self,char_to_id, tag_to_id):
        config = OrderedDict()
        config["num_chars"] = len(char_to_id)
        config["char_dim"] = self.FLAGS.char_dim
        config["num_tags"] = len(tag_to_id)
        config["seg_dim"] = self.FLAGS.seg_dim
        config["lstm_dim"] = self.FLAGS.lstm_dim
        config["batch_size"] = self.FLAGS.batch_size
        config["use_start_end_crf"] = self.FLAGS.use_start_end_crf

        config["emb_file"] = self.FLAGS.emb_file
        config["clip"] = self.FLAGS.clip
        config["dropout_keep"] = 1.0 - self.FLAGS.dropout
        config["optimizer"] = self.FLAGS.optimizer
        config["lr"] = self.FLAGS.lr
        config["tag_schema"] = self.FLAGS.tag_schema
        config["pre_emb"] = self.FLAGS.pre_emb
        config["zeros"] = self.FLAGS.zeros
        config["lower"] = self.FLAGS.lower
        return config


    def evaluate(self,sess, model, name, data, id_to_tag, logger):
        logger.info("evaluate:{}".format(name))
        ner_results = model.evaluate(sess, data, id_to_tag)
        eval_lines = test_ner(ner_results, self.FLAGS.result_path)
        for line in eval_lines:
            logger.info(line)
        f1 = float(eval_lines[1].strip().split()[-1])

        if name == "dev":
            best_test_f1 = model.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(model.best_dev_f1, f1).eval()
                logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1
        elif name == "test":
            best_test_f1 = model.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(model.best_test_f1, f1).eval()
                logger.info("new best test f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1


    def train(self):
        # load data sets
        train_sentences = load_sentences(self.FLAGS.train_file, self.FLAGS.lower, self.FLAGS.zeros)
        dev_sentences = load_sentences(self.FLAGS.dev_file, self.FLAGS.lower, self.FLAGS.zeros)
        test_sentences = load_sentences(self.FLAGS.test_file, self.FLAGS.lower, self.FLAGS.zeros)

        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(train_sentences, self.FLAGS.tag_schema)
        update_tag_scheme(test_sentences, self.FLAGS.tag_schema)

        # create maps if not exist
        if not os.path.isfile(self.FLAGS.map_file):
            # create dictionary for word
            if self.FLAGS.pre_emb:
                dico_chars_train = char_mapping(train_sentences, self.FLAGS.lower)[0]
                dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                    dico_chars_train.copy(),
                    self.FLAGS.emb_file,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in test_sentences])
                    )
                )
            else:
                _c, char_to_id, id_to_char = char_mapping(train_sentences, self.FLAGS.lower)

            # Create a dictionary and a mapping for tags
            _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
            print(tag_to_id)
            with open(self.FLAGS.map_file, "wb") as f:
                pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
        else:
            with open(self.FLAGS.map_file, "rb") as f:
                char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        # prepare data, get a collection of list containing index
        train_data = prepare_dataset(
            train_sentences, char_to_id, tag_to_id, self.FLAGS.lower
        )
        dev_data = prepare_dataset(
            dev_sentences, char_to_id, tag_to_id, self.FLAGS.lower
        )
        test_data = prepare_dataset(
            test_sentences, char_to_id, tag_to_id, self.FLAGS.lower
        )
        print("%i / %i / %i sentences in train / dev / test." % (
            len(train_data), 0, len(test_data)))

        train_manager = BatchManager(train_data, self.FLAGS.batch_size)
        dev_manager = BatchManager(dev_data, 100)
        test_manager = BatchManager(test_data, 100)
        # make path for store log and model if not exist
        make_path(self.FLAGS)
        if os.path.isfile(self.FLAGS.config_file):
            config = load_config(self.FLAGS.config_file)
        else:
            config = self.config_model(char_to_id, tag_to_id)
            save_config(config, self.FLAGS.config_file)
        make_path(self.FLAGS)

        log_path = os.path.join("log", self.FLAGS.log_file)
        logger = get_logger(log_path)
        print_config(config, logger)

        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, self.FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
            logger.info("start training")
            loss = []
            for i in range(100):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % self.FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []

                best = self.evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
                if best:
                    save_model(sess, model, self.FLAGS.ckpt_path, logger)
                self.evaluate(sess, model, "test", test_manager, id_to_tag, logger)


    def evaluate_line(self):
        config = load_config(self.FLAGS.config_file)
        logger = get_logger(self.FLAGS.log_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with open(self.FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, self.FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
            while True:
                # try:
                #     line = input("请输入测试句子:")
                #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                #     print(result)
                # except Exception as e:
                #     logger.info(e)

                    line = input("请输入测试句子:")
                    result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                    print(result)


def main():
    con_key_book = {
        'trian_type': False,
        'ckpt_path': 'ckpt_book',
        'summary_path': 'summary_book',
        'log_file': 'train_book.log',
        'map_file': 'maps_book.pkl',
        'result_path': 'result_book',
        'train_file': 'train_book.data',
        'test_file': 'test_book.data',
        'dev_file': 'dev_book.data',
    }
    ner_model = NER_LSTM_CRF(con_key_book)
    if ner_model.FLAGS.train:
        if ner_model.FLAGS.clean:
            clean(ner_model.FLAGS)
        ner_model.train()
    else:
        ner_model.evaluate_line()

def test(con_key,sentence):
    ner_model = NER_LSTM_CRF(con_key)
    config = load_config(ner_model.FLAGS.config_file)
    logger = get_logger(ner_model.FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(ner_model.FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    result_list = []
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, ner_model.FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        for line in sentence:
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            result_list.append(result)
    return result_list

if __name__ == "__main__":
    # main()
    # from tqdm import tqdm
    # con_key_book = {
    #     'trian_type': False,
    #     'ckpt_path': 'ckpt_book',
    #     'summary_path': 'summary_book',
    #     'log_file': 'train_book.log',
    #     'map_file': 'maps_book.pkl',
    #     'result_path': 'result_book',
    #     'train_file': 'train_book.data',
    #     'test_file': 'test_book.data',
    #     'dev_file': 'dev_book.data',
    # }
    # file = open('./question_entity.txt', mode='r', encoding='utf-8')
    # file_write = open('./question_entity_result.txt', mode='a', encoding='utf-8')
    # sentences = file.readlines()
    # sentences_lsit = []
    # for line in sentences:
    #     sentences_lsit.append(line)
    # ner_model = NER_LSTM_CRF(con_key_book)
    # config = load_config(ner_model.FLAGS.config_file)
    # logger = get_logger(ner_model.FLAGS.log_file)
    # # limit GPU memory
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with open(ner_model.FLAGS.map_file, "rb") as f:
    #     char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    # with tf.Session(config=tf_config) as sess:
    #     model = create_model(sess, Model, ner_model.FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
    #     for i in tqdm(range(int(len(sentences_lsit) / 5))):
    #         entity = []
    #         result = model.evaluate_line(sess, input_from_line(sentences_lsit[i * 5].replace('\n', ''), char_to_id), id_to_tag)
    #         for j in result['entities']:
    #             a = (j['start'], j['end'], j['type'], j['word'])
    #             entity.append(a)
    #         file_write.write(sentences_lsit[i * 5])
    #         file_write.write('term_words:--------' + sentences_lsit[i * 5 + 3])
    #         file_write.write('crf_book:----------' + sentences_lsit[i * 5 + 1])
    #         file_write.write('crf_qa:------------' + sentences_lsit[i * 5 + 2])
    #         file_write.write('bilstm_crf_book:---'+ str([entity]))
    #         file_write.write(sentences[i * 5 + 4]+'\n')

    from tqdm import tqdm

    con_key_book = {
        'trian_type': False,
        'ckpt_path': 'ckpt',
        'summary_path': 'summary',
        'log_file': 'train.log',
        'map_file': 'maps.pkl',
        'result_path': 'result',
        'train_file': 'train.data',
        'test_file': 'test.data',
        'dev_file': 'dev.data',
    }
    file = open('./question_entity_result.txt', mode='r', encoding='utf-8')
    file_write = open('./question_entity.txt', mode='a', encoding='utf-8')
    sentences = file.readlines()
    sentences_lsit = []
    for line in sentences:
        sentences_lsit.append(line)
    ner_model = NER_LSTM_CRF(con_key_book)
    config = load_config(ner_model.FLAGS.config_file)
    logger = get_logger(ner_model.FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(ner_model.FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, ner_model.FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        for i in tqdm(range(int(len(sentences_lsit) / 6))):
            entity = []
            result = model.evaluate_line(sess, input_from_line(sentences_lsit[i * 6].replace('\n', ''), char_to_id),
                                         id_to_tag)
            for j in result['entities']:
                a = (j['start'], j['end'], j['type'], j['word'])
                entity.append(a)
            file_write.write(sentences_lsit[i * 6])
            file_write.write(sentences_lsit[i * 6 + 1])
            file_write.write(sentences_lsit[i * 6 + 2])
            file_write.write(sentences_lsit[i * 6 + 3])
            file_write.write(sentences_lsit[i * 6 + 4])
            file_write.write('bilstm_crf_qa:-----' + str([entity]))
            file_write.write(sentences[i * 6 + 5] + '\n')


    # con_key_qa = {
    #     'trian_type': False,
    #     'ckpt_path': 'ckpt',
    #     'summary_path': 'summary',
    #     'log_file': 'train.log',
    #     'map_file': 'maps.pkl',
    #     'result_path': 'result',
    #     'train_file': 'train.data',
    #     'test_file': 'test.data',
    #     'dev_file': 'dev.data',
    # }
    # sentence = ['小规模纳税人为什么可以底销项税不能底进项税']
    # # print('book',test(con_key_book, sentence))
    # print('qa',test(con_key_qa,sentence))



