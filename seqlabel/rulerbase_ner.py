# -*- coding: UTF-8 -*-
import os
import sys
father_path = os.path.abspath('..')
sys.path.append(father_path)
from ahocorasick import Automaton
from nlp_code.utils_nlp.IO import readfile



class RulerBaseNer:
    def __init__(self, dict_dir, vocab_suffix='.in', min_word_len=2, pro=1):
        self.automaton = self.create_automaton(dict_dir, vocab_suffix=vocab_suffix, min_word_len=min_word_len)
        self.pro = pro

    def __call__(self, sentences, marginal=False, pro=None, out_p=False):
        pro = pro or self.pro
        if isinstance(sentences, str):
            sentences = [sentences]
            single = True
        else:
            single = False
        results = []
        for sentence in sentences:
            result = self.parse_single_sentence(sentence, marginal, pro)
            results.append(result)
        if single:
            return results[0]
        else:
            return results

    def parse_single_sentence(self, sentence, marginal=False, pro=1):
        result = []
        for end_idx, (word_len, word, tag) in self.automaton.iter(sentence):
            start_idx = end_idx - word_len + 1
            r = (start_idx, end_idx, tag, word)
            if marginal:
                r = tuple([*r, pro])
            result.append(r)
        return result

    @staticmethod
    def create_automaton(dict_dir, vocab_suffix, min_word_len=3):
        assert isinstance(min_word_len, int) or isinstance(min_word_len, dict)
        automaton = Automaton()
        if os.path.isdir(dict_dir):
            dicts_path = [os.path.join(dict_dir, i) for i in os.listdir(dict_dir) if i.endswith(vocab_suffix)]
        else:
            dicts_path = [dict_dir]
        for path in dicts_path:
            tag = os.path.split(path)[-1].strip(vocab_suffix)
            vocab = set(readfile(path, deal_func=lambda x: x.strip()))
            tag_min_word_len = min_word_len if isinstance(min_word_len, int) else min_word_len[tag]
            for word in vocab:
                word_len = len(word)
                if word_len >= tag_min_word_len:
                    automaton.add_word(word, (word_len, word, tag))
        automaton.make_automaton()
        return automaton

if __name__ == '__main__':
    dict = father_path+'/nlp_code/data/kuaiji/dict.txt'
    test = RulerBaseNer(dict)
    sentence = '外购货物还有哪些没增值税？'
    print(test.parse_single_sentence(sentence))
