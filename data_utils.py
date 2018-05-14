# -*- coding:utf-8 -*-
import pickle
import numpy as np
import collections

PAD_ID = 0
GO_ID = 1  # 翻译的开始
EOS_ID = 2  # 句子结束
UNK_ID = 3
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


# max_vocabulary_size = 10000

def pad_data(terms_list, max_len, pad_pre=False):
    if max_len is None:
        max_len = 0
        for terms in terms_list:
            if len(terms) > max_len:
                max_len = len(terms)
    new_terms_list = []
    for terms in terms_list:
        pad_len = max_len - len(terms)
        if pad_len > 0:
            if pad_pre:
                new_terms = [PAD_ID] * pad_len + terms
            else:
                new_terms = terms + [PAD_ID] * pad_len
        else:
            new_terms = terms[-max_len:]
        new_terms_list.append(new_terms)
    return new_terms_list


def load_vocab(vocabulary_path):
    vocab_dict = {}
    vocab_res = {}
    vid = 0
    with open(vocabulary_path, mode="r", encoding='utf-8') as vocab_file:
        for w in vocab_file:

            vocab_dict[w.strip()] = vid
            vocab_res[vid] = w.strip()
            vid += 1
    return vocab_dict, vocab_res


class DataConverter(object):
    def __init__(self):
        self.vocab = dict()
        self.keywords_vocab = dict()

    def build_vocab(self, lines):
        counter = 0
        for line in lines:
            counter += 1
            if counter % 100000 == 0:
                print("processing line %d" % counter)
            # print(line)
            w_list = line.split(' ')
            for w in w_list:
                # print("build_vocab")
                # print(w)
                if w == ' ' or w == '' or w == '\t' or w == '\n' or w == '\r':
                    continue
                if w in self.vocab:
                    self.vocab[w] += 1
                else:
                    self.vocab[w] = 1

    def save_vocab(self, vocabulary_path):
        vocab_list = _START_VOCAB + sorted(self.vocab, key=self.vocab.get, reverse=True)
        # if len(vocab_list) > max_vocabulary_size:
        #     vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w", encoding='utf-8') as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")
        print('save vocab done.')




    def convert(self, fileout='./data.pkl'):
        data_dir = './data/'

        training_file = 'TrainingData_Text.txt'
        training_keywords_file = 'TrainingData_keywords.txt'

        with open(data_dir + training_file, 'r', encoding='utf-8') as fin:
            all_lines = []
            for line in fin:
                terms = line.split('\t')
                # if len(terms[-1]) > 50:
                #     continue
                all_lines.append(terms[-1].strip())
            self.build_vocab(all_lines)
        vocab_path = './vocab.txt'
        self.save_vocab(vocab_path)

        with open(data_dir + training_keywords_file, 'r', encoding='utf-8') as fr:
            kwd_ls = []
            for line in fr:
                kwd = line.split()
                kwd_ls += kwd

            c = collections.Counter(kwd_ls)

            kwd_voc = []
            for word in c:
                if c[word] >= 1:
                    kwd_voc.append(word)
            with open('./keywords_vocab.txt', mode="w", encoding='utf-8') as vocab_file:
                for w in kwd_voc:
                    vocab_file.write(w + "\n")
            print('save kwd done.')


        converted = []
        self.vocab_dict, self.vocab_res = load_vocab(vocab_path)
        print('start padded.')
        max_len = 0

        with open(data_dir + training_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                terms = line.split('\t')
                poet = terms[-1].strip()
                # if len(terms[-1]) > 50:
                #     continue
                term_ids = []
                poet_list = poet.split()
                if len(poet_list) > max_len:
                    max_len = len(poet_list)
                term_ids.append(GO_ID)

                for w in poet_list:
                    if w in self.vocab_dict:
                        term_ids.append(self.vocab_dict[w])
                    else:
                        term_ids.append(UNK_ID)
                # 结束
                term_ids.append(EOS_ID)
                converted.append(term_ids)
        print('max len: {}'.format(max_len))

        keyword_to_idx = {ch: i for i, ch in enumerate(kwd_voc)}
        keyword_voc_size = len(kwd_voc)
        converted_keywords = []
        with open(data_dir + training_keywords_file, 'r', encoding='utf-8') as fr:

            for line in fr:
                tmp = np.zeros(keyword_voc_size)
                for wd in line.strip('\n').split(' '):
                    tmp[keyword_to_idx[wd]] = 1.0
                converted_keywords.append(tmp)


        # max_len = 50
        padded = pad_data(converted, max_len + 2)
        print('padded done.')

        save('./data_binary.txt', padded)
        padded = np.array(padded, dtype='int32')
        pickle.dump(padded, open(fileout, 'wb'))
        print('data done.')

        save('./kwd_binary.txt', converted_keywords)
        padded_kwd = np.array(converted_keywords, dtype='int32')
        pickle.dump(padded_kwd, open('kwd.pkl', 'wb'))
        print('keywords done')


def load_data(fn='./data.pkl'):
    data = pickle.load(open(fn, 'rb'))
    return data

def save(path, data):
    with open(path, mode="w", encoding='utf-8') as file:
        for w in data:

            file.write(' '.join(str(_) for _ in w))
            file.write("\n")
    print('save done.')


if __name__ == '__main__':
    converter = DataConverter()
    converter.convert()
