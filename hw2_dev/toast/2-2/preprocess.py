import time
import logging
import argparse
from collections import OrderedDict

import torch
import numpy as np
from gensim.models import word2vec

from config import MAX_LENGTH, EMBED_SIZE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_dim", type=int, default=EMBED_SIZE, help="dimension of word embedding")
    parser.add_argument("--make", action="store_true", default=False, help="make txt for training word2vec")
    parser.add_argument("-m", "--min_count", type=int, default=5, help="min count of gensim word2vec")
    return parser.parse_args()

# # to be deprecate
# class Loader:
#     def __init__(self, model, dictionary, dir_path, batch_size, max_length=MAX_LENGTH):
#         self.model = model
#         self.dictionary = dictionary
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.dir_path = dir_path
#
#         self.i = 0
#         self.x = torch.zeros((max_length, 1)).type(torch.LongTensor)
#         self.y = torch.zeros((max_length, 1)).type(torch.LongTensor)
#         self.load_data()
#         self.num_data = self.x.shape[1]
#         self.num_batch = self.num_data / self.batch_size
#
#     def load_data(self):
#         # print('loading data from file..')
#         # start = time.time()
#         # 489928 lines in clr_conversation.txt
#         with open(self.dir_path, 'r') as f:
#             sentence1 = ""
#             sentence2 = ""
#             for idx, line in enumerate(f):
#                 # print("reading process.. %2.2f%%" % (idx/489928), end='\r')
#                 line = line.strip('\n')
#                 if line != '+++$+++':
#                     sentence1 = sentence2
#                     sentence2 = line
#
#                     x, y = self.make_pair(sentence1, sentence2)
#                     if x is None or y is None:
#                         continue
#                     self.x = torch.cat([self.x, x.unsqueeze(1)], dim=1)
#                     self.y = torch.cat([self.y, y.unsqueeze(1)], dim=1)
#
#                 else:
#                     sentence1 = ""
#                     sentence2 = ""
#         if self.x.shape[1] == 1:
#             # if only one sentence exist in a conversation..
#             raise Exception
#         self.x = self.x[:, 1:]
#         self.y = self.y[:, 1:]
#         # total_time = time.time() - start
#         # print('total time: %2d:%2d' % (total_time//60, total_time % 60))
#
#     def make_pair(self, x, y):
#         if x == "" or y == "":
#             return None, None
#         # Word2Vec.wv.vocab is a dictionary: 'word': Vocab object
#         # Vocab object contains (count, index, sample_int)
#         x = [self.model.wv.vocab[word].index for word in x.split(' ')]
#         y = [self.model.wv.vocab[word].index for word in y.split(' ')]
#
#         # padding x
#         if len(x) > self.max_length-1:
#             x[self.max_length-1] = self.dictionary("<EOS>")
#             x = x[:self.max_length]
#         else:
#             x.append(self.dictionary("<EOS>"))
#             while len(x) < self.max_length:
#                 x.append(self.dictionary("<PAD>"))
#         # padding y
#         if len(y) > self.max_length-1:
#             y[self.max_length-1] = self.dictionary("<EOS>")
#             y = y[:self.max_length]
#         else:
#             y.append(self.dictionary("<EOS>"))
#             while len(y) < self.max_length:
#                 y.append(self.dictionary("<PAD>"))
#
#         return torch.LongTensor(x), torch.LongTensor(y)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.i < self.num_batch:
#             if (self.i+1) * self.batch_size <= self.num_data:
#                 batch_x = self.x[:, self.i*self.batch_size:(self.i+1) * self.batch_size]
#                 batch_y = self.y[:, self.i*self.batch_size:(self.i+1) * self.batch_size]
#             else:
#                 batch_x = self.x[:, self.i * self.batch_size:]
#                 batch_y = self.y[:, self.i * self.batch_size:]
#             self.i += 1
#             return batch_x, batch_y
#         else:
#             self.i = 0
#             raise StopIteration
#
#     def reset(self):
#         self.i = 0


class Dictionary:
    def __init__(self, word2vec_model):
        self.wv = word2vec_model.wv.syn0
        self.word2index = {key: value.index+4 for key, value in word2vec_model.wv.vocab.items()}
        self.index2word = {value: key for key, value in self.word2index.items()}

        # add <SOS>, <EOS>, <UNK>, <PAD> token to vocabulary
        self.word2index["<PAD>"] = 0
        self.word2index["<SOS>"] = 1
        self.word2index["<EOS>"] = 2
        self.word2index["<UNK>"] = 3
        self.index2word[0] = "<PAD>"
        self.index2word[1] = "<SOS>"
        self.index2word[2] = "<EOS>"
        self.index2word[3] = "<UNK>"

        # also, concatenate four random vector to word vector tensor
        self.wv = np.concatenate((np.random.rand(4, self.wv.shape[1]), self.wv), 0)

    def __call__(self, x):
        if isinstance(x, int):
            return self.index2word[x]
        elif isinstance(x, str):
            return self.word2index[x]


def make_sentence():
    sentence_file = open('data/clr_conversation_modified.txt', 'w+')
    with open("data/clr_conversation.txt", 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line != '+++$+++':
                sentence_file.write(line + '\n')
            else:
                pass

    sentence_file.close()


def make_train_pair():
    input_file = open('data/clr_input.txt', 'w+')
    output_file = open('data/clr_output.txt', 'w+')

    sentence1 = ""
    sentence2 = ""
    with open("data/clr_conversation.txt", 'r') as f:
        for line in f:
            if line != '+++$+++\n':
                sentence1 = sentence2
                sentence2 = line
            else:
                sentence1 = ""
                sentence2 = ""

            if sentence1 != "" and sentence2 != "":
                input_file.write(sentence1)
                output_file.write(sentence2)
    input_file.close()
    output_file.close()


def make_vocabulary(model):
    inverse_vocab = {id.index+4: word for word, id in model.wv.vocab.items()}
    inverse_vocab[0] = "<PAD>"
    inverse_vocab[1] = "<SOS>"
    inverse_vocab[2] = "<EOS>"
    inverse_vocab[3] = "<UNK>"
    # sorted by index
    od = OrderedDict(sorted(inverse_vocab.items()))
    with open('data/vocab.txt', 'w+') as f:
        for idx, word in od.items():
            f.write('{} {}\n'.format(idx, word))


def make_jieba_userdict(path='data/clr_conversation_modified.txt'):
    userdict = open('data/userdict.txt', 'w+')
    with open(path) as f:
        replaced = f.read().replace('\n', '@@').replace(' ', '@@').replace('@@', ' 1\n')
        userdict.write(replaced)


def train_word_vector(args):
    sentences = word2vec.LineSentence('data/clr_conversation_modified.txt')
    print("training word embedding..")
    start = time.time()
    # min count matters!
    model = word2vec.Word2Vec(sentences, size=args.word_dim, min_count=args.min_count)
    print("completed training word embedding!")
    end = time.time()
    print("training time: %2d:%2d" % ((end-start)//60, (end-start) % 60))

    model.save('model/word2vec.%dd' % args.word_dim)
    model.wv.save_word2vec_format('data/word2vec.%dd.txt' % args.word_dim, binary=False)

    return model


if __name__ == "__main__":
    # set logging config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

    # get arguments
    args = get_args()

    # build the txt file for training word2vec
    if args.make:
        make_sentence()
        make_train_pair()

    # train the model
    model = train_word_vector(args)

    if args.make:
        make_vocabulary(model)
