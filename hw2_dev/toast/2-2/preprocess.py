import argparse
import logging
import time

import torch
import numpy as np
from gensim.models import word2vec


class Loader:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        self.i = 0
        self.x = torch.zeros(1)
        self.y = torch.zeros(1)
        self.load_data()
        self.num_data = self.x.shape[0]
        self.num_batch = self.num_data / self.batch_size

    def load_data(self):
        print('loading data from file..')
        with open("data/clr_conversation.txt", 'r') as f:
            sentence1 = ""
            sentence2 = ""
            for line in f:
                line = line.strip('\n')
                if line != '+++$+++':
                    sentence1 = sentence2
                    sentence2 = line
                    try:
                        x, y = self.make_pair(sentence1, sentence2)
                    except Exception:
                        pass
                    self.x = torch.cat([self.x, x], dim=0)
                    self.y = torch.cat([self.y, y], dim=0)
                else:
                    sentence1 = ""
                    sentence2 = ""
        self.x = self.x[1:]
        self.y = self.y[1:]

    def make_pair(self, x, y):
        if x == "":
            raise Exception
        # Word2Vec.wv.vocab is a dictionary: 'word': Vocab object
        # Vocab object contains (count, index, sample_int)
        x = [self.model.wv.vocab[word].index for word in x.split(' ')]
        y = [self.model.wv.vocab[word].index for word in y.split(' ')]
        return torch.LongTensor(x), torch.LongTensor(y)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.num_batch:
            if (self.i+1) * self.batch_size <= self.num_data:
                batch_x = self.x[self.i*self.batch_size:(self.i+1) * self.batch_size]
                batch_y = self.y[self.i*self.batch_size:(self.i+1) * self.batch_size]
            else:
                batch_x = self.x[self.i * self.batch_size:]
                batch_y = self.y[self.i * self.batch_size:]
            self.i += 1
            return batch_x, batch_y
        else:
            raise StopIteration

    def reset(self):
        self.i = 0


class Dictionary:
    def __init__(self, word2vec_model):
        self.wv = word2vec_model.wv.syn0
        self.word2index = {key: value.index for key, value in word2vec_model.wv.vocab.items()}
        self.index2word = {value: key for key, value in self.word2index.items()}

        # add <SOS>, <EOS>, <UNK>, <PAD> token to vocabulary
        self.word2index["<SOS>"] = len(self.word2index)
        self.index2word[len(self.index2word)] = "<SOS>"
        self.word2index["<EOS>"] = len(self.word2index)
        self.index2word[len(self.index2word)] = "<EOS>"
        self.word2index["<UNK>"] = len(self.word2index)
        self.index2word[len(self.index2word)] = "<UNK>"
        self.word2index["<PAD>"] = len(self.word2index)
        self.index2word[len(self.index2word)] = "<PAD>"

        # also, concatenate four random vector to word vector tensor
        self.wv = np.concatenate((self.wv, np.random.rand(4, self.wv.shape[1])), 0)

    def __call__(self, x):
        if isinstance(x, int):
            return self.index2word[x]
        elif isinstance(x, str):
            return self.word2index[x]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_dim", type=int, default=100, help="dimension of word embedding")

    return parser.parse_args()


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


def train_word_vector(args):
    sentences = word2vec.LineSentence('data/clr_conversation_modified.txt')
    print("training word embedding..")
    start = time.time()
    model = word2vec.Word2Vec(sentences, size=args.word_dim, min_count=1)
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
    make_sentence()

    # train the model
    model = train_word_vector(args)

