import re
import time
import json
import random

import numpy as np
import torch
from torch.autograd import Variable


MAX_LENGTH = 40


class Dictionary:
    def __init__(self, model):
        self.word2index = {}
        self.index2word = {}
        self.size = 0

        # if pretrain is 'glove':
        #     print('loading pretrain gloVe word vector..')
        #     words, tensor = torchwordemb.load_glove_text(word_vector_path)
        #     self.word2index = words
        #     self.word_vec = torch.cat((tensor, torch.rand(4, word_dim)), 0)
        #
        #     self.index2word = {}
        #     for key, value in self.word2index.items():
        #         self.index2word[value] = key
        #     self.size = len(self.word2index)
        #
        # elif pretrain is 'word2vec':
        #     print('loading pretrain Word2Vec word vector..')
        #
        #     self.word2index = words
        #     self.word_vec = tensor
        #
        #     self.index2word = {}
        #     for key, value in self.word2index.items():
        #         self.index2word[value] = key
        #     self.size = len(self.word2index)

        self.word_vec = np.concatenate((np.random.randn(4, 256), model.wv.syn0), 0)

        self.add_word("<BOS>")
        self.add_word("<EOS>")
        self.add_word("<PAD>")
        self.add_word("<UNK>")

        for word, id in model.wv.vocab.items():
            self.word2index[word] = id.index+4
            self.index2word[id.index+4] = word
        self.size = len(self.word2index)

    def add_sentence(self, sentence):
        words = self.sentence2words(sentence)
        for word in words:
            self.add_word(word)
        return words

    def add_word(self, word):
        if word not in self.word2index.keys():
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.size += 1

    @staticmethod
    def sentence2words(sentence):
        sentence = re.sub("[^a-zA-Z]+", ' ', sentence)
        sentence = sentence.lower()
        words = sentence.strip().split(' ')
        return words

    def make_tensor(self, sentence, max_length=None):
        words = self.sentence2words(sentence)
        # clip the sentence
        if max_length is not None:
            if len(words) >= max_length:
                words = words[:max_length-1]
        indexes = [self.word2index.get(word, self.word2index["<UNK>"]) for word in words]  # if word not in dict return <UNK>
        indexes.append(self.word2index["<EOS>"])   # add <EOS>
        # padding
        if max_length is not None:
            while(len(indexes) < max_length):
                indexes.append(self.word2index["<PAD>"])

        if torch.cuda.is_available():
            return torch.LongTensor(indexes).view(-1, 1).cuda()
        else:
            return torch.LongTensor(indexes).view(-1, 1)

    def __call__(self, x):
        if isinstance(x, int):
            return self.index2word[x]
        elif isinstance(x, str):
            return self.word2index[x]


def load_features(train_list, mode='train'):
    """
    load the extracted features from the files in train_list

    :param: train_list: a list of npy file name
    :return: torch tensor, which has shape (num_file=1450, num_frame=80, num_dim=4096)
    """

    num_frame = 80
    num_dim = 4096
    # features = np.zeros((1, num_frame, num_dim))
    features = []
    print('loading features from files..')
    start = time.time()
    for id in train_list:
        feature = np.load("data/MLDS_hw2_1_data/"+mode+"ing_data/feat/" + id + ".npy")
        # feature = feature.reshape(1, num_frame, num_dim)
        # features = np.concatenate((features, feature), axis=0)
        features.append(feature)
        print("{:.1f}%".format(len(features)/len(train_list) * 100), end='\r')
    # features = features[1:]
    features = np.asarray(features)
    features *= 1 + 0.1* (np.random.randn()-0.5)
    end = time.time()
    print("loaded completed! time cost: {:2d}:{:2d}".format(int((end-start)//60), int((end-start) % 60)))
    if torch.cuda.is_available():
        return torch.FloatTensor(features).cuda()
    else:
        return torch.FloatTensor(features)


def load_labels(train_list, dictionary=None, max_length=None):
    data = json.load(open("data/MLDS_hw2_1_data/training_label.json"))
    # make a new dict to store id:caption relationship
    label_dict = {}
    for data_dict in data:
        label_dict[data_dict['id']] = random.sample(data_dict['caption'], 1)[0]
        # label_dict[data_dict['id']] = data_dict['caption'][0]

    # return a torch variable
    assert dictionary is not None, "argument should contain dictionary if to_variable=True"

    labels = torch.zeros((max_length, 1)).type(torch.LongTensor)
    if torch.cuda.is_available():
        labels = labels.cuda()
    for id in train_list:
        sentence = label_dict[id]
        label = dictionary.make_tensor(sentence, max_length)
        labels = torch.cat((labels, label), dim=1)

    return labels.transpose(0, 1)[1:], max_length


class Loader:
    def __init__(self, batch_size, dictionary, path="data/MLDS_hw2_1_data/training_id.txt"):
        self.batch_size = batch_size
        self.path = path
        self.dictionary = dictionary
        self.i = 0
        self.num_data = 0

        self.x, self.y = self.load()
        self.num_batch = self.num_data/self.batch_size

    # make this class an iterable
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

    def load(self):
        with open(self.path, 'r') as f:
            train_list = [id for id in f.read().split('\n')[:-1]]
        self.num_data = len(train_list)
        train_x = load_features(train_list)
        train_y, _ = load_labels(train_list,
                                 dictionary=self.dictionary, max_length=MAX_LENGTH)
        return train_x, train_y

    def reset(self):
        self.i = 0
        # change the label!
        with open(self.path, 'r') as f:
            train_list = [id for id in f.read().split('\n')[:-1]]
        self.y, _ = load_labels(train_list, dictionary=self.dictionary, max_length=MAX_LENGTH)




###########
# test code
###########
def test_load_labels():
    with open("data/MLDS_hw2_1_data/training_id.txt", 'r') as f:
        train_list = [id for id in f.read().split('\n')[:-1]]
    d = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.50d.txt')
    l, m = load_labels(train_list, True, d)
    print(l[:3])
    print(m)


def test_load_features():
    with open("data/MLDS_hw2_1_data/training_id.txt", 'r') as f:
        train_list = [id for id in f.read().split('\n')[:-1]]
    f = load_features(train_list)
    print(f.shape)


def test_dictionary():
    d = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.300d.txt')

    print(d.make_variable('hey! how are you?'))
    print(d.size)


def test_dataset():
    pass


if __name__ == '__main__':
    d = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.50d.txt')
    l = Loader(1, d)
    for idx, i in enumerate(l):
        print(i[0].shape, i[1].shape)
        if idx == 4:
            break
