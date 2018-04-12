import re
import torch
from torch.autograd import Variable
import torchwordemb


class Dictionary:
    # use torchwordemb to load word vector
    # https://github.com/iamalbert/pytorch-wordemb
    def __init__(self, pretrain=None, word_vector_path=None):
        self.word2index = {}
        self.index2word = {}
        self.size = 0

        if pretrain is 'glove':
            words, tensor = torchwordemb.load_glove_text(word_vector_path)
            self.word2index = words
            self.word_vec = torch.cat((tensor, torch.rand(4, 300)), 0)

            self.index2word = {}
            for key, value in self.word2index.items():
                self.index2word[value] = key
            self.size = len(self.word2index)

        elif pretrain is 'word2vec':
            words, tensor = torchwordemb.load_word2vec_text(word_vector_path)
            self.word2index = words
            self.word_vec = tensor

            self.index2word = {}
            for key, value in self.word2index.items():
                self.index2word[value] = key
            self.size = len(self.word2index)

        self.add_word("<BOS>")
        self.add_word("<EOS>")
        self.add_word("<PAD>")
        self.add_word("<UNK>")

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

    def make_variable(self, sentence, max_length=None):
        words = self.sentence2words(sentence)
        # clip the sentence
        if max_length is not None:
            if len(words) >= max_length:
                words = words[:max_length]
        indexes = [self.word2index.get(word, self.word2index["<UNK>"]) for word in words]  # if word not in dict return <UNK>
        indexes.append(self.word2index["<EOS>"])   # add <EOS>
        # padding
        if max_length is not None:
            while(len(indexes) < max_length):
                indexes.append(self.word2index["<PAD>"])

        if torch.cuda.is_available():
            return Variable(torch.LongTensor(indexes).view(-1, 1)).cuda()
        else:
            return Variable(torch.LongTensor(indexes).view(-1, 1))

    def __call__(self, x):
        if isinstance(x, int):
            return self.index2word[x]
        elif isinstance(x, str):
            return self.word2index[x]


class Dataset:
    pass

def test_dictionary():
    d = Dictionary(pretrain='glove', word_vector_path='data/word_vector/glove.6B.300d.txt')

    print(d.make_variable('hey! how are you?'))
    print(d.size)


def test_dataset():
    pass

if __name__ == '__main__':
    test_dictionary()
    test_dataset()
