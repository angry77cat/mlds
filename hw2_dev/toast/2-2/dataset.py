import torch
import torch.utils.data as data
from config import MAX_LENGTH


class Conversation(data.Dataset):
    def __init__(self, x='data/clr_input.txt', y='data/clr_output.txt', vocab='data/vocab.txt'):
        print('loading data..')
        self.x = list(self.load_data(x))
        self.y = list(self.load_data(y))
        self.vocab = vocab
        self.num_data = len(self.x)

        print('loading vocabulary..')
        self.word2id, self.id2word = self.make_vocab()

    @staticmethod
    def load_data(path):
        with open(path) as f:
            for line in f.readlines():
                yield(line.strip().split(' '))

    def make_vocab(self):
        word2id, id2word = {}, {}
        with open(self.vocab, 'r') as f:
            for line in f:
                id, word = line.strip().split(' ')
                word2id[word] = int(id)
                id2word[int(id)] = word
        return word2id, id2word

    def __getitem__(self, id):
        x = [self.word2id.get(i, 3) for i in self.x[id]]
        y = [self.word2id.get(i, 3) for i in self.y[id]]

        # remove <UNK>
        x = list(filter(lambda i: i != 3, x))
        y = list(filter(lambda i: i != 3, y))

        # trim to max length
        if len(x) >= MAX_LENGTH:
            x[MAX_LENGTH-1] = 2 # <EOS>
            x = x[:MAX_LENGTH]
        else:
            x += [2] + [0] * (MAX_LENGTH-len(x)-1)

        if len(y) >= MAX_LENGTH:
            y[MAX_LENGTH-1] = 1
            y = y[:MAX_LENGTH]
        else:
            y += [2] + [0] * (MAX_LENGTH-len(y)-1)

        return torch.LongTensor(x), torch.LongTensor(y)

    def __len__(self):
        return self.num_data
