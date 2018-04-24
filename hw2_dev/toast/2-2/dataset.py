import torch
import torch.utils.data as data


class Conversation(data.Dataset):
    def __init__(self, x='data/clr_input.txt', y='data/clr_output.txt', vocab='data/vocab.txt'):
        self.x = list(self.load_data(x))
        self.y = list(self.load_data(y))
        self.vocab = vocab
        self.num_data = len(self.x)

        self.word2id, self.id2word = self.make_vocab()

    @staticmethod
    def load_data(path):
        with open(path) as f:
            for line in f.readlines():
                yield(line.strip().split(' '))

    def make_vocab(self):
        word2id, id2word = {}, {}
        with open(self.vocab) as f:
            for line in f:
                id, word = line.strip().split(' ')
                word2id[word] = int(id)
                id2word[int(id)] = word
        return word2id, id2word

    def __getitem__(self, id):
        x = [self.word2id[i] for i in self.x[id]]
        y = [self.word2id[i] for i in self.y[id]]
        return torch.LongTensor(x), torch.LongTensor(y)

    def __len__(self):
        return self.num_data
