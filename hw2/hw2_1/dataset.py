import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from config import MAX_LENGTH, CROP_SIZE


class Corpus(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.word2index = {}
        self.index2word = {}
        self._load_vocab()
        self._load_feature_and_label()

    def __getitem__(self, idx):
        # start_point = random.randrange(0, 80 - CROP_SIZE)
        x = self.features[self.labels[idx][0]]
        y = self.labels[idx][1]
        # x = x[start_point:start_point+CROP_SIZE]
        # return x * (1 + 0.01* torch.randn(CROP_SIZE, 4096)), y
        return x, y

    def __len__(self):
        # return len(self.labels)
        return 5

    def _load_vocab(self):
        with open('model/vocab.txt', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                idx = line[0]
                word = line[1]
                self.word2index[word] = int(idx)
                self.index2word[int(idx)] = word

    def _load_feature_and_label(self):
        # id list
        with open(self.mode+'ing_data/id.txt', 'r') as f:
            id_list = [id for id in f.read().split('\n')[:-1]]
        features = []
        # load features
        for id in id_list:
            feature = np.load(self.mode+'ing_data/feat/'+id+'.npy')
            features.append(feature)
        self.features = torch.FloatTensor(features)

        # load labels
        data = json.load(open(self.mode+"ing_label.json"))
        labels = []
        for idx, data_dict in enumerate(data):
            for caption in data_dict['caption']:
                labels.append((idx, self._padding(caption)))
        self.labels = labels

    def _padding(self, caption):
        indexes = [self.word2index.get(word, 3) for word in caption.strip().split(' ')]
        if len(indexes) >= MAX_LENGTH:
            indexes = indexes[:MAX_LENGTH-1]
            indexes.append(2) # <EOS>
        else:
            indexes += [2] + [0] * (MAX_LENGTH - len(indexes) - 1)
        return torch.LongTensor(indexes)
