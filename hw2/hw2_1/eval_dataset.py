import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from eval_config import MAX_LENGTH, CROP_SIZE


class Corpus(Dataset):
    def __init__(self, dirname):
        self.dirname = dirname
        self.word2index = {}
        self.index2word = {}
        self._load_vocab()

    def _load_vocab(self):
        with open('stored_model/vocab.txt', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                idx = line[0]
                word = line[1]
                self.word2index[word] = int(idx)
                self.index2word[int(idx)] = word
