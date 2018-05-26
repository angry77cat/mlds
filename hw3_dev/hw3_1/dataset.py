import random
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
from skimage import io, transform
import torchvision.utils as vutils


class ImageData(Dataset):
    def __init__(self, args, mode= 'default'):
        self.csvroot1 = args.datadir + 'tags_clean.csv'
        self.imagedir1 = args.datadir + 'faces/'
        self.csvroot2 = args.datadir + 'extra_data/tags.csv'
        self.imagedir2 = args.datadir + 'extra_data/images'

        self.mode = mode
        self.args = args

        self._load_labels()

    def __getitem__(self,idx):
        x = self.feature_list[self.id_list[idx]]
        y = self.label_list[self.id_list[idx]]
        return x, y

    def __len__(self):
        return len(self.feature_list)

    def _load_labels(self):
        if self.mode == 'default':
            csv_reader = pd.read_csv(self.csvroot1, sep=',', header = None)
            self.id_list = csv_reader.iloc[:,0].values
            self.label_list = csv_reader.iloc[:,1].values
            self.feature_list = []

            for idx in tqdm(self.id_list[:]): # change # of data
                image = io.imread(self.imagedir1 + str(idx) + '.jpg')
                image = transform.resize(image, (64,64))
                self.feature_list.append(np.array(image))

            self.feature_list = torch.FloatTensor(self.feature_list).permute(0,3,1,2)

        elif self.mode == 'extra':
            csv_reader = pd.read_csv(self.csvroot2, sep=',', header = None)
            self.id_list = csv_reader.iloc[:,0].values
            self.label_list = csv_reader.iloc[:,1].values


if __name__ == '__main__':
    test()
