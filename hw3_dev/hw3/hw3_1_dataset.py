import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
from skimage import io, transform
import torchvision.transforms as transforms
import pickle

class ImageData(Dataset):
    def __init__(self, args, mode= 'default'):
        self.csvroot1 = args.datadir + 'tags_clean.csv'
        self.imagedir1 = args.datadir + 'faces/'
        self.csvroot2 = args.datadir + 'extra_data/tags.csv'
        self.imagedir2 = args.datadir + 'extra_data/images/'

        self.mode = mode
        self.args = args
        self.transforms = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

        self._load_labels()
        self._one_hot_labeling()

    def __getitem__(self,idx):
        x = self.feature_list[self.id_list[idx]]
        x = self.transforms(x)
        y = self.one_hot_label[self.id_list[idx]]
        return x, y

    def __len__(self):
        return len(self.feature_list)

    def _load_labels(self):
        if self.mode == 'default':
            csv_reader = pd.read_csv(self.csvroot1, sep=',', header = None)
            self.id_list = csv_reader.iloc[:,0].values
            self.label_list = csv_reader.iloc[:,1].values
            self.feature_list = []
            for idx in tqdm(self.id_list[0:1]): # change # of data
                image = io.imread(self.imagedir1 + str(idx) + '.jpg')
                image = transform.resize(image, (64,64))
                self.feature_list.append(np.array(image))
            self.feature_list = torch.FloatTensor(self.feature_list).permute(0,3,1,2)

        elif self.mode == 'extra':
            csv_reader = pd.read_csv(self.csvroot2, sep=',', header = None)
            self.id_list = csv_reader.iloc[:,0].values
            self.label_list = csv_reader.iloc[:,1].values
            self.feature_list = []
            for idx in tqdm(self.id_list[0:128]): # change # of data 36736
                image = io.imread(self.imagedir2 + str(idx) + '.jpg')
                image = transform.resize(image, (64,64))
                self.feature_list.append(np.array(image))
            # print(self.feature_list[0])
            self.feature_list = torch.FloatTensor(self.feature_list).permute(0,3,1,2)

    def _one_hot_labeling(self):
        i = 0
        label_dict = {}
        for label in self.label_list[:]:
            # print(label)
            hair_color = label.strip().split(' ')[0]
            eye_color = label.strip().split(' ')[2]
            if (hair_color, eye_color) not in label_dict:
                label_dict[(hair_color, eye_color)] = i
                i += 1

        # print(self.label_dict)
        n_comb = len(label_dict)
        with open('tag2onehot.pkl', 'wb') as f:
            pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)
        # with open('tag2onehot.txt', 'w+') as f:
        #     for i, j in label_dict.items():
        #         f.write("{}\t{}\n".format(i, j))
        # print(n_comb)
        self.one_hot_label = []
        # print(len(self.label_dict))
        for label in self.label_list[:]:
            hair_color = label.strip().split(' ')[0]
            eye_color = label.strip().split(' ')[2]
            index = label_dict[(hair_color, eye_color)]
            self.one_hot_label.append([0] * (index) + [1] + [0] * (n_comb-index-1))
        # print(self.one_hot_label[0])
        self.one_hot_label = torch.FloatTensor(self.one_hot_label)
            # self.one_hot_label.append(torch.zeros(n_comb).scatter_(label, 1))
        # print(self.one_hot_label[1])


def test():
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('-p', '--pretrain', action='store_true', help='continue previous model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--rmsprop_lr', type= float, default= 0.002, help='learing rate, default=0.002')
    parser.add_argument('--adam_lr', type=float, default=0.00002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--ngf', type=int, default=64, help='number of layers in G')
    parser.add_argument('--ndf', type=int, default=64, help='number of layers in D')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--datadir', default= 'data/')
    parser.add_argument('--modeldir', default= 'model/')
    parser.add_argument('--outdir', default= 'output/')
    opt = parser.parse_args()
    
    test = ImageData(opt, mode= 'extra')
    # dataloader = DataLoader(test, batch_size= opt.batch_size, shuffle= True)
    # for i, data in enumerate(dataloader, 0):
    #     print(i, data[1])

if __name__ == '__main__':
    test()
