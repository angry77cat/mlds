# Train at least 5 models with different training approach.
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data

import torchvision
import time
import numpy as np
import random

import matplotlib
matplotlib.use('Agg') # this backend for matplotlib can run in server
import matplotlib.pyplot as plt

# Hyperparameters

BATCH_SIZE = [8, 32, 64, 128, 256, 1024]
LR = 1e-4

EPOCH = 5
USE_CUDA = torch.cuda.is_available()


# Helper Function
def calculate_sharpness(model, loss_func, x, y, num_sample, epsilon):
    sharpness = -99999
    origin_loss = loss_func(model(x), y).data[0]
    model_sample = CNN()
    if USE_CUDA:
        model_sample = model_sample.cuda()
    state_dict = model.state_dict()
    weights = dict2array(state_dict)
    for i in range(num_sample):
        new_weights = (1 + epsilon * (2*random.random()-1)) * weights
        new_dict = array2dict(state_dict, new_weights)
        model_sample.load_state_dict(new_dict)
        loss = loss_func(model_sample(x), y).data[0]

        sharpness = max((loss - origin_loss)/origin_loss, sharpness)

    return sharpness


def dict2array(state_dict):
    weights = torch.zeros(1)
    if USE_CUDA:
        weights = weights.cuda()
    for key, values in state_dict.items():
        weights = torch.cat((weights, values.view(-1)), 0)
    return weights[1:]


def array2dict(state_dict, weights):
    count = 0
    new_dict = {}
    for key, values in state_dict.items():
        shape = values.shape
        count_ = 1
        for s in shape:
            count_ *= s
        weights_ = weights[count:count+count_].view(shape)
        new_dict[key] = weights_
        count += count_
    return new_dict


def test_dict(dict1, dict2):
    for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
        print('==========')
        print("layer:", key1)
        if (value1 != value2).sum() == 0:
            print('matched')
        else:
            print('not matched')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )  # m * 19 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


train_data = torchvision.datasets.MNIST(root="./data/mnist/",
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
validset = torchvision.datasets.MNIST(root="./data/mnist",
                                      train=False,
                                      transform=torchvision.transforms.ToTensor())

loaders = [data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True) for bs in BATCH_SIZE]

test_data = validset.test_data.type(torch.FloatTensor).unsqueeze(1)/255.
test_label = validset.test_labels

# set volatile = True!!
# just for evaluation
test_data = Variable(test_data, volatile=True)[:1000]
test_label = Variable(test_label, volatile=True)[:1000]

# for plotting the 'train loss' in the last part
data = train_data.train_data.type(torch.FloatTensor).unsqueeze(1)/255.
label = train_data.train_labels
data = Variable(data, volatile=True)[:1000]
label = Variable(label, volatile=True)[:1000]

if USE_CUDA:
    models = [CNN().cuda() for m in BATCH_SIZE]
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    data = data.cuda()
    label = label.cuda()
else:
    models = [CNN() for m in BATCH_SIZE]

optimizers = [optim.Adam(params=model.parameters(), lr=LR) for model in models]
loss_func = nn.CrossEntropyLoss()

# print(calculate_sharpness(models[0], loss_func, data, label, 0, 0.05))

# Train
print('training five models in different approch..')
start = time.time()
for idx, (model, optimizer, loader) in enumerate(zip(models, optimizers, loaders)):
    print("training #{} model".format(idx))
    for epoch in range(EPOCH):
        for x, y in loader:
            x, y = Variable(x), Variable(y)
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
            pred = model(x)
            optimizer.zero_grad()
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            pred = torch.max(pred, 1)[1].data.squeeze()
            acc = sum(pred == y.data) / float(y.shape[0])
            print("epoch: %2d | loss: %.4f| acc: %.4f%%" % (epoch, loss.data[0], acc * 100))

end = time.time() - start
print('total time cost: %2d:%2d' % (end // 60, end % 60))

sharpness = []
train_losses = []
test_losses = []
for model in models:
    sharp = calculate_sharpness(model, loss_func, data, label, 50, 0.05)
    train_loss = loss_func(model(data), label).data[0]
    test_loss = loss_func(model(test_data), test_label).data[0]
    sharpness.append(sharp)
    train_losses.append(np.log(train_loss))
    test_losses.append(np.log(test_loss))

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(BATCH_SIZE, train_losses, color='b', label='train loss')
ax.plot(BATCH_SIZE, test_losses, color='b', linestyle='--', label='test loss')
ax2.plot(BATCH_SIZE, sharpness, color='r', label='sharpness')

ax.set_ylabel('loss (log-scale)')
ax2.set_ylabel('sharpness')
ax.yaxis.label.set_color('blue')
ax2.yaxis.label.set_color('red')
ax.legend()
plt.xlabel('batch size')
plt.savefig('hw_1_3_3part2.png')
