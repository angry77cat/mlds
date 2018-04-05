import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import copy

LR = 1e-4
EPOCH = 7000


def sample_func(x):
    return np.sinc(x)


def compute_weight(model1, model2, alpha):
    dict1 = model1.state_dict()
    dict2 = model2.state_dict()
    dict_mix = {}
    for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
        dict_mix[key1] = (1 - alpha) * value1 + alpha * value2
    return dict_mix


train_x = (2 * np.random.rand(2000) - 1) * 3
train_y = sample_func(train_x)
train_x, train_y = Variable(torch.FloatTensor(train_x)), Variable(torch.FloatTensor(train_y))
train_x = train_x.unsqueeze(1)
train_y = train_y.unsqueeze(1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


net = Net()
optimizer = optim.Adam(params=net.parameters(), lr=LR)
loss_func = nn.MSELoss()
net_origin = copy.deepcopy(net)

for epoch in range(EPOCH):
    pred = net(train_x)
    optimizer.zero_grad()
    loss = loss_func(pred, train_y)
    loss.backward()
    optimizer.step()
net_final = copy.deepcopy(net)

# plt.scatter(train_x.data, train_y.data)
# plt.scatter(train_x.data, net(train_x).data)
# plt.show()


"""
losses = []
scan_range = np.arange(-0.3, 1.1, 0.001)
for alpha in scan_range:
    net_mix = Net()
    dict_mix = compute_weight(net_origin, net_final, alpha)
    net_mix.load_state_dict(dict_mix)

    pred = net_mix(train_x)
    loss = loss_func(pred, train_y)
    losses.append(loss.data[0])
    
plt.plot(scan_range, losses)
plt.show()
"""

import random

N = 3
losses_1 = [[] for _ in range(N)]
losses_2 = [[] for _ in range(N)]
losses_3 = [[] for _ in range(N)]
idx = [[random.randint(0,63) for _ in range(4)] for i in range(N)]
scan_range = np.arange(0.5, 1.6, 0.02)
for ratio in scan_range:
    for i in range(N):
        net_final.fc1.weight[idx[i][0]].data.mul_(ratio)
        pred = net_final(train_x)
        loss = loss_func(pred, train_y)
        net_final.fc1.weight[idx[i][0]].data.mul_(1 / ratio)
        losses_1[i].append(loss.data[0])

        net_final.fc2.weight[idx[i][1], idx[i][2]].data.mul_(ratio)
        pred = net_final(train_x)
        loss = loss_func(pred, train_y)
        net_final.fc2.weight[idx[i][1], idx[i][2]].data.mul_(1 / ratio)
        losses_2[i].append(loss.data[0])

        net_final.fc3.weight[0, idx[i][3]].data.mul_(ratio)
        pred = net_final(train_x)
        loss = loss_func(pred, train_y)
        net_final.fc3.weight[0, idx[i][3]].data.mul_(1 / ratio)
        losses_3[i].append(loss.data[0])

for i in range(N):
    plt.plot(scan_range, losses_1[i])
    plt.plot(scan_range, losses_2[i])
    plt.plot(scan_range, losses_3[i])
plt.savefig("hw1-2-bonus.png")

