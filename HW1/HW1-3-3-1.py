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

import matplotlib
matplotlib.use('Agg') # this backend for matplotlib can run in server
import matplotlib.pyplot as plt

# Hyperparameters

BATCH_SIZE1 = 32
BATCH_SIZE2 = 1024
LR1 = 1e-4
LR2 = 1e-2
EPOCH = 20
USE_CUDA = torch.cuda.is_available()


# Helper Function

def compute_weight(model_1, model_2, alpha):
    dict1 = model_1.state_dict()
    dict2 = model_2.state_dict()
    dict_mix = {}
    for (key1, d1), (key2, d2) in zip(dict1.items(), dict2.items()):
        dict_mix[key1] = (1 - alpha) * d1 + alpha * d2
    return dict_mix


train_data = torchvision.datasets.MNIST(root="./data/mnist/",
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
validset = torchvision.datasets.MNIST(root="./data/mnist",
                                      train=False,
                                      transform=torchvision.transforms.ToTensor())

loader1 = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE1, shuffle=True)
loader2 = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE2, shuffle=True)

test_data = validset.test_data.type(torch.FloatTensor).unsqueeze(1)/255.
test_label = validset.test_labels

# set volatile = True!!
# just for evaluation
test_data = Variable(test_data, volatile=True)[:3000]
test_label = Variable(test_label, volatile=True)[:3000]

# for plotting the 'train loss' in the last part
data = train_data.train_data.type(torch.FloatTensor).unsqueeze(1)/255.
label = train_data.train_labels
data = Variable(data, volatile=True)[:3000]
label = Variable(label, volatile=True)[:3000]

# Define Network


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


model_1 = CNN()
model_2 = CNN()
model_mix = CNN()
model_mix.eval()

optimizer1 = optim.Adam(params=model_1.parameters(), lr=LR1)
optimizer2 = optim.Adam(params=model_2.parameters(), lr=LR2)
loss_func = nn.CrossEntropyLoss()

if USE_CUDA:
    model_1.cuda()
    model_2.cuda()
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    data = data.cuda()
    label = label.cuda()

# Train
print('training two model in different approch..')
start = time.time()
for epoch in range(EPOCH):
    pred1 = 0
    pred2 = 0
    for x1, y1 in loader1:
        x1, y1 = Variable(x1), Variable(y1)

        if USE_CUDA:
            x1 = x1.cuda()
            y1 = y1.cuda()

        # first model
        pred1 = model_1(x1)
        optimizer1.zero_grad()
        loss1 = loss_func(pred1, y1)
        loss1.backward()
        optimizer1.step()

    for x2, y2 in loader2:
        x2, y2 = Variable(x2), Variable(y2)

        if USE_CUDA:
            x2 = x2.cuda()
            y2 = y2.cuda()

        # second model
        pred2 = model_2(x2)
        optimizer2.zero_grad()
        loss2 = loss_func(pred2, y2)
        loss2.backward()
        optimizer2.step()

    if epoch % 5 == 0:
        pred1 = torch.max(pred1, 1)[1].data.squeeze()
        pred2 = torch.max(pred2, 1)[1].data.squeeze()
        acc1 = sum(pred1 == y1.data) / float(y1.shape[0])
        acc2 = sum(pred2 == y2.data) / float(y2.shape[0])
        print("epoch: %2d | loss1: %.4f | loss2: %.4f | acc1: %.4f%% | acc2: %.4f%%" % (epoch,
                                                                                        loss1.data[0],
                                                                                        loss2.data[0],
                                                                                        acc1 * 100,
                                                                                        acc2 * 100))

end = time.time() - start
print('total time cost: %2d:%2d' % (end // 60, end % 60))

model_1.eval()
model_2.eval()
pred1 = model_1(test_data)
pred2 = model_2(test_data)

loss1 = loss_func(pred1, test_label)
loss2 = loss_func(pred2, test_label)

pred1 = torch.max(pred1, 1)[1].data.squeeze()
pred2 = torch.max(pred2, 1)[1].data.squeeze()

acc1 = sum(pred1 == test_label.data) / float(test_label.shape[0])
acc2 = sum(pred2 == test_label.data) / float(test_label.shape[0])

print("accuracy of model 1: %.4f%% | 2: %.4f%%" % (acc1 * 100, acc2 * 100))
print("loss of model 1: %.4f | 2: %.4f" % (loss1.data[0], loss2.data[0]))

model_1 = model_1.cpu()
model_2 = model_2.cpu()
torch.cuda.empty_cache()
print("training complete!")

dict1 = model_1.state_dict()
dict2 = model_2.state_dict()

acc_list = []
loss_list = []
acc_train_list = []
loss_train_list = []
for alpha in np.arange(-1, 2, 0.1):
    print('alpha: %.1f' % alpha)
    dict_mix = compute_weight(model_1, model_2, alpha)
    model_mix.load_state_dict(dict_mix)

    if USE_CUDA:
        model_mix.cuda()

    # train
    pred = model_mix(data)
    loss = loss_func(pred, label)

    pred = torch.max(pred, 1)[1].data.squeeze()
    acc = sum(pred == label.data)/ float(label.shape[0])
    loss_train_list.append(loss.data[0])
    acc_train_list.append(acc)

    # test
    pred = model_mix(test_data)
    loss = loss_func(pred, test_label)

    pred = torch.max(pred, 1)[1].data.squeeze()
    acc = sum(pred == test_label.data) / float(test_label.shape[0])
    loss_list.append(loss.data[0])
    acc_list.append(acc)


with open('log_hw1_3_3.csv', 'w+') as f:
    f.write("loss,accuracy")
    for l, a in zip(loss_list, acc_list):
        f.write('%.4f,%.4f\n' % (l, a))

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(np.arange(-1, 2, 0.1), np.log(np.array(loss_list)), label="test loss", color='r')
ax.plot(np.arange(-1, 2, 0.1), np.log(np.array(loss_train_list)), label="train loss", color='r', linestyle='--')
ax2.plot(np.arange(-1, 2, 0.1), acc_list, label="test accuracy", color='b')
ax2.plot(np.arange(-1, 2, 0.1), acc_train_list, label="train accuracy", color='b', linestyle='--')

ax.set_ylabel('loss')
ax2.set_ylabel('accuracy')
ax.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')

plt.legend()
plt.title('batch size 32 vs 1024')
plt.savefig('hw_1_3_3.png')
