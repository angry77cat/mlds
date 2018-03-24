import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

import matplotlib.pyplot as plt


# at least 10 similar-structured model

class CNN(nn.Module):
    def __init__(self, i):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 2 * i, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.BatchNorm2d(16))
        # m * 16 * 14 * 14
        self.conv2 = nn.Sequential(nn.Conv2d(2 * i, 2 * (i+1), 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.BatchNorm2d(32))
        # m * 32 * 7 * 7
        self.fc1 = nn.Linear(2 * (i+1) * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# hyper parameters
LR = 1e-4
EPOCH = 100
BATCH_SIZE = 32
USE_CUDA = torch.cuda.is_available()

# data
dataset = torchvision.datasets.MNIST(root='data/mnist',
                                     train=True,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True
                                     )
validset = torchvision.datasets.MNIST(root='data/mnist',
                                      train=False
                                      )
test_data = validset.test_data[:1000]
test_label = validset.test_labels[:1000]

test_data = test_data.unsqueeze(1).type(torch.FloatTensor)

loader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# create models
model_list = [CNN(i+1) for i in range(25)]

if USE_CUDA:
    print('using cuda...')
    for model in model_list:
        model.cuda()
optimizer_list = [optim.Adam(params=model.parameters(), lr=LR) for model in model_list]
params_list = [count_parameters(model) for model in model_list]
loss_func = nn.CrossEntropyLoss()

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for step, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
    for epoch in range(EPOCH):
        for x, y in loader:

            x, y = Variable(x), Variable(y)
            pred = model(x)
            optimizer.zero_grad()
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('epoch: %3d | loss: %.4f | acc: %.4f' % (epoch, loss.data[0]))

    train_pred = torch.max(model(x), 1)[1].data.squeeze()
    test_pred = torch.max(model(test_data), 1)[1].data.squeeze()
    train_acc = sum(train_pred == y)/float(y.size(0))
    test_acc = sum(test_pred == test_label)/float(test_label.size(0))

    test_loss = loss_func(model(test_data), test_label)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(loss.data[0])
    test_loss_list.append(test_loss.data[0])


fig = plt.fiqure()
plt.scatter(params_list, train_acc_list, label='train')
plt.scatter(params_list, test_acc_list, label='test')
plt.title('accuracy')
plt.xlabel('accuracy')
plt.ylabel('parameters')
plt.legend()
plt.savefig('acc.png')
plt.close()

fig = plt.figure()
plt.scatter(params_list, train_loss_list, label='train')
plt.scatter(params_list, test_loss_list, label='test')
plt.title('loss')
plt.xlabel('loss')
plt.ylabel('parameters')
plt.legend()
plt.savefig('loss.png')
plt.close()
