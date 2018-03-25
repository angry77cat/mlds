import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

import matplotlib
matplotlib.use('Agg') # this backend for matplotlib can run in server
import matplotlib.pyplot as plt


# at least 10 similar-structured model

class CNN(nn.Module):
    def __init__(self, i):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8 * i, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.BatchNorm2d(8 * i))
        # m * 16 * 14 * 14
        self.conv2 = nn.Sequential(nn.Conv2d(8 * i, 8 * (i+1), 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.BatchNorm2d(8 * (i+1)))
        # m * 32 * 7 * 7
        self.fc1 = nn.Linear(8 * (i+1) * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# hyper parameters
LR = 1e-4
EPOCH = 10
BATCH_SIZE = 32
NUM_MODEL = 25
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
model_list = [CNN(i+1) for i in range(NUM_MODEL)]

optimizer_list = [optim.Adam(params=model.parameters(), lr=LR) for model in model_list]
params_list = [count_parameters(model) for model in model_list]
loss_func = nn.CrossEntropyLoss()

if USE_CUDA:
    print('using cuda...')
    test_data = Variable(test_data.cuda(), volatile=True)
    test_label = Variable(test_label.cuda(), volatile=True)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for step, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
    print("model #%2d" % (step+1))
    if USE_CUDA:
        model.cuda()

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

        if epoch % 10 == 0:
            print('epoch: %3d | loss: %.4f' % (epoch, loss.data[0]))

    train_pred = torch.max(model(x), 1)[1].data.squeeze()
    test_pred = torch.max(model(test_data), 1)[1].data.squeeze()
    train_acc = sum(train_pred == y.data)/float(y.size(0))
    test_acc = sum(test_pred == test_label.data)/float(test_label.size(0))

    test_loss = loss_func(model(test_data), test_label)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(loss.data[0])
    test_loss_list.append(test_loss.data[0])

    print('move the model out from gpu...')
    model.cpu()

    # free gpu memory
    torch.cuda.empty_cache()


fig = plt.figure()
plt.scatter(params_list, train_acc_list, label='train')
plt.scatter(params_list, test_acc_list, label='test')
plt.title('accuracy')
plt.xlabel('parameters')
plt.ylabel('accuracy')
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
