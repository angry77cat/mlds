
# coding: utf-8

# In[42]:

import numpy as np
import matplotlib.pyplot as plt
import sys 

# In[43]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

# In[62]:

BATCH_SIZE = 256
LR = 5e-4
EPOCH = 50

np.random.seed(1)
torch.manual_seed(1)
CUDA= torch.cuda.is_available()


# ## Load Data

# In[63]:

train_data = torchvision.datasets.MNIST(root="./data/mnist/",
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# ## Define Model Structure

# In[64]:

class CNN_shallow(nn.Module):
    def __init__(self):
        super(CNN_shallow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 66, 7, 1, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # m * 19 * 14 * 14
        self.fc1 = nn.Linear(66 * 14 * 14, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CNN_deep(nn.Module):
    def __init__(self):
        super(CNN_deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
         # m * 8 * 14 * 14
            nn.Conv2d(32, 48, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
         # m * 16 * 7 * 7
            nn.Conv2d(48, 72, 5, 2, 1),
            nn.ReLU(),
        ) # m * 39 * 3 * 3
        self.fc1 = nn.Linear(72 * 3 * 3, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CNN_deeper(nn.Module):
    def __init__(self):
        super(CNN_deeper, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # m * 5 * 14 * 14
            nn.Conv2d(20, 36, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # m * 10 * 7 * 7
            nn.Conv2d(36, 62, 5, 1, 1),
            nn.ReLU(),
            # m * 15 * 5 * 5
            nn.Conv2d(62, 74, 3, 1, 0),
            nn.ReLU(),
            # m * 18 * 3 * 3
            nn.Conv2d(74, 102, 1, 1, 0),
            nn.ReLU(),
            # m * 22 * 3 * 3

        )
        self.fc1 = nn.Linear(102 * 3 * 3, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# ## Set Optimizer and loss

# In[65]:

shallow = CNN_shallow()
deep = CNN_deep()
deeper = CNN_deeper()
if CUDA:
    shallow= shallow.cuda()
    deep= deep.cuda()
    deeper= deeper.cuda()
optim_shallow = optim.Adam(params=shallow.parameters(),lr=LR)
optim_deep = optim.Adam(params=deep.parameters(),lr=LR)
optim_deeper = optim.Adam(params=deeper.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

# ## Show the model structure

print(shallow)
print(deep)
print(deeper)


# ## Count the number of parameters in each model

# In[66]:

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_shallow = count_parameters(shallow)
count_deep = count_parameters(deep)
count_deeper = count_parameters(deeper)

print("parameters in shallow model:")
print(count_shallow)
print("parameters in deep model:")
print(count_deep)
print("parameters in deeper model:")
print(count_deeper)


# ## Train

# In[67]:

print('start')
loss_shallow_list = []
loss_deep_list = []
loss_deeper_list = []
acc_shallow_list= []
acc_deep_list= []
acc_deeper_list= []
for epoch in range(EPOCH):
    for idx, (x, y) in enumerate(train_loader):
        if CUDA:
            x, y= x.cuda(), y.cuda()
        x, y= Variable(x), Variable(y)
        #shallow
        pred_shallow = shallow(x)
        loss_shallow = loss_func(pred_shallow, y)
        optim_shallow.zero_grad()
        loss_shallow.backward()
        optim_shallow.step()
        loss_shallow_list.append(loss_shallow.data[0])
        
        #deep
        pred_deep = deep(x)
        loss_deep = loss_func(pred_deep, y)
        optim_deep.zero_grad()
        loss_deep.backward()
        optim_deep.step()
        loss_deep_list.append(loss_deep.data[0])
        
        #deeper
        pred_deeper = deeper(x)
        loss_deeper = loss_func(pred_deeper, y)
        optim_deeper.zero_grad()
        loss_deeper.backward()
        optim_deeper.step()
        loss_deeper_list.append(loss_deeper.data[0])

        #record acc
        acc_shallow_list.append(sum(torch.max(pred_shallow, 1)[1].data.squeeze() == y.data)/float(y.size(0)))
        acc_deep_list.append(sum(torch.max(pred_deep, 1)[1].data.squeeze() == y.data)/float(y.size(0)))
        acc_deeper_list.append(sum(torch.max(pred_deeper, 1)[1].data.squeeze() == y.data)/float(y.size(0)))
        
        if idx % 100 == 0:
            pred_shallow = torch.max(pred_shallow, 1)[1].data.squeeze()
            pred_deep = torch.max(pred_deep, 1)[1].data.squeeze()
            pred_deeper = torch.max(pred_deeper, 1)[1].data.squeeze()

            acc_shallow = sum(pred_shallow == y.data)/float(y.size(0))
            acc_deep = sum(pred_deep == y.data)/float(y.size(0))
            acc_deeper = sum(pred_deeper == y.data)/float(y.size(0))
            

            print("epoch: %2d | batch: %4d | shallow acc: %.4f | deep acc: %.4f | deeper acc: %.4f"
                 % (epoch, idx, acc_shallow, acc_deep, acc_deeper))
print('end of training')
        


# ## Loss\Acc Visualization

# In[70]:

num_show = 11750
plt.plot(np.arange(num_show), np.log(np.array(loss_shallow_list[:num_show])), label='shallow')
plt.plot(np.arange(num_show), np.log(np.array(loss_deep_list[:num_show])), label='deep')
plt.plot(np.arange(num_show), np.log(np.array(loss_deeper_list[:num_show])), label='deeper')
plt.title('Loss')
plt.xlabel('batch')
plt.ylabel('Log(loss)')
plt.legend()
plt.show()

plt.plot(np.arange(num_show), np.array(acc_shallow_list[:num_show]), label='shallow')
plt.plot(np.arange(num_show), np.array(acc_deep_list[:num_show]), label='deep')
plt.plot(np.arange(num_show), np.array(acc_deeper_list[:num_show]), label='deeper')
plt.title('Accuracy')
plt.xlabel('batch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

