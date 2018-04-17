
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import sys 

# In[2]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# ## Generate data

# In[46]: Define functions

def sample_func1(x):
    return np.sinc(x)
def sample_func2(x):
    y= []
    for i in x:
        y.append(1 if np.abs(i)<=5 else 0)
    return y


# In[47]: 

np.random.seed(1)
torch.manual_seed(1)
LR = [5e-3, 5e-3, 5e-3]
EPOCH = 2000

train_x1 = (2 * np.random.rand(2000)-1) * 10
train_y1 = sample_func1(train_x1)
train_x1, train_y1 = Variable(torch.FloatTensor(train_x1)), Variable(torch.FloatTensor(train_y1))
train_x1 = train_x1.unsqueeze(1)
train_y1 = train_y1.unsqueeze(1)

train_x2= (2 * np.random.rand(2000)-1) * 10
train_y2 = sample_func2(train_x2)
train_x2, train_y2 = Variable(torch.FloatTensor(train_x2)), Variable(torch.FloatTensor(train_y2))
train_x2 = train_x2.unsqueeze(1)
train_y2 = train_y2.unsqueeze(1)


# In[48]: Plot the sample functions

plt.scatter(train_x1.data, train_y1.data)
plt.title('Sample function 1')
plt.ylim((-1.25,1.25))
plt.show()
plt.scatter(train_x2.data, train_y2.data)
plt.title('Sample function 2')
plt.ylim((-1.25,1.25))
plt.show()


# ## Define Network Structure

# In[6]:

class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()
        self.fc1 = nn.Linear(1, 1880)
        self.fc2 = nn.Linear(1880, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class Deep(nn.Module):
    def __init__(self):
        super(Deep, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
    
class Deeper(nn.Module):
    def __init__(self):
        super(Deeper, self).__init__()
        params1 = [1, 14, 20, 30, 30, 30, 30, 30, 30]
        params2 = [14, 20, 30, 30, 30, 30, 30, 30, 1]
        
        self.fc1 = nn.Linear(params1[0], params2[0])
        self.fc2 = nn.Linear(params1[1], params2[1])
        self.fc3 = nn.Linear(params1[2], params2[2])
        self.fc4 = nn.Linear(params1[3], params2[3])
        self.fc5 = nn.Linear(params1[4], params2[4])
        self.fc6 = nn.Linear(params1[5], params2[5])
        self.fc7 = nn.Linear(params1[6], params2[6])
        self.fc8 = nn.Linear(params1[7], params2[7])
        self.fc9 = nn.Linear(params1[8], params2[8])    
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        return x


# ## Construct model and show the structures

# In[7]:
shallow = Shallow()
deep = Deep()
deeper = Deeper()
print(shallow)
print(deep)
print(deeper)


# ## count the parameters in each model

# In[8]:

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
shallow_count = count_parameters(shallow)
deep_count = count_parameters(deep)
deeper_count = count_parameters(deeper)
print("parameters in shallow model:")
print(shallow_count)
print("parameters in deep model:")
print(deep_count)
print("parameters in deeper model:")
print(deeper_count)


# ## Train (SampleFunction1)

# In[17]: Initial model and define optim and loss function
shallow = Shallow()
deep = Deep()
deeper = Deeper()
optim_shallow = optim.Adam(params=shallow.parameters(), lr=LR[0])
optim_deep = optim.Adam(params=deep.parameters(), lr=LR[1])
optim_deeper = optim.Adam(params=deeper.parameters(), lr=LR[2])
loss_func = nn.MSELoss()

# Start training
loss_shallow_list1 = []
loss_deep_list1 = []
loss_deeper_list1 = []
print('start')
for epoch in range(EPOCH):
    pred_shallow = shallow(train_x1)
    optim_shallow.zero_grad()
    loss_shallow = loss_func(pred_shallow, train_y1)
    loss_shallow.backward()
    optim_shallow.step()
    loss_shallow_list1.append(loss_shallow.data[0])
    
    pred_deep = deep(train_x1)
    optim_deep.zero_grad()
    loss_deep = loss_func(pred_deep, train_y1)
    loss_deep.backward()
    optim_deep.step()
    loss_deep_list1.append(loss_deep.data[0])
    
    pred_deeper = deeper(train_x1)
    optim_deeper.zero_grad()
    loss_deeper = loss_func(pred_deeper, train_y1)
    loss_deeper.backward()
    optim_deeper.step()
    loss_deeper_list1.append(loss_deeper.data[0])
    
    if epoch % 200 == 1:
        
        print('epoch: %4d | shallow loss: %.5f | deep loss: %.4f | deeper loss : %.4f' 
              % (epoch, loss_shallow.data[0], loss_deep.data[0], loss_deeper.data[0]))
        fig, ax = plt.subplots(1, 3, figsize=(14,6))
        ax[0].scatter(train_x1.data, train_y1.data, label="ground_truth")
        ax[0].scatter(train_x1.data, pred_shallow, label="shallow")
        ax[1].scatter(train_x1.data, train_y1.data, label="ground_truth")
        ax[1].scatter(train_x1.data, pred_deep, label="deep")
        ax[2].scatter(train_x1.data, train_y1.data, label="ground_truth")
        ax[2].scatter(train_x1.data, pred_deeper, label="deeper")
        ax[0].set_title("shallow model")
        ax[1].set_title("deep model")
        ax[2].set_title("deeper model")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()


# In[18]: Show training loss history

fig = plt.figure(figsize=(10,6))
num_show = 2000
plt.plot(np.arange(num_show), np.log(np.array(loss_shallow_list1[:num_show])), label='shallow')
plt.plot(np.arange(num_show), np.log(np.array(loss_deep_list1[:num_show])), label='deep')
plt.plot(np.arange(num_show), np.log(np.array(loss_deeper_list1[:num_show])), label='deeper')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('log(MSELoss)')
plt.legend()
plt.show()
        


# ## Train (SampleFunction2)

# In[49]: Initial model and define optim and loss function
shallow = Shallow()
deep = Deep()
deeper = Deeper()
optim_shallow = optim.Adam(params=shallow.parameters(), lr=LR[0])
optim_deep = optim.Adam(params=deep.parameters(), lr=LR[1])
optim_deeper = optim.Adam(params=deeper.parameters(), lr=LR[2])
loss_func = nn.MSELoss()

# Start training
print('start')
loss_shallow_list2 = []
loss_deep_list2 = []
loss_deeper_list2 = []
for epoch in range(EPOCH):
    pred_shallow = shallow(train_x2)
    optim_shallow.zero_grad()
    loss_shallow = loss_func(pred_shallow, train_y2)
    loss_shallow.backward()
    optim_shallow.step()
    loss_shallow_list2.append(loss_shallow.data[0])
    
    pred_deep = deep(train_x2)
    optim_deep.zero_grad()
    loss_deep = loss_func(pred_deep, train_y2)
    loss_deep.backward()
    optim_deep.step()
    loss_deep_list2.append(loss_deep.data[0])
    
    pred_deeper = deeper(train_x2)
    optim_deeper.zero_grad()
    loss_deeper = loss_func(pred_deeper, train_y2)
    loss_deeper.backward()
    optim_deeper.step()
    loss_deeper_list2.append(loss_deeper.data[0])
    
    if epoch % 200 == 1:
        
        print('epoch: %4d | shallow loss: %.5f | deep loss: %.4f | deeper loss : %.4f' 
              % (epoch, loss_shallow.data[0], loss_deep.data[0], loss_deeper.data[0]))
        fig, ax = plt.subplots(1, 3, figsize=(14,6))
        ax[0].scatter(train_x2.data, train_y2.data, label="ground_truth")
        ax[0].scatter(train_x2.data, pred_shallow, label="shallow")
        ax[1].scatter(train_x2.data, train_y2.data, label="ground_truth")
        ax[1].scatter(train_x2.data, pred_deep, label="deep")
        ax[2].scatter(train_x2.data, train_y2.data, label="ground_truth")
        ax[2].scatter(train_x2.data, pred_deeper, label="deeper")
        ax[0].set_title("shallow model")
        ax[1].set_title("deep model")
        ax[2].set_title("deeper model")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()


# Show training loss history

# In[50]: 

fig = plt.figure(figsize=(10,6))
num_show = 1000
plt.plot(np.arange(num_show), np.log(np.array(loss_shallow_list2[:num_show])), label='shallow')
plt.plot(np.arange(num_show), np.log(np.array(loss_deep_list2[:num_show])), label='deep')
plt.plot(np.arange(num_show), np.log(np.array(loss_deeper_list2[:num_show])), label='deeper')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('log(MSELoss)')
plt.legend()
plt.show()


