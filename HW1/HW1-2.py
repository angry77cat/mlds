# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.utils.data as data

import numpy as np
from sklearn.decomposition import PCA
import time

import matplotlib
matplotlib.use('Agg') # this backend for matplotlib can run in server
import matplotlib.pyplot as plt

# #### Hyprerparameters

np.random.seed(1)
torch.manual_seed(1)

NUM_MODEL = 8
NUM_MIN = 15 # for plotting minimal ratio
LR = 5e-3
EPOCH = 10

# for 2nd training
LR2 = 1e-4
EPOCH2 = 10
BATCH_SIZE = 32
USE_CUDA = torch.cuda.is_available()


# #### Generate Data
dataset = torchvision.datasets.MNIST(root='data/mnist',
                                     transform=torchvision.transforms.ToTensor(),
                                     train=True
                                     )
validset = torchvision.datasets.MNIST(root='data/mnist',
                                      transform=torchvision.transforms.ToTensor(),
                                      train=False
                                      )
loader = data.DataLoader(dataset=dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)


# #### Define Network Structure

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )  # m * 19 * 14 * 14

        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# #### Set Loss

loss_func = nn.CrossEntropyLoss()

# #### Some Helper Functions

def calculate_grad(model):
    grad_all = 0
    for layer in model.parameters():
        if layer.grad is not None:
            grad = (layer.grad.cpu().data.numpy() **2).sum()
            grad_all += grad
    return grad_all

# this function can add grad into computational graph
def calculate_grad2(model, loss):
    grad_norm = 0
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    return grad_norm

def minimal_ratio(matrix):
    w, _ = np.linalg.eig(matrix)
    return (w > 0).mean()

# dosent work ...
def calculate_hessian(loss, model):
    var = model.parameters()
    temp = []
    grads = torch.autograd.grad(loss, var, create_graph=True)
    grads = torch.cat([g.view(-1) for g in grads])
    for grad in grads:
        grad2 = torch.autograd.grad(grad, var, create_graph=True)
        temp.append(grad2)
    return np.array(temp)

# return a layer-structured hessian matrix (bad)
def calculate_hessian2(loss, model):
    var = [ i for i in model.parameters()]
    temp = []
    grads = torch.autograd.grad(loss, var, create_graph=True, retain_graph=True)
    for layer in grads:
        for grad in layer:
            for g in grad:
                temp2 = []
#                 g.backward(retain_graph=True) # not necessary
                for v in var:
                    grad2 = torch.autograd.grad(g, v, create_graph=True, retain_graph=True)
                    for grd in grad2:
                        temp2.append(grd)
                temp.append(temp2)
    return np.array(temp)

# return a 2D hessian matrix (good)
def calculate_hessian3(loss, model):
    # dont use gpu here, it takes too much memory
    var = [ i for i in model.parameters()]
    hessian = torch.zeros(1)

    grads = torch.autograd.grad(loss, var, create_graph=True, retain_graph=True)
    print('calculate hessian..')
    for layer in grads:
        for grad in layer:
            for ggg in grad:
                for gg in ggg:
                    for g in gg:
                        for v in var:
                            grad2 = torch.autograd.grad(g, v, create_graph=True, retain_graph=True)
                            for grd in grad2:
                                # make the
                                grd = grd.cpu()
                                grd.contiguous()
                                grd = grd.data.view(-1)
                                hessian = torch.cat((hessian, grd), 0)
    size = int(np.sqrt(hessian.size()))
    return np.array(hessian[1:].view(size, size))

def get_W(model):
    var = model.parameters()
    W = torch.zeros(1)
    if USE_CUDA:
        W = W.cuda()
    for v in var:
        W = torch.cat((W, v.data.view(-1)), 0)
    return W[1:].cpu().numpy()

# # 1. Visualize the Optimization Process

# manage models
model_list = [CNN() for i in range(NUM_MODEL) ]
if USE_CUDA:
    model_list = [model.cuda() for model in model_list]
optim_list = [optim.Adam(params=m.parameters(), lr=LR) for m in model_list]
Ws = []

for step, (model, optimizer) in enumerate(zip(model_list, optim_list)):
    print("training model #%d" % (step + 1))
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

        if epoch % 2 == 0:
            print("loss: %.6f" % loss.data[0])
            W = get_W(model)
            Ws.append(W)
Ws = np.array(Ws)


# #### Dimension Reduction by PCA

pca = PCA(n_components=2)
pca.fit(Ws)
Ws_2D = pca.transform(Ws)

for i in range(8):
    for j in range(5):
        plt.scatter(Ws_2D[i*5 + j][0], Ws_2D[i*5 + j][1], color='C%d' % i)
        plt.text(Ws_2D[i*5 + j][0], Ws_2D[i*5 + j][1], "%d" % j)
plt.savefig('hw1-2-mnist-1.png')
plt.close()


# # 2. Observe Gradient Norm During Training

def normal_train(verbose=False):
    net = CNN()
    if USE_CUDA:
        net.cuda()
    optimizer = optim.Adam(params=net.parameters(), lr=LR)
    grad_list = []
    loss_list = []
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):
            x, y = Variable(x), Variable(y)
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
            pred = net(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            grad_list.append(calculate_grad(net))
            loss_list.append(loss.data[0])

        if verbose:
            print('epoch: %4d | loss: %.6f | grad: %.6f' % (epoch, loss.data[0], calculate_grad(net)))

    return net, grad_list, loss_list



net, grad_list, loss_list = normal_train(verbose=True)


fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(len(grad_list)), (np.array(grad_list)))
ax[1].plot(np.arange(len(grad_list)), (np.array(loss_list)))
ax[0].set_ylabel('grad')
ax[1].set_ylabel('loss')
plt.savefig('hw1-2-mnist-2.png')
plt.close()



# # 3. What Happened When Gradient is Almost Zero

def grad_train(net, verbose=False):
    # minimize grad (change the loss to grad_norm)
    if USE_CUDA:
        net.cuda()
    grad_list = []
    loss_list = []
    optimizer = optim.Adam(net.parameters(), lr=LR2)
    for epoch in range(EPOCH2):
        for step, (x, y) in enumerate(loader):
            x, y = Variable(x), Variable(y)
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            pred = net(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()

            # instead of backward by loss, this time backward by grad_norm
            grad_norm = calculate_grad2(net, loss)
            grad_norm.backward(retain_graph=True)

            optimizer.step()

            grad_list.append(calculate_grad(net))
            loss_list.append(loss.data[0])

        if verbose:
            print('epoch: %4d | loss: %.6f | grad: %.6f' % (epoch, loss.data[0], calculate_grad(net)))

    return net, loss, grad_list, loss_list

losses = []
mini = []
for i in range(NUM_MIN):
    print("training... #%2d / %2d" % (i+1, NUM_MIN))
    start = time.time()
    net, _, _ = normal_train()
    net, loss, grad_list2, loss_list2 = grad_train(net)
    end_train = time.time()
    print('training: %2d:%2d' % ((end_train-start)//60, (end_train-start)%60))
    hessian = calculate_hessian3(loss, net)
    end_hessian = time.time()
    print('compute hessian: %2d:%2d' % ((end_hessian-end_train)//60, (end_hessian-end_train)%60))


    min_ratio = minimal_ratio(hessian)
    losses.append(loss.data[0])
    mini.append(min_ratio)

    print("remove model from gpu..")
    del net
    torch.cuda.empty_cache()



plt.scatter(mini, losses)
plt.xlabel('minimal ratio')
plt.ylabel('loss')
plt.ylim(-0.001, 0.005)
plt.savefig('hw1-2-mnist-3.png')
plt.close()

# #### (Note) the eigenvalues of hessian is full of zeros

# hessian = calculate_hessian3(loss, net)
# print(hessian.shape)
# min_ratio = minimal_ratio(hessian)
# print(min_ratio)


w, _ = np.linalg.eig(hessian)
print("number of total eigenvalue: ", w.size)
print("number of zero eigenvalue:", (w == 0).sum())
print("zero ratio : %.2f%%" % ((w == 0).sum()/w.size * 100))


print("elements in hessian: ", hessian.size)
print("zero in hessian: ", (hessian == 0).sum())
print("zero ratio: %.2f%%" % ((hessian == 0).sum()/hessian.size * 100))


net, _, _ = normal_train()
net, loss, grad_list2, loss_list2 = grad_train(net)

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(len(grad_list2)), (np.array(grad_list2)))
ax[1].plot(np.arange(len(grad_list2)), (np.array(loss_list2)))
ax[0].set_ylabel('grad')
ax[1].set_ylabel('loss')
plt.savefig('hw1-2-mnist-3.png')
plt.close()


# ## (Reference)To Calculate Higher Order Gradient
# https://github.com/pytorch/pytorch/releases/tag/v0.2.0
# ```python
# import torch
# from torchvision.models import resnet18
# from torch.autograd import Variable
#
# model = resnet18().cuda()
#
# # dummy inputs for the example
# input = Variable(torch.randn(2,3,224,224).cuda(), requires_grad=True)
# target = Variable(torch.zeros(2).long().cuda())
#
# # as usual
# output = model(input)
# loss = torch.nn.functional.nll_loss(output, target)
#
# grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
# # torch.autograd.grad does not accumuate the gradients into the .grad attributes
# # It instead returns the gradients as Variable tuples.
#
# # now compute the 2-norm of the grad_params
# grad_norm = 0
# for grad in grad_params:
#     grad_norm += grad.pow(2).sum()
# grad_norm = grad_norm.sqrt()
#
# # take the gradients wrt grad_norm. backward() will accumulate
# # the gradients into the .grad attributes
# grad_norm.backward()
#
# # do an optimization step
# optimizer.step()
# ```

# ## (Reference)
# https://discuss.pytorch.org/t/calculating-hessian-vector-product/11240/3
# ```python
# v = Variable(torch.Tensor([1, 1]))
# x = Variable(torch.Tensor([0.1, 0.1]), requires_grad=True)
# f = 3 * x[0] ** 2 + 4 * x[0] * x[1] + x[1] **2
# grad_f, = torch.autograd.grad(f, x, create_graph=True)
# z = grad_f @ v
# ```
#
# https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930/3
#
# :before view(-1), you should make a tensor contiguous (in memory)

# In[ ]:


##TEST SECTION##
# a = torch.FloatTensor([1])
# b = torch.FloatTensor([3])
#
# a, b = Variable(a, requires_grad=True), Variable(b, requires_grad=True)
#
# c = a + 3 * b**2
# c = c.sum()
#
# grad_b = torch.autograd.grad(c, b, create_graph=True)
# grad2_b = torch.autograd.grad(grad_b, b, create_graph=True)
#
# print(grad2_b)
