import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(6400, 256)
		self.fc2 = nn.Linear(256, 256)
		self.out = nn.Linear(256, 3)
        
        # lecun initialization
		self.fc1.reset_parameters()
		self.fc2.reset_parameters()
		self.out.reset_parameters()


	def forward(self, x):
		x = x.view(-1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.out(x)

		return F.softmax(x, dim=0)

	######################
	# toast add, override
	######################
	def reset_parameters(self):
		stdv = 3. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		ng = 4
		self.main = nn.Sequential(
			nn.Conv2d(1, ng * 2, 3, 3, 0),
			nn.BatchNorm2d(ng * 2),
			nn.ReLU(),
			nn.Conv2d(ng * 2, ng * 4, 3, 2, 0),
			nn.BatchNorm2d(ng * 4),
			nn.ReLU(),
			nn.Conv2d(ng * 4, ng * 8, 3, 2, 0),
			nn.BatchNorm2d(ng * 8),
			nn.ReLU(),
			nn.Conv2d(ng * 8, ng * 16, 3, 2, 0),
			nn.BatchNorm2d(ng * 16),
			nn.ReLU()
			)
		self.fc1 = nn.Linear(ng*16*2*2, ng*8*2*2)
		self.fc2 = nn.Linear(ng*8*2*2, ng*4*2*2)
		self.fc3 = nn.Linear(ng*4*2*2, 3)


	def forward(self, x):
		x = x.squeeze().unsqueeze(0).unsqueeze(1)
		x = self.main(x)
		x = x.view(-1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.softmax(x, dim = 0)
		# print(x.shape)
		return x

class CNN2(nn.Module):
	def __init__(self):
		super(CNN2, self).__init__()
		ng = 8
		self.main = nn.Sequential(
			nn.Conv2d(1, ng * 2, 3, 3, 0),
			nn.BatchNorm2d(ng * 2),
			nn.ReLU(),
			nn.Conv2d(ng * 2, ng * 4, 3, 2, 0),
			nn.BatchNorm2d(ng * 4),
			nn.ReLU(),
			nn.Conv2d(ng * 4, ng * 8, 3, 2, 0),
			nn.BatchNorm2d(ng * 8),
			nn.ReLU(),
			nn.Conv2d(ng * 8, ng * 16, 3, 2, 0),
			nn.BatchNorm2d(ng * 16),
			nn.ReLU()
			)
		self.fc1 = nn.Linear(ng*16*2*2, ng*8*2*2)
		self.fc2 = nn.Linear(ng*8*2*2, ng*4*2*2)
		self.fc3 = nn.Linear(ng*4*2*2, 3)

		self.main[0].reset_parameters()
		self.main[3].reset_parameters()
		self.main[6].reset_parameters()
		self.main[9].reset_parameters()


	def forward(self, x):
		x = x.squeeze().unsqueeze(0).unsqueeze(1)
		x = self.main(x)
		x = x.view(-1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.softmax(x, dim = 0)
		# print(x.shape)
		return x

	######################
	# toast add, override
	######################
	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 3. / math.sqrt(n)
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(4608, 3)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))