import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(6400, 256)
		self.fc2 = nn.Linear(256, 256)
		self.out = nn.Linear(256, 3)


	def forward(self, x):
		x = x.view(-1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.out(x)

		return F.softmax(x, dim=0)

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
