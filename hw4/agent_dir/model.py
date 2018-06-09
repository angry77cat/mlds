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
