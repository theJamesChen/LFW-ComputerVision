import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

# Implementation of Flattten
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class Siamese(nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		#Make two sequential models to in order to implement flatten using view()
		self.model = nn.Sequential(
			nn.Conv2d(3, 64, 5, stride=(1,1), padding=2) #(in_channel size, out_channels size, kernel_size, stride, padding)
			nn.ReLU(inplace=True)
			nn.BatchNorm2d(64) #64 features
			nn.MaxPool2d((2,2)) #(kernel_size, stride= (default kernel_size))
			nn.Conv2d(64, 128, 5, stride=(1,1), padding=2)
			nn.ReLU(inplace=True)
			nn.BatchNorm2d(128) #128 features
			nn.MaxPool2d((2,2))
			nn.Conv2d(128, 256, 3, stride=(1,1), padding=1)
			nn.ReLU(inplace=True)
			nn.BatchNorm2d(256)
			nn.MaxPool2d((2,2))
			nn.Conv2d(256, 512, 3, stride=(1,1), padding=1)
			nn.ReLU(inplace=True)
			nn.BatchNorm2d(512))

		#Feed flatten layer to here
		self.fullyconnectedmodel = nn.Sequential(
			nn.Linear(131072, 1024)
			nn.ReLU(inplace=True)
			nn.BatchNorm2d(1024)
			)
	def forward(self, x):
		output = self.model(x)
		output = output.view(-1, 16*16*512)
		output = self.model(output)
		return output

# HELPER 

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()



