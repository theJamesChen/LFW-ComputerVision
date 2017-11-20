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

#CONFIG
training_txt = 'train.txt'
testing_txt = 'test.txt'
img_dir = './lfw/'


class LFW_Dataset(Dataset):
	def __init__(self, txt_path, randomtransform=False):
		self.txt_path = txt_path    
		self.randomtransform = randomtransform

class Siamese(nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		#Make two sequential models to in order to implement flatten using view()
		self.model = nn.Sequential(
			# 1
			nn.Conv2d(3, 64, 5, stride=(1,1), padding=2) #(in_channel size, out_channels size, kernel_size, stride, padding)
			# 2
			nn.ReLU(inplace=True)
			# 3
			nn.BatchNorm2d(64) #64 features
			# 4
			nn.MaxPool2d((2,2)) #(kernel_size, stride= (default kernel_size))
			# 5
			nn.Conv2d(64, 128, 5, stride=(1,1), padding=2)
			# 6
			nn.ReLU(inplace=True)
			# 7
			nn.BatchNorm2d(128) #128 features
			# 8
			nn.MaxPool2d((2,2))
			# 9
			nn.Conv2d(128, 256, 3, stride=(1,1), padding=1)
			# 10
			nn.ReLU(inplace=True)
			# 11
			nn.BatchNorm2d(256)
			# 12
			nn.MaxPool2d((2,2))
			# 13
			nn.Conv2d(256, 512, 3, stride=(1,1), padding=1)
			# 14
			nn.ReLU(inplace=True)
			# 15
			nn.BatchNorm2d(512))

		#Feed flatten layer to here
		self.fullyconnectedmodel = nn.Sequential(
			# 17
			nn.Linear(131072, 1024)
			# 18
			nn.ReLU(inplace=True)
			# 19
			nn.BatchNorm1d(1024)
			)

		self.fullyconnectedmodelconcatenated = nn.Sequential(
			# 20
			nn.Linear(2048, 1)
			# 21
			nn.Sigmoid()
			)
	def forward(self, x, y):
		x = self.model(x)
		y = self.model(y)
		# 16
		x = x.view(-1, 16*16*512)
		y = y.view(-1, 16*16*512)
		x = self.fullyconnectedmodel(x)
		y = self.fullyconnectedmodel(y)
		# Combine f1 and f2 through concatenation
		xy = torch.cat((x,y), 1)
		output = self.fullyconnectedmodelconcatenated(xy)
		
		return output

# LOSS
loss_fn = nn.BCELoss()
learning_rate = 1e-6

# HELPER 

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()



