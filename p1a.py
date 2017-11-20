import os
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, transform
from torch.autograd import Variable

#CONFIG
training_txt = 'train.txt'
testing_txt = 'test.txt'
root_dir = './lfw/'
image_size = (128,128)
plt.ion()	# interactive mode

class LFWDataset(Dataset):
	def __init__(self, txt_file, root_dir, transform=False):
		"""
		Args:
			txt_file (string): Path to the txt file with names of pictures
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional random transform to be applied on a sample.
		"""
		self.txt_file = pd.read_csv(txt_file, delim_whitespace=True, header=None)
		self.root_dir = root_dir    
		self.transform = transform

	def __len__(self):
		return len(self.txt_file)

	def __getitem__(self, idx):
		image1name = os.path.join(root_dir, txt_file.iloc[idx, 0])
		image2name = os.path.join(root_dir, txt_file.iloc[idx, 1])
		label = txt_file.iloc[idx, 2]
		#Resizes so all images are (128,128) as per architecture
		image1 = cv2.resize(io.imread(image1name), image_size)
		image2 = cv2.resize(io.imread(image2name), image_size)
		sample = {'image1': image1, 'image2': image2, 'label': label}
		return sample

		#ADD TRANSFORM IF STATEMENT HERE

class Siamese(nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		#Make two sequential models to in order to implement flatten using view()
		self.model = nn.Sequential(
			# 1
			nn.Conv2d(3, 64, 5, stride=(1,1), padding=2), #(in_channel size, out_channels size, kernel_size, stride, padding)
			# 2
			nn.ReLU(inplace=True),
			# 3
			nn.BatchNorm2d(64), #64 features
			# 4
			nn.MaxPool2d((2,2)), #(kernel_size, stride= (default kernel_size))
			# 5
			nn.Conv2d(64, 128, 5, stride=(1,1), padding=2),
			# 6
			nn.ReLU(inplace=True),
			# 7
			nn.BatchNorm2d(128), #128 features
			# 8
			nn.MaxPool2d((2,2)),
			# 9
			nn.Conv2d(128, 256, 3, stride=(1,1), padding=1),
			# 10
			nn.ReLU(inplace=True),
			# 11
			nn.BatchNorm2d(256),
			# 12
			nn.MaxPool2d((2,2)),
			# 13
			nn.Conv2d(256, 512, 3, stride=(1,1), padding=1),
			# 14
			nn.ReLU(inplace=True),
			# 15
			nn.BatchNorm2d(512))

		#Feed flatten layer to here
		self.fullyconnectedmodel = nn.Sequential(
			# 17
			nn.Linear(131072, 1024),
			# 18
			nn.ReLU(inplace=True),
			# 19
			nn.BatchNorm1d(1024),
			)

		self.fullyconnectedmodelconcatenated = nn.Sequential(
			# 20
			nn.Linear(2048, 1),
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

lfw = LFWDataset(txt_file=training_txt, root_dir=root_dir)

dataloader = DataLoader(lfw, batch_size=8, shuffle=True, num_workers=4)

# LOSS
loss_fn = nn.BCELoss()
learning_rate = 1e-6

# HELPER 

def imshow(img):
	npimg = np.asarray(img)
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()



