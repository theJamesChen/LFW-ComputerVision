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
batch_size = 8
learning_rate = 1e-6
plt.ion()	# interactive mode

#SETUP DATASETS AND DATALOADERS
training_lfw = LFWDataset(txt_file=training_txt, root_dir=root_dir)
training_dataloader = DataLoader(training_lfw, batch_size=batch_size, shuffle=True, num_workers=8)

testing_lfw = LFWDataset(txt_file=testing_txt, root_dir=root_dir)
testing_dataloader = DataLoader(testing_lfw, batch_size=batch_size, shuffle=True, num_workers=8)

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
		image1name = os.path.join(self.root_dir, self.txt_file.iloc[idx, 0])
		image2name = os.path.join(self.root_dir, self.txt_file.iloc[idx, 1])
		label = self.txt_file.iloc[idx, 2]
		#Resizes so all images are (128,128) as per architecture
		image1 = cv2.resize(io.imread(image1name), image_size)
		# swap color axis because, numpy image: H x W x C, torch image: C X H X W
		image1 = np.transpose(image1, (2,0,1))
		image2 = cv2.resize(io.imread(image2name), image_size)
		image2 = np.transpose(image2, (2,0,1))
		sample = {'image1': image1, 'image2': image2, 'label': label}
		return sample

		#ADD TRANSFORM IF STATEMENT HERE

class Siamese(nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		#Make three sequential models to in order to implement flatten using view() and sigmoid activation
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


# LOSS
criterion = nn.BCELoss()
model = Siamese() # On CPU
model.float()
# model = Siamese().cuda() # On GPU
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

loss_history = []
iteration_history =[]
iteration_count = 0

def train(epoch):
	global loss_history, iteration_history, iteration_count

	#Setting network to train
	model.train()
	for batch_idx, data in enumerate(training_dataloader):
		# image1, image2, label = data['image1'].cuda(), data['image2'].cuda(), data['label'].cuda() # On GPU
		image1, image2, label = Variable(data['image1'].float()), Variable(data['image2'].float()), Variable(data['label'].float())
		#print image1
		output = model(image1,image2)
		#Zero the gradients
		optimizer.zero_grad()
		loss = criterion(torch.squeeze(output), label)
		loss.backward()
		optimizer.step()

		if batch_idx % 10 == 0:
			#print "Epoch %d, Batch Progress %d Loss %f" % (epoch, batch_idx, loss.data[0])
			print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image1), len(training_dataloader.dataset), 100. * batch_idx / len(training_dataloader), loss.data[0]))
			iteration_count += 10
			iteration_history.append(iteration_count)
			loss_history.append(loss.data[0])

#Training Model

for epoch in range(1, 3):
	train(epoch)

# Plot Loss

plt.plot(iteration,loss)
plt.show()


# Debug
# Show example batch

dataiter = iter(training_dataloader)
example_batch = next(dataiter)
concatenated = torch.cat((example_batch['image1'],example_batch['image2']),0)
grid = utils.make_grid(concatenated)
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.title('Batch from dataloader')
plt.axis('off')