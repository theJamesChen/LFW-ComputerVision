import os
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_float
from torch.autograd import Variable
import random
import argparse


# ******* CONFIG *******
class Config():
	training_txt = 'train.txt'
	testing_txt = 'test.txt'
	root_dir = './lfw/'
	image_size = (128,128)
	batch_size = 16
	learning_rate = 1e-6
	transform_probability = 0.7

#plt.ion()	# interactive mode

# ******* CLASSES *******

# ----------> ******* TRANSFORMATIONS *******

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image1 = image1.transpose((2, 0, 1))
		image2 = image2.transpose((2, 0, 1))
		if gpu:
			return {'image1': torch.from_numpy(image1.copy()).float(), 'image2': torch.from_numpy(image2.copy()).float(), 'label': label}
		return {'image1': torch.from_numpy(image1.copy()).float(), 'image2': torch.from_numpy(image2.copy()).float(), 'label': label}

class RandomHorizontalFlip(object):
	"""Horizontally flip the given sample randomly with a probability of 0.5."""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		if random.random() < 0.5:
			image1 = np.fliplr(image1)
		if random.random() < 0.5:
			image2 = np.fliplr(image2)
		return {'image1': image1, 'image2': image2, 'label': label}

class RandomVerticalFlip(object):
	"""Vertically flip the given sample randomly with a probability of 0.5."""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		if random.random() < 0.5:
			image1 = np.flipud(image1)
		if random.random() < 0.5:
			image2 = np.flipud(image2)
		return {'image1': image1, 'image2': image2, 'label': label}

class RandomRotationCenter(object):
	"""Rotates (+/- 30 degrees wrt the center) randomly with a probability of 0.5"""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		if random.random() < 0.5:
			# Between +/- 30 degrees
			theta1 = random.uniform(-30, 30)
			theta2 = random.uniform(-30, 30)
			# False so that rotated image does not exactly fits, pads with black
			image1 = transform.rotate(image1, theta1, resize=False, mode='constant')
			image2 = transform.rotate(image2, theta2, resize=False, mode='constant')
			return {'image1': image1, 'image2': image2, 'label': label}
		return sample

class RandomScaling(object):
	"""Scales (0.7 to 1.3) randomly with a probability of 0.5, (scales first and then center crop/pad to (128,128)"""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		if random.random() < 0.5:
			# Between 0.7 to 1.3
			scalingfactor1 = random.uniform(0.7, 1.3)
			scalingfactor2 = random.uniform(0.7, 1.3)
			th, tw = Config.image_size
			image1 = transform.rescale(image1, scalingfactor1, mode='constant')
			image2 = transform.rescale(image2, scalingfactor2, mode='constant')            
			if scalingfactor1 >= 1:
				h, w, c = image1.shape
				starth = int(round((h - th) / 2.))
				startw = int(round((w - tw) / 2.))
				image1 = image1[starth:starth+th,startw:startw+tw]
			else:
				# Calculates the padding needed for when scaling factor < 1
				h_rescale, w_rescale, c_rescale = image1.shape
				diff = th - h_rescale
				diff1, diff2 = diff//2, diff//2                
				if diff % 2 != 0:
					diff1, diff2 = diff//2, diff//2 + 1
				# npad is a tuple of (n_before, n_after) for each dimension
				npad = ((diff1, diff2), (diff1, diff2), (0, 0))
				image1 = np.pad(image1, npad , mode='constant')
			if scalingfactor2 >= 1:
				h, w, c = image2.shape
				starth = int(round((h - th) / 2.))
				startw = int(round((w - tw) / 2.))
				image2 = image2[starth:starth+th,startw:startw+tw]
			else:
				# Calculates the padding needed for when scaling factor < 1
				h_rescale, w_rescale, c_rescale = image2.shape
				diff = th - h_rescale
				diff1, diff2 = diff//2, diff//2                
				if diff % 2 != 0:
					diff1, diff2 = diff//2, diff//2 + 1
				# npad is a tuple of (n_before, n_after) for each dimension
				npad = ((diff1, diff2), (diff1, diff2), (0, 0))
				image2 = np.pad(image2, npad, mode='constant')
			return {'image1': image1, 'image2': image2, 'label': label}
		return sample

class RandomTranslation(object):
	"""Translates (-10 to 10) randomly with a probability of 0.5"""
	def __call__(self, sample):
		image1, image2, label = sample['image1'], sample['image2'], sample['label']
		if random.random() < 0.5:
			# Between -10 to 10
			x_translation, y_translation = random.uniform(-10, 10), random.uniform(-10, 10)
			image1 = transform.warp(image1, transform.AffineTransform(translation = (x_translation, y_translation)), mode='constant')
			x_translation, y_translation = random.uniform(-10, 10), random.uniform(-10, 10)
			image2 = transform.warp(image2, transform.AffineTransform(translation = (x_translation, y_translation)), mode='constant')
			return {'image1': image1, 'image2': image2, 'label': label}
		return sample

# ----------> ******* DATASET *******

class LFWDataset(Dataset):
	def __init__(self, txt_file, root_dir, transform):
		"""
		Args:
		txt_file (string): Path to the txt file with names of pictures
		root_dir (string): Directory with all the images.
		transform (callable, optional): Optional random transform to be applied on a sample:
			mirror flipping, rotation (+/- 30 degrees rotation wrt the center), translation (+/- 10 pixels), scaling (0.7 to 1.3)
			Any borders that don't have valid image data make black
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
			image1 = img_as_float(cv2.resize(io.imread(image1name), Config.image_size))
			image2 = img_as_float(cv2.resize(io.imread(image2name), Config.image_size))

			sample = {'image1': image1, 'image2': image2, 'label': np.uint8(label)}
			
			#Random transforms
			if self.transform:
				if random.random() < Config.transform_probability:
					composed = [RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotationCenter(), RandomScaling(), RandomTranslation()]
					random.shuffle(composed) #IN PLACE SHUFFLE
					#self.transform = transforms.Compose(composed)
					for t in composed:
						sample = t(sample)
					#sample = self.transform(sample)
					
			# swap color axis because, numpy image: H x W x C, torch image: C X H X W
			#image1, image2, label = sample['image1'], sample['image2'], sample['label']
			#image1 = np.transpose(image1, (2,0,1))
			#image2 = np.transpose(image2, (2,0,1))
			#print sample['image1'].shape
			#sample = {'image1': image1, 'image2': image2, 'label': label}  
			t = ToTensor()          
			sample = t(sample)
			#print sample['image1'].size
			#print sample['image2'] 
			#print sample['image2'].size
			return sample 

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

# ******* TRAINING *******

def train(epoch, randomTransform, savePath, gpu):
	'''randomTransform = TRUE for Data Augmentation '''
	# ******* MODEL PARAM SETUP *******
	print "<----------------", "Model Param Setup", "---------------->"
	if gpu:
		criterion = nn.BCELoss().cuda() #On GPU
		model = Siamese().cuda() # On GPU
	else:
		criterion = nn.BCELoss()# On CPU
		model = Siamese() # On CPU
	model.float()
	optimizer = optim.Adam(model.parameters(),lr = Config.learning_rate)
	print "<----------------", "Begin Training with Data Augmentation", randomTransform, "---------------->"
	loss_history = []
	iteration_history =[]
	iteration_count = 0

	# ******* SETUP DATASETS AND DATALOADERS *******
	print "<----------------", "Setup Datasets and Dataloaders", "---------------->"
	training_lfw = LFWDataset(txt_file=Config.training_txt, root_dir=Config.root_dir, transform=randomTransform)
	training_dataloader = DataLoader(training_lfw, batch_size=Config.batch_size, shuffle=True, num_workers=4)

	print "<----------------", "model.train() ON", "---------------->"
	for n_epoch in range(1, (epoch+1)):
		#Setting network to train
		model.train()
		for batch_idx, data in enumerate(training_dataloader):
			image1, image2, label = data['image1'], data['image2'], data['label']
			if gpu:
				image1, image2, label = image1.cuda(), image2.cuda(), label.cuda() # On GPU
			image1, image2, label = Variable(image1.float()), Variable(image2.float()), Variable(label.float())
			output = model(image1,image2)
			#Zero the gradients
			optimizer.zero_grad()
			loss = criterion(torch.squeeze(output), label)
			loss.backward()
			optimizer.step()

			if batch_idx % 10 == 0:
				#print "Epoch %d, Batch Progress %d Loss %f" % (n_epoch, batch_idx, loss.data[0])
				print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(n_epoch, (batch_idx) * len(image1), len(training_dataloader.dataset), 100. * (batch_idx) / len(training_dataloader), loss.data[0]))
				iteration_count += 10
				iteration_history.append(iteration_count)
				loss_history.append(loss.data[0])

	print "<----------------", "Epochs Ran", "---------------->"
	text = ["Training", "WithDataAugmentation:", str(randomTransform) ]
	savePlot(iteration_history, loss_history, text)
	print "<----------------", "Plot Saved", "---------------->"
	torch.save(model.state_dict(), savePath)
	print "<----------------", "Model Saved", "---------------->"


# ******* TESTING *******

def test(testfile, loadPath, gpu):
	'''testfile should be either Config.training_txt or Config.testing_txt '''
	# ******* MODEL PARAM SETUP *******
	print "<----------------", "Model Param Setup", "---------------->"
	if gpu:
		criterion = nn.BCELoss().cuda() #On GPU
		model = Siamese().cuda() # On GPU
	else:
		criterion = nn.BCELoss()# On CPU
		model = Siamese() # On CPU
	model.float()
	optimizer = optim.Adam(model.parameters(),lr = Config.learning_rate)

	print "<----------------", "Loading Saved Network Weights", "---------------->"
	model.load_state_dict(torch.load(loadPath))

	print "<----------------", "Begin Testing", "---------------->"
	# ******* SETUP DATASETS AND DATALOADERS *******
	print "<----------------", "Setup Datasets and Dataloaders", "---------------->"
	testing_lfw = LFWDataset(txt_file=testfile, root_dir=Config.root_dir, transform=False)
	testing_dataloader = DataLoader(testing_lfw, batch_size=Config.batch_size, shuffle=True, num_workers=4)
	print "<----------------", "Model.eval ON", "---------------->"
	model.eval()
	correct = 0
	for batch_idx, data in enumerate(testing_dataloader):
		image1, image2, label = data['image1'], data['image2'], data['label']
		if gpu:
			image1, image2, label = image1.cuda(), image2.cuda(), label.cuda() # On GPU
		image1, image2, label = Variable(image1.float(), volatile=True), Variable(image2.float(), volatile=True), Variable(label.float(), volatile=True)
		output = model(image1,image2)
		loss = criterion(torch.squeeze(output), label)

		if gpu:
			prediction = np.squeeze(output.cpu().data.numpy())
		else:
			prediction = np.squeeze(output.data.numpy())
		#Set > 0.5 to 1, < 0.5 to 0
		prediction[prediction > 0.5] = 1
		prediction[prediction <= 0.5] = 0

		#Batch labels
		correct += np.sum(np.equal(prediction, label.cpu()))
	
	percentcorrect = float(correct)/Config.batch_size/len(testing_dataloader)

	print "Accuracy:", percentcorrect
	print "<----------------", "Testing Complete", "---------------->"
	return percentcorrect

# ******* PLOT *******

def savePlot(iteration_history, loss_history, text):
	plt.plot(iteration_history,loss_history)
	title = "LossVsIterationFor " + text[0] + text[1] + text[2]
	plt.title(title)
	plt.ylabel('Loss')
	plt.xlabel('Iterations')
	#plt.show()
	savetitle = "LossIteration" + text[0] + text[1] + text[2]
	plt.savefig(savetitle, bbox_inches='tight')

# ******* PARSE ARGUMENTS *******

def parse_args():
	parser = argparse.ArgumentParser(description='James Chen: p1a')
	parser.add_argument('--epoch', type=int, default=20,
	                    help='Number of training EPOCH. Default is 20')
	parser.add_argument("--load", help="Automatically load the saved network weights from the file LOAD and test over both the train and test data, displaying accuracy statistics for both")
	parser.add_argument("--save", help="Train and save weight data into SAVE")
	# Switch
	parser.add_argument('--cpu', action='store_true',
						help='CPU mode ON')
	# Switch
	parser.add_argument('--transform', action='store_true',
						help='Data Augmentation ON')
	arg = parser.parse_args()
	return arg

def main():
	global gpu 
	args = parse_args()
	# Default Values
	gpu = True
	transform = False
	if args.cpu:
		print "<----------------", "CPU MODE", "---------------->"
		gpu = False
	else:
		print "<----------------", "GPU MODE", "---------------->"

	if args.transform:
		print "<----------------", "Data Augmentation ON", "---------------->"
		transform = True
	else:
		print "<----------------", "Data Augmentation OFF", "---------------->"
		transform = False

# ******* SAVE *******	
	if args.save is not None:
		 print "Train and save weight data into:", args.save, "with ", args.epoch, " epochs"
		 train(args.epoch, transform, args.save, gpu)
		 print "<----------------", "SAVE DONE", "---------------->"

# ******* LOAD *******	
	if args.load is not None:
		 print "Automatically load the saved network weights from the file ", args.load, "and test over both the train and test data, displaying accuracy statistics for both"
		 
		 print "<----------------", "Testing Training Data", "---------------->"
		 training_data_accuracy = test(Config.training_txt, args.load, gpu)
		 print "<----------------", "Testing Test Data", "---------------->"
		 testing_data_accuracy = test(Config.testing_txt, args.load, gpu)

		 print "<----------------", "Summary", "---------------->"
		 print "Training Accuracy", training_data_accuracy
		 print "Testing Accuracy", testing_data_accuracy
		 print "<----------------", "LOAD DONE", "---------------->"

if __name__ == '__main__':
	main()


# Debug
# Show example batch

# dataiter = iter(training_dataloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch['image1'],example_batch['image2']),0)
# grid = utils.make_grid(concatenated)
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.title('Batch from dataloader')
# plt.axis('off')