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


class Siamese(nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		# Conv2d parameters : (in_channel size, out_channels size, kernel_size, stride, padding)
		self.model = nn.Sequential(
			nn.Conv2d(3, 64, 5, stride=(1,1), padding=2)
			nn.ReLU(inplace=True) 




			)

# HELPER 

def imshow(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()



