## define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32,32,5)
        self.pool1 = nn.MaxPool2d(2,2)	
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64,64,5)
        self.pool2 = nn.MaxPool2d(2,2)	
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv6 = nn.Conv2d(128,128,5)
        self.pool3 = nn.MaxPool2d(2,2)	
        self.conv7 = nn.Conv2d(128, 256, 5)
        self.conv8 = nn.Conv2d(256,256,5)
        self.pool4 = nn.MaxPool2d(2,2)
        self.FC1   = nn.Linear(9216,1024)
        self.FC2   = nn.Linear(1024,68*2)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## Dropout
        self.Dropout = nn.Dropout(0.2)

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.pool1(F.relu(self.conv2(x)))

        x = self.conv3(x)
        x = self.pool2(F.relu(self.conv4(x)))
        
        x = self.conv5(x)
        x = self.pool3(F.relu(self.conv6(x)))

        x = self.conv7(x)
        x = self.pool4(F.relu(self.conv8(x)))

        x =x.reshape(x.shape[0],-1)

        ###print(x.shape)
        x =  self.FC1(x)
        x =  self.Dropout(x)
        x =  self.FC2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
