## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv_1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_4 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_5 = nn.Conv2d(256, 512, 1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.3)
        self.dropout_4 = nn.Dropout(0.4)
        self.dropout_5 = nn.Dropout(0.5)
        self.dropout_6 = nn.Dropout(0.6)
        
        self.fc1 = nn.Linear(in_features = 18432, out_features = 1000)  
        self.fc2 = nn.Linear(in_features = 1000, out_features =  500)  
        self.fc3 = nn.Linear(in_features =  500, out_features =   136)
        self.drop_1 = nn.Dropout(p=0.4)
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_1(self.pool(F.relu(self.conv_1(x))))
        x = self.dropout_2(self.pool(F.relu(self.conv_2(x))))
        x = self.dropout_3(self.pool(F.relu(self.conv_3(x))))
        x = self.dropout_4(self.pool(F.relu(self.conv_4(x))))
        x = self.dropout_5(self.pool(F.relu(self.conv_5(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout_6(x)
        x = F.relu(self.fc2(x))
        x = self.drop_1(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x