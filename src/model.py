import torch
import torch.nn as nn

######################## CNN Model ########################
class CNN(nn.Module):
    """ Custom model"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,  32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(4*4*64, 10)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.maxpool(self.relu (self.conv1(x)))
        x = self.maxpool(self.relu (self.conv2(x)))
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output

############################################################