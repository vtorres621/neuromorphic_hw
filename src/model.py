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

######################## CNN Model ########################
class CNN3(nn.Module):
    """ Custom model"""

    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(1,  32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(4*4*64, 10)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.maxpool(self.relu (self.conv2 (self.relu (self.conv1(x)))))
        x = self.maxpool(self.relu (self.conv4 (self.relu (self.conv3(x)))))
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output

############################################################


# ##################### CNN Model Quantized ##################
# class CNN_quant(nn.Module):
#     """ Custom model"""

#     def __init__(self):
#         super(CNN_quant, self).__init__()
#         self.conv1 = nn.Conv2d(1,  32, 5)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.fc1 = nn.Linear(4*4*64, 10)
#         self.maxpool = nn.MaxPool2d(2)
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         # QuantStub converts tensors from floating point to quantized
#         self.quant1 = torch.quantization.QuantStub()
#         self.quant2 = torch.quantization.QuantStub()
#         self.quant3 = torch.quantization.QuantStub()
#         # DeQuantStub converts tensors from quantized to floating point
#         self.dequant = torch.quantization.DeQuantStub()


#     def forward(self, x):
#         # manually specify where tensors will be converted from floating
#         # point to quantized in the quantized model
#         x = self.quant1(x)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.dequant(x)
#         x = self.maxpool(x)
#         x = self.quant2(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.dequant(x)
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.quant3(x)
#         output = self.fc1(x)
#         output = self.dequant(output)
#         return output

# ############################################################

##################### CNN Model Quantized ##################
class CNN_quant(nn.Module):
    """ Custom model"""

    def __init__(self):
        super(CNN_quant, self).__init__()
        self.conv1 = nn.Conv2d(1,  32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(4*4*64, 10)
        self.maxpool = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        output = self.dequant(output)
        return output

############################################################