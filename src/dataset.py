import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MNIST_dataset(Dataset):
    def __init__(self,set):
        self.inputs, self.labels = set
        

        #Convert to tensor
        self.inputs = torch.from_numpy(self.inputs)
        self.labels = torch.from_numpy(self.labels)
        
        #Create one-hot encoding and convert to float
        self.labels = nn.functional.one_hot(self.labels).float()

    def __len__(self):
        return len(self.inputs)

    
    def __getitem__(self,idx):
        #Reshape
        reshaped_input = self.inputs[idx].reshape((1,28,28))
        return reshaped_input, self.labels[idx]