import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.extract_dataset import extract_dataset
from src.dataset import MNIST_dataset
from src.model import CNN, CNN_quant
import matplotlib.pyplot as plt
#plt.style.use('seaborn')
from tqdm import tqdm
import numpy as np

#Extract and load test dataset 
_, _, test_set = extract_dataset()
test_set = MNIST_dataset(test_set)

#Load CUDA 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("[INFO] Using device: ", device)

#Create dataloaders
#testloader = DataLoader(test_set, batch_size=1, shuffle=False)

#print(len(test_set))
images = 0
for image, label in tqdm(test_set):
    images += image.squeeze()

print(images/len(test_set))
    
plt.imshow(images/len(test_set), cmap='gray')
plt.show()