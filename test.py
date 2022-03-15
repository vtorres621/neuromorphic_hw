import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.extract_dataset import extract_dataset
from src.dataset import MNIST_dataset
from src.model import CNN, CNN_quant
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm

#Extract and load test dataset 
_, _, test_set = extract_dataset()
test_set = MNIST_dataset(test_set)

#Load CUDA 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("[INFO] Using device: ", device)

#Create dataloaders
testloader = DataLoader(test_set, batch_size=1, shuffle=False)

#Load pretrained model
model_path = "model/CNN_best.pth"
model = CNN_quant()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

#Test loop
print(f"[INFO] Performing Inference on {len(test_set)} examples...")
correct = 0
with torch.no_grad():
    for data in tqdm(testloader):
        #Send data to GPU 
        input, label = data[0].to(device), data[1].to(device)

        #Predict
        pred = model(input)
        
        #Softmax for final prediction and accumulate correct
        if torch.argmax(pred) == torch.argmax(label):
            correct +=1

    #Compute total correct guesses on test data
    test_acc_curr = correct / len(testloader)

print(f"Test accuracy: {test_acc_curr*100:.2f} %")

