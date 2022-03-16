import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
model = CNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

module = model.conv1
#print(list(module.named_parameters()))
#print(list(module.named_buffers()))

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)


print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)

print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
        )
    )
)

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

