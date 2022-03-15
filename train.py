from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.extract_dataset import extract_dataset
from src.dataset import MNIST_dataset
from src.model import CNN
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#Extract and load dataset 
train_set, val_set, test_set = extract_dataset()
train_set = MNIST_dataset(train_set)
val_set = MNIST_dataset(val_set)
test_set = MNIST_dataset(test_set)

#Load CUDA 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Using device: ", device)

#Create dataloaders
trainloader = DataLoader(train_set, batch_size=512, shuffle=False)
valloader = DataLoader(val_set, batch_size=512, shuffle=False)
testloader = DataLoader(test_set, batch_size=1, shuffle=False)

#Model
model = CNN().to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#Train
print("[INFO] Training... ")
train_loss = []
val_loss = []
test_accuracy = []
epochs = 20
best_accuracy = 0

for epoch in range(1, epochs+1):
    running_train_loss = 0
    running_val_loss = 0
    correct = 0

    #Loop through train data
    model.train()
    for data in trainloader:

        #Send data to GPU 
        input, label = data[0].to(device), data[1].to(device)

        #Zero gradients
        optimizer.zero_grad()

        #Forward
        pred = model(input)

        #Compute loss
        loss = criterion(pred, label)
        
        #Backprop
        loss.backward()

        #Optimize
        optimizer.step()

        #Accumulate loss
        running_train_loss += loss.item()

    #Loop through validation data 
    model.eval()
    with torch.no_grad():
        for data in valloader:
            
            #Send data to GPU 
            input, label = data[0].to(device), data[1].to(device)

            #Forward
            pred = model(input)

            #Compute loss
            loss = criterion(pred, label)

            #Accumulate loss
            running_val_loss += loss.item()

        for data in testloader:
            #Send data to GPU 
            input, label = data[0].to(device), data[1].to(device)

            #Predict
            pred = model(input)
            
            #Softmax for final prediction and accumulate correct
            if torch.argmax(pred) == torch.argmax(label):
                correct +=1

    #Compute total correct guesses on test data
    test_acc_curr = correct / len(testloader)

    #If best test results, save model
    if test_acc_curr > best_accuracy:
        best_accuracy = test_acc_curr
        torch.save(model.state_dict(), f"model/CNN_best.pth")

    #Compute epoch train & validation loss
    train_loss_curr = running_train_loss/len(trainloader)
    val_loss_curr = running_val_loss/len(valloader)
    
    #Save epoch train & validation loss / epoch accuracy
    train_loss.append(train_loss_curr)
    val_loss.append(val_loss_curr)
    test_accuracy.append(test_acc_curr)

    #Print results per epoch
    print(f"[{epoch:3d}/{epochs:3d}] Train loss: {train_loss_curr:.6f} | Val loss: {val_loss_curr:.6f} | Test Accuracy: {test_acc_curr*100:.2f}%")

#Draw and save loss & accuracy plots
fig, ax = plt.subplots(1,2, figsize = (20,8))
ax[0].plot(train_loss, label = 'Train Loss' )
ax[0].plot(val_loss, label = 'Validation Loss' )
ax[1].plot(test_accuracy)
ax[1].axhline(y=0.99, color = 'r', label =          f'Target Accuracy: 99.00%')
ax[1].axhline(y=best_accuracy, color = 'g', label = f'Best accuracy:    {best_accuracy*100:.2f}%')
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss Curve")
ax[0].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy Plot")
ax[1].legend()
plt.savefig("plots/CNN_loss.png")