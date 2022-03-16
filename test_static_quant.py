from multiprocessing.spawn import prepare
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

#Create dataloaders
testloader = DataLoader(test_set, batch_size=1, shuffle=False)

#Load pretrained model
model_path = "model/CNN_best.pth"
model_fp32 = CNN_quant()
model_fp32.load_state_dict(torch.load(model_path))
model_fp32.eval()

#Attach global qconfig
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

#Fuse activations to preceding layers
#model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'relu1'], ['conv2', 'relu2']])
model_fp32_fused = model_fp32

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# Calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
with torch.no_grad():
    for data in testloader:
        #Get data
        input_fp32, label = data[0], data[1]
        model_fp32_prepared(input_fp32)
        break


# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
#Test loop
print(f"[INFO] Performing Inference on {len(test_set)} examples...")
correct = 0
correct_quant = 0
with torch.no_grad():
    for data in tqdm(testloader):
        #Get data
        input, label = data[0], data[1]

        #Predict
        pred = model_int8(input)
        
        #Softmax for final prediction and accumulate correct
        if torch.argmax(pred) == torch.argmax(label):
            correct +=1

    
    #Compute total correct guesses on test data
    test_acc_curr = correct / len(testloader)

print(f"Test accuracy: {test_acc_curr*100:.2f} %")


#Save quantized model
#TODO
print(model_fp32)
print(f"Float32 model: {model_fp32.conv1.weight[0]}")
print(f"Int8 model: {model_int8.conv1.weight().dequantize()[0]}")
#print(f"Int8 model: {model_int8.conv1.weight()}")