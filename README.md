# Neuromorphic Computing Hardware Design Final Project

## Requirements

If you want to run this code on your PC create an anaconda environment: 

`conda create --name nhw`

`conda activate  nhw`

Install the following packages:

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge tqdm`


## Training

Run `train.py` to start training.  
Script will automatically save the model that achieves the best test accuracy under `model/` directory.
Script will save test loss, validation loss and test accuracy under `plots/`.




