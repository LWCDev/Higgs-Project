import pandas as pd
import torch
import torch.nn as nn
import csv
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
# read_csv function which is used to read the required CSV file
data = pd.read_csv('x_train.csv')
data2 = pd.read_csv('y_train.csv')
dataxtest = pd.read_csv('x_test.csv')
dataytest = pd.read_csv('y_test.csv')
# display
#print("Original 'input.csv' CSV Data: \n")
#print(data)

# pop function which is used in removing or deleting columns from the CSV files
#data.pop('weight')

# display
#print("\nCSV Data after deleting the column 'year':\n")
#print(data)

class SignalDataSet(Dataset):
    def __init__(self, file, transform=None):
        #data loading
        xy = np.loadtxt(file, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 2:28])
        self.y = torch.from_numpy(xy[:, [29]]) #n_samples, y
        self.n_samples = xy.shape[0]
        self.transform = transform
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        #len(dataset)
        return self.n_samples
input_size = 26
num_classes = 2
model = nn.Linear(input_size, num_classes)

#print(data)
val1 = data._get_value(10, "10")
torch_tensor = torch.from_numpy(data.values)
#torch_tensor2 = torch.from_numpy(data['25'].values)
target = pd.DataFrame(data2['0'])
target2 = pd.DataFrame(dataytest['0'])
target = np.array(target)
target2 = np.array(target2)
#target = target.astype(float)
#print(target[0])
#print(type(target[0]))
#print(val1)
#print(torch_tensor)

train = torch.utils.data.TensorDataset(torch.Tensor(np.array(data)), torch.Tensor(target))
train_loader = torch.utils.data.DataLoader(train, batch_size = 10, shuffle=True)
print(train_loader)
signal, label = train[0]
#signal, label = train_loader[0]
print("Signal shape", signal.shape, "label", label)
input_dim = 26
hidden_dim = 100
output_dim = 2
learning_rate = 1e-1
n_epochs = 100

