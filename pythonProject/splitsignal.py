import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

signal_mass = [300, 400, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result





def dataload(name, mass):
    data = pd.read_csv(name, index_col=[0])
    data = data.loc[data['mAtrue'] == mass]
    data.drop(columns="mAtrue", inplace=True)
    data = normalize(data)
    return(data)




for i in range(len(signal_mass)):
    print(signal_mass[i])
    data = dataload("background_1.csv", signal_mass[i])
    data.to_csv(str(signal_mass[i])+"true.csv", index=False, header=False)