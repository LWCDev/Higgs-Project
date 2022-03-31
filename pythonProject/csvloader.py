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
#signal_mass = ['300.csv', '400.csv', '420.csv', '440.csv', '460.csv', '500.csv', '600.csv', '700.csv', '800.csv',
#               '900.csv', '1000.csv', '1200.csv', '1400.csv', '1600.csv', '2000.csv']
signal_mass = ['300', '400', '420', '440', '460', '500', '600', '700', '800',
               '900', '1000', '1200', '1400', '1600', '2000']


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result





def dataload(name):
    data = pd.read_csv(name, index_col=[0])
    data.drop(columns='weight', inplace=True)
    data.drop(columns='MCChannelNumber', inplace=True)
    data.drop(columns='region', inplace=True)
    testdata = pd.read_csv(name, index_col=[0])
    testdata.drop(columns='weight', inplace=True)
    testdata.drop(columns='MCChannelNumber', inplace=True)
    testdata.drop(columns='region', inplace=True)
    data.drop(data.index[3801:], 0, inplace=True)
    testdata.drop(testdata.index[5430:], 0, inplace=True)
    testdata.drop(range(0,3800), 0, inplace=True)
    data_labels = data.copy()
    test_labels = testdata.copy()
    #data = np.array(data)
    #testdata = np.array(testdata)
    data_labels['nTags'].replace([1], [0], inplace=True)
    data_labels['nTags'].replace([2], [1], inplace=True)
    test_labels['nTags'].replace([1], [0], inplace=True)
    test_labels['nTags'].replace([2], [1], inplace=True)
    data_labels['nTags'].replace([3], [1], inplace=True)
    test_labels['nTags'].replace([3], [1], inplace=True)
    data_labels = data_labels.pop('nTags')
    test_labels = test_labels.pop('nTags')
    test_labels = test_labels.to_frame()
    data_labels = data_labels.to_frame()
    data.drop(columns='nTags', inplace=True)
    testdata.drop(columns='nTags', inplace=True)
    data.drop(columns='MV2c10B3', inplace=True)
    data.drop(columns='pTJ3', inplace=True)
    data.drop(columns='etaJ3', inplace=True)
    data.drop(columns='phiJ3', inplace=True)
    testdata.drop(columns='MV2c10B3', inplace=True)
    testdata.drop(columns='pTJ3', inplace=True)
    testdata.drop(columns='etaJ3', inplace=True)
    testdata.drop(columns='phiJ3', inplace=True)
    #data_labels = np.array(data_labels)
    #test_labels = np.array(test_labels)
    return(data, testdata,data_labels, test_labels)

#for filename in range(len(signal_mass)):
#    dataframes = dataload(signal_mass[filename]+".csv")
#    newnames = signal_mass[filename]+"datax.csv", signal_mass[filename]+"datay.csv", signal_mass[filename]+"testx.csv", signal_mass[filename]+"testy.csv"
#    x = dataframes[0]
 #   x = normalize(x)
#    #result = x.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
#    #result = x.apply(lambda iterator: ((iterator - iterator.mean()) / iterator.std()).round(2))
#    y = dataframes[2]
#    testx = dataframes[1]
#    testx = normalize(testx)
#    testy = dataframes[3]
#    x.to_csv(newnames[0], index=False, header=False)
#    y.to_csv(newnames[1], index=False, header=False)
#    #testx.drop(columns='Unnamed', inplace=True)
#    testx.to_csv(newnames[2], index=False, header=False)
#    testy.to_csv(newnames[3], index=False, header=False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
