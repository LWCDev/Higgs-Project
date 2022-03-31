import pandas as pd
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


def dataload(name):
    data = pd.read_csv(name)
    data.drop(columns='weight', inplace=True)
    data.drop(columns='MCChannelNumber', inplace=True)
    data.drop(columns='region', inplace=True)
    data.drop(data.index[5430:], 0, inplace=True)
    data_labels = data.copy()
    data = np.array(data)
    data_labels = data_labels.pop('nTags')
    return(data, data_labels)

normalize = layers.Normalization()

model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])
data = pd.read_csv('300.csv')
data.drop(columns='weight', inplace=True)
data.drop(columns='MCChannelNumber', inplace=True)
data.drop(columns='region', inplace=True)
x_val = data.copy()
y_val = data.copy()
y_val = y_val.pop('nTags')
data.drop(columns='nTags', inplace=True)
#x_val.drop(x_val.index[7001:9000],0,inplace=True)
data.drop(data.index[5430:7000], 0, inplace=True)
data_labels = data.copy()
data = np.array(data)
data_labels = data_labels.pop('regime')
x_val = np.array(x_val)
print(len(x_val))
print(len(y_val))
print(data.shape)
model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam(), metrics=[keras.metrics.MeanSquaredError()])


#model.fit(datasetread, datasetread_labels, epochs=5)

signal_mass = ['300.csv', '400.csv', '420.csv', '440.csv', '460.csv', '500.csv', '600.csv', '700.csv', '800.csv',
               '900.csv', '1000.csv', '1200.csv', '1400.csv', '1600.csv', '2000.csv']

for names in signal_mass:
    print("Currently training on " + names)
    features, labels = dataload(names)
    normalize.adapt(features)
    model.fit(features, labels, epochs=12)
    #, validation_data=(x_val,y_val))
print(model.summary())

print("Evaluate on test data")
data = pd.read_csv('300.csv')
data.drop(columns='weight', inplace=True)
data.drop(columns='MCChannelNumber', inplace=True)
data.drop(columns='region', inplace=True)
data.drop(data.index[5430:7000], 0, inplace=True)
data_labels = data.copy()
data = np.array(data)
data_labels = data_labels.pop('regime')
results = model.evaluate(data, data_labels, batch_size=18)
print("Test loss, test acc: ", results)