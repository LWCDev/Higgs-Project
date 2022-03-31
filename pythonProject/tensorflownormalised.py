import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
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
# signals
signal_mass = ['300', '400', '420', '440', '460', '500', '600', '700', '800',
               '900', '1000', '1200', '1400', '1600', '2000']
# Configuration options
feature_vector_length = 21
num_classes = 2
#(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#print(Y_train.shape)
#Y_train = to_categorical(Y_train, 10)
#print(Y_train)
#print(Y_train.shape)

normlayer = tf.keras.layers.Normalization(axis=-1)
input_shape = (feature_vector_length,)
model = Sequential()
model.add(Dense(350, input_shape=input_shape, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(f'Feature shape: {input_shape}')
for name in signal_mass:
    train_x = np.array(pd.read_csv(name+"datax.csv"))
    test_x = np.array(pd.read_csv(name + "testx.csv"))
    train_y = np.array(pd.read_csv(name + "datay.csv"))
    test_y = np.array(pd.read_csv(name + "testy.csv"))
    #print(train_x.shape, test_x.shape)
    #print(train_y.shape, test_y.shape)
    print("Processing "+name)
    train_x = train_x.reshape(train_x.shape[0], feature_vector_length)
    test_x = test_x.reshape(test_x.shape[0], feature_vector_length)
    #normlayer.adapt(train_x)
    #train_x = normlayer(train_x)
    #test_x = normlayer(test_x)
    train_y = np.where(train_y==1, 0, 1)
    #print(train_y)
    #print(train_y)
    test_y = np.where(test_y==1, 0, 1)
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)
    #print(train_x.shape,train_y.shape)
    model.fit(train_x, train_y, epochs=50, batch_size=120, validation_split=0.2)

# Test the model after training
test_results = model.evaluate(test_x, test_y)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
#print(test_results[0], test_results[1])