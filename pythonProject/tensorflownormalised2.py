import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from  IPython import display
from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import pathlib
import shutil
import tempfile

from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=10,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)


def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]










# signals
signal_mass = ['300', '400', '420', '440', '460', '500', '600', '700', '800',
               '900', '1000', '1200', '1400', '1600', '2000']
# Configuration options
feature_vector_length = 21
num_classes = 2

input_shape = (feature_vector_length,)


combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(feature_vector_length,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

combined_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=False, name='binary_crossentropy'),
                  'accuracy'])

print(f'Feature shape: {input_shape}')
for name in signal_mass:
    train_x = np.array(pd.read_csv(name+"datax.csv"))
    test_x = np.array(pd.read_csv(name + "testx.csv"))
    train_y = np.array(pd.read_csv(name + "datay.csv"))
    test_y = np.array(pd.read_csv(name + "testy.csv"))
    print("Processing "+name)
    #train_x - train_x.shuffle(400).repeat().batch(100)
    #train_x - train_x.shuffle(100).repeat().batch(25)
    train_x = train_x.reshape(train_x.shape[0], feature_vector_length)
    test_x = test_x.reshape(test_x.shape[0], feature_vector_length)
    train_y = np.where(train_y==1, 0, 1)
    test_y = np.where(test_y==1, 0, 1)
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)
    #print(train_x.shape,train_y.shape)
    combined_model.fit(train_x, train_y, epochs=25, batch_size=100, validation_split=0.2)

# Test the model after training
test_results = combined_model.evaluate(test_x, test_y)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
#print(test_results[0], test_results[1])