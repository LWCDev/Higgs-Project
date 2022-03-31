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


import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers


import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile
masses =["300","400","420","440","460","500","600","700","800","900","1000","1200","1400","1600","2000"]
epochs = [250,200,250,200,250,250,1500,1000,200,2000,1500,1000,200,2000,1500,250,150,300,1200]

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)


FEATURES = 21

#gz = '2000normalised.csv'
#ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1))
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

#packed_ds = ds.batch(1000).map(pack_row).unbatch()

#for features,label in packed_ds.batch(1000).take(1):
#  print(features[0])
#  plt.hist(features.numpy().flatten(), bins = 101)
#  plt.show()

N_VALIDATION = int(1629)
N_TRAIN = int(3801)
BUFFER_SIZE = int(3801)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

#validate_ds = packed_ds.take(N_VALIDATION).cache()
#train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

#validate_ds = validate_ds.batch(BATCH_SIZE)
#train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, max_epochs, optimizer=None, **kwargs):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'], **kwargs)

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0
    ,**kwargs)
  return history


combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
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

#regularizer_histories = {}
#regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
#plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy', smoothing_std=10)

#plotter.plot(regularizer_histories)
#plt.ylim([0.4, 1.1])
#plt.show()

for i in range(len(masses)):
    gz = masses[i]+'normalised.csv'
    ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1))
    packed_ds = ds.batch(1000).map(pack_row).unbatch()
    N_VALIDATION = int(1629)
    N_TRAIN = int(3801)
    BUFFER_SIZE = int(3801)
    BATCH_SIZE = 500
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    validate_ds = packed_ds.take(N_VALIDATION).cache()
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

    validate_ds = validate_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    regularizer_histories = {}
    compile_and_fit(combined_model, "regularizers/combined", 75)
    #plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy', smoothing_std=10)
    #plotter.plot(regularizer_histories)
    #plt.ylim([0.4, 1.1])
    #print("This is for " + masses[i])
    #plt.show()


combined_model.save('saved_model3/my_model3')
