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
from tensorflow.keras.models import load_model
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

def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label


FEATURES = 21


importedmodel = load_model('saved_model3/my_model3', compile=False)


gz = '300normalised.csv'
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1))
packed_ds = ds.batch(1000).map(pack_row).unbatch()


train_ds = packed_ds.shuffle(3801).repeat().batch(500)

importedmodel.summary()