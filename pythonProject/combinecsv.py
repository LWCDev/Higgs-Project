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
    data.drop(columns="mAtrue", inplace=True)
    #testdata = pd.read_csv(name, index_col=[0])
    #testdata.drop(columns='weight', inplace=True)
    #testdata.drop(columns='MCChannelNumber', inplace=True)
    #testdata.drop(columns='region', inplace=True)
    #data.drop(data.index[3801:], 0, inplace=True)
    #testdata.drop(testdata.index[5430:], 0, inplace=True)
    #testdata.drop(range(0,3800), 0, inplace=True)
    data = normalize(data)
    #if name == "background_1.csv":
    #    data['new'] = 0
    #if name == "signal.csv":
    #    data['new'] = 1
    #test_labels = testdata.copy()
    #data = np.array(data)
    #testdata = np.array(testdata)
    #data_labels['nTags'].replace([1], [0], inplace=True)
    #data_labels['nTags'].replace([2], [1], inplace=True)
    #data_labels['nTags'].replace([3], [1], inplace=True)
    #data_labels['nTags'].replace([4], [0], inplace=True)
    #data_labels['nTags'].replace([5], [0], inplace=True)
    #data_labels['nTags'].replace([6], [1], inplace=True)
    #data_labels['nTags'].replace([7], [1], inplace=True)
    #data_labels['nTags'].replace([8], [0], inplace=True)
    #data_labels['nTags'].replace([9], [0], inplace=True)
    #ata_labels['regime'].replace([1], [0], inplace=True)
    #data_labels['regime'].replace([2], [1], inplace=True)
    #test_labels['nTags'].replace([1], [0], inplace=True)
    #test_labels['nTags'].replace([2], [1], inplace=True)
    #data_labels = data_labels.pop('regime')
    #data_labels=(data_labels-data_labels.min())/(data_labels.max()-data_labels.min())
    #test_labels = test_labels.pop('nTags')
    #test_labels = test_labels.to_frame()
    #data_labels = data_labels.to_frame()
    #data.drop(columns='regime', inplace=True)
    #data['MV2c10B3'].replace([-99], [-10], inplace=True)
    #testdata.drop(columns='nTags', inplace=True)
    #data.drop(columns='MV2c10B3', inplace=True)
    #data.drop(columns='pTJ3', inplace=True)
    #data['pTJ3'].replace([-99], [-10], inplace=True)
    #data.drop(columns='etaJ3', inplace=True)
    #data['etaJ3'].replace([-99], [-10], inplace=True)
    #data.drop(columns='phiJ3', inplace=True)
    #data['phiJ3'].replace([-99], [-10], inplace=True)
    #testdata.drop(columns='MV2c10B3', inplace=True)
    #testdata.drop(columns='pTJ3', inplace=True)
    #testdata.drop(columns='etaJ3', inplace=True)
    #testdata.drop(columns='phiJ3', inplace=True)
    #data_labels = np.array(data_labels)
    #test_labels = np.array(test_labels)

    return(data)


sb = ["signal.csv", "background_1.csv"]

#for filename in range(len(sb)):
   # dataframe = dataload(sb[filename])
    #newnames = signal_mass[filename]+"datax.csv", signal_mass[filename]+"datay.csv"
   # x = dataframes[0]
   # x = normalize(x)
    #result = x.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
    #result = x.apply(lambda iterator: ((iterator - iterator.mean()) / iterator.std()).round(2))
  #  y = dataframes[1]
 #   y = y.join(x)
    #testx = dataframes[1]
    #testx = normalize(testx)
    #testy = dataframes[3]
    #x.to_csv(newnames[0], index=False, header=False)#
#    y.to_csv(signal_mass[filename]+"normalised.csv", index=False, header=False)
    #testx.drop(columns='Unnamed', inplace=True)
    #testx.to_csv(newnames[2], index=False, header=False)
    #testy.to_csv(newnames[3], index=False, header=False)

#signal = dataload("signal.csv")
#background = dataload("background_1.csv")
#combined = signal.append(background, ignore_index=True)
#combined = combined.sample(frac=1)
#combined.to_csv("Combined.csv", index=False, header=False)
#signal.to_csv("NormSign.csv", index=False, header=False)
#background.to_csv("NormBack.csv", index=False, header=False)
signal_masses = [300, 400, 420, 440, 460, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000]
for i in range(len(signal_masses)):
    print(signal_masses[i])
    x = dataload(str(signal_masses[i])+".csv")
    #x = normalize(x)
    x.to_csv(str(signal_masses[i])+"normalised.csv", header=False, index=False)