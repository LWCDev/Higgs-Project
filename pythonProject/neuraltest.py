import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Normalization())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics =['accuracy'])
#features = 25
dataset = pd.read_csv('300normalised.csv')
dataset_features = dataset.copy()
dataset_labels = dataset.iloc[:, 0]
dataset_labels = np.array(dataset_labels)
print(dataset_labels)
signal_mass = ['300', '400', '420', '440', '460', '500', '600', '700', '800',
               '900', '1000', '1200', '1400', '1600', '2000']
for i in range(len(signal_mass)):
    dataset = pd.read_csv(signal_mass[i]+'normalised.csv')
    dataset_features = dataset.copy()
    dataset_labels = dataset.iloc[:, 0]
    dataset_labels = np.array(dataset_labels)
    model.fit(dataset_features, dataset_labels, epochs=3)
print("Done Fitting")



list2 = []
for i in range(len(signal_mass)):
    dataset = pd.read_csv(signal_mass[i]+'normalised.csv')
    predictions = model.predict(dataset)
    print("Loop, " + signal_mass[i])
    for i in range(len(predictions)):
        list2.append(np.argmax(predictions[i]))
        #list2.append(predictions[i])
print("Done Looping")
plt.hist(list2, bins=101)
plt.show()
print(list2[1:2000])

