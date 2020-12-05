# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:07:26 2020

@author: yudha
"""

import numpy as n
import matplotlib.pyplot as p
import pandas as pd

dataset = pd.read_csv('bank_customers.csv')

X = dataset.iloc[:, 3:13 ].values
y = dataset.iloc[:, 13 ].values

from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer as ct
Ubah_Geografi = le()
X[:, 1] = Ubah_Geografi.fit_transform(X[:, 1])
Ubah_Gender = le()
X[:, 2] = Ubah_Gender.fit_transform(X[:, 2])
Penyetaraan = ct([('Geography', ohe(), [1])], remainder="passthrough")
X = Penyetaraan.fit_transform(X[:,0:])
X = X[:, 1:]

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, activation = 'relu', input_dim = 11))

classifier.add(Dense(6, activation = 'relu'))

classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size= 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)