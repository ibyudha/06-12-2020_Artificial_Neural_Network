"""
Created on Sat Dec  5 19:07:26 2020

@author: Gus Yudha
"""

""" Import Dataset nasabah bank """
import pandas as pd
dataset = pd.read_csv('bank_customers.csv')
X = dataset.iloc[:, 3:13 ].values   # Pilah Fitur yang penting (Dari CreditScore - EstimatedSalary)
y = dataset.iloc[:, 13 ].values     # Pilah Jawaban (Exited)

""" Data preprocessing """
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer as ct
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1]) # Ubah nama negara menjadi numerik
X[:, 2] = le.fit_transform(X[:, 2]) # Ubah gender menjadi numerik
Setarakan = ct([('Pilah Jadi 3', ohe(), [1])], remainder="passthrough")
X = Setarakan.fit_transform(X[:,0:]) # Setarakan kategori negara
X = X[:, 1:]  # hilangkan 1 fitur variabel sampah

""" Pilah Data latihan dengan Data Ujian """
from sklearn.model_selection import train_test_split as tts
Soal_latihan, Soal_ujian, Jawaban_latihan, Jawaban_ujian = tts(X, y, test_size = 0.2, random_state = 0)

""" Standarisasi Soal latihan dan Soal Ujian """
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Soal_latihan = ss.fit_transform(Soal_latihan)  # Standarisasi Soal Latihan
Soal_ujian = ss.transform(Soal_ujian)          # Standarisasi Soal Ujian

""" Inisialisasi Arsitektur ANN (11-6-6-1) """
from keras.models import Sequential
from keras.layers import Dense as layerHiddennya
modelnya = Sequential()
modelnya.add(layerHiddennya(6, activation = 'relu', input_dim = 11))
modelnya.add(layerHiddennya(6, activation = 'relu'))
modelnya.add(layerHiddennya(1, activation = 'sigmoid'))
modelnya.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])
modelnya.fit(Soal_latihan, Jawaban_latihan, batch_size= 10, epochs = 100)

""" Evaluasi Arsitektur ANN """
from sklearn.metrics import confusion_matrix
prediksiNasabah = modelnya.predict(Soal_ujian)
hasilPrediksi = (prediksiNasabah > 0.5)
cm = confusion_matrix(Jawaban_ujian, hasilPrediksi)
print('Nilai Akurasi = ', (cm[0,0]+cm[1,1])/2000)

""" Prediksi tunggal terhadap serorang nasabah """

""" 1: Male, 0: Female """
""" 2: New Delhi, 1: Mumbai, 0: Agra"""

"""
Geography: Mumbai
Credit Score: 645
Gender: Male
Age: 40
Tenure: 3
Balance: 6000
Number of Product: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

import numpy as n
nasabahBaru = ss.transform(n.array([[1,0,   # Mumbai
                                     645,   # Credit Score
                                     1,     # Male
                                     40,    # Age
                                     3,     # Tenure
                                     6000,  # Balance
                                     2,     # Number of Product
                                     1,     # Has Credit Card: Yes
                                     1,     # Is Active Member: Yes
                                     50000  # Estimated Salary
                                     ]]))

hasilPrediksi = modelnya.predict(nasabahBaru)
prediksinya = (hasilPrediksi > 0.5)

if (prediksinya):
    print("Nasabah meninggalkan Bank")
else:
    print("Nasbah akan tetap di Bank")
