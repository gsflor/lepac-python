import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#loading the dataset
dataset_Treino = np.genfromtxt("8agoCarga_0m1aConv.csv", delimiter=';')
dataset_Treino = dataset_Treino.astype(float)

dataset_Teste = np.genfromtxt("Ensaio_MotPreto22ago22_rmsM3geralCSV.csv", delimiter=';')
dataset_Teste = dataset_Teste.astype(float)

dataset_Teste2 = np.genfromtxt("12agoCarga_0m2aConv2a.csv", delimiter=';')
dataset_Teste2 = dataset_Teste2.astype(float)

all_dataset = np.concatenate((dataset_Treino,dataset_Teste,dataset_Teste2),axis = 0)

random.shuffle(all_dataset)

#all_dataset = pd.DataFrame(all_dataset)

#all_dataset = all_dataset.sample(frac=1, replace=True, random_state=1)

#all_dataset = all_dataset.astype(float)


print(dataset_Treino.shape)
print(dataset_Treino)

print(dataset_Teste.shape)
print(dataset_Teste)

print(all_dataset.shape)
print(all_dataset)

#ploting 
#tempo=list(range(256))
#for i in range(0,256):     
#   tempo[i] = i

#plt.plot(tempo[0:256],dataset[0:256,0:1])
#plt.plot(tempo[0:256],dataset[0:256,1:2])
#plt.plot(tempo[0:256],dataset[0:256,2:3])
#plt.ylabel('Amp')
#plt.xlabel('Tempo')
#plt.legend(['Corrente A','Corrente B','Corrente C'], loc='upper left')
#plt.show()

from sklearn.preprocessing import MinMaxScaler

# rescaling dataset_Treino
scaler_Treino = MinMaxScaler()
scaler_Treino.fit(dataset_Treino)
scaled_dataset_Treino = scaler_Treino.transform(dataset_Treino)

scaler_Teste = MinMaxScaler()
scaler_Teste.fit(dataset_Teste)
scaled_dataset_Teste = scaler_Teste.transform(dataset_Teste)

# rescaling dataset_Treino
scaler_Treino_all = MinMaxScaler()
scaler_Treino_all.fit(all_dataset)
scaled_dataset_Treino_all = scaler_Treino.transform(all_dataset)

#separando entradas e saidas
inputs_Treino = scaled_dataset_Treino[:,0:12]
#outputs = scaled_dataset[:,4:5]
outputs_Treino = dataset_Treino[:,12:13]

#separando entradas e saidas
inputs_Teste = scaled_dataset_Teste[:,0:12]
#outputs = scaled_dataset[:,4:5]
outputs_Teste = dataset_Teste[:,12:13]

#separando entradas e saidas
inputs_Treino_all = scaled_dataset_Treino_all[:,0:12]
#outputs = scaled_dataset[:,4:5]
outputs_Treino_all = all_dataset[:,12:13]


print(inputs_Treino.shape)
print(outputs_Treino.shape)
print(inputs_Treino)
print(outputs_Treino)

print(inputs_Teste.shape)
print(outputs_Teste.shape)
print(inputs_Teste)
print(outputs_Teste)

from sklearn.model_selection import train_test_split

training_inputs,test_inputs,training_outputs,test_outputs = train_test_split(inputs_Treino_all,
                                                                             outputs_Treino_all,
                                                                             test_size=0.20,
                                                                             random_state=42)

print(training_inputs.shape)
print(test_inputs.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score

#convert y values to categorical values
#lab = preprocessing.LabelEncoder()
#training_outputsML = lab.fit_transform(training_outputs)
#test_outputsML = lab.fit_transform(test_outputs)
#print(training_outputs)
#print(training_outputsML)

#training_outputsML = outputs_Treino
#test_outputsML = inputs_Teste

#neigh = KNeighborsClassifier(n_neighbors=4)
#neigh.fit(inputs_Treino,outputs_Treino)
#training_y_pred = neigh.predict(inputs_Treino)

neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(training_inputs,training_outputs)
training_y_pred = neigh.predict(training_inputs)

scores = accuracy_score(training_outputs, training_y_pred)
print('Acurácia Treinamento')
print('%.4f' %scores)

test_y_pred = neigh.predict(test_inputs)
scores = accuracy_score(test_outputs, test_y_pred)
print('Acurácia Teste')
print('%.4f' %scores)

print(outputs_Teste)
print(test_y_pred)
