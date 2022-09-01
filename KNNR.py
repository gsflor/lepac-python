import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#loading the dataset
dataset_Treino = np.genfromtxt("verde_8agoCarga_0m1aConvRPM_CSV.csv", delimiter=';')
dataset_Treino = dataset_Treino.astype(float)

dataset_Teste = np.genfromtxt("Ensaio_MotPreto22ago22_rmsM3geral_RPM_CSVa.csv", delimiter=';')
dataset_Teste = dataset_Teste.astype(float)

dataset_Teste2 = np.genfromtxt("amarelo_12agoCarga_0m2aConv2aRPM_CSVa.csv", delimiter=';')
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
inputs_Treino = scaled_dataset_Treino[:,0:13]
#outputs = scaled_dataset[:,4:5]
outputs_Treino = dataset_Treino[:,13:14]

#separando entradas e saidas
inputs_Teste = scaled_dataset_Teste[:,0:13]
#outputs = scaled_dataset[:,4:5]
outputs_Teste = dataset_Teste[:,13:14]

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


# example of making a prediction with the direct multioutput regression model
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from math import sqrt


# k-nearest neighbors for multioutput regression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


# define model
model = KNeighborsRegressor()

# fit model
model.fit(training_inputs, training_outputs)

y_pred = model.predict(test_inputs)

# summarize prediction
print(y_pred[19])
print(test_outputs[19])

mse = mean_squared_error(test_outputs, y_pred)
print(mse)

