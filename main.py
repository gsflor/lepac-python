import csv
import string
from _csv import writer
from mlxtend.plotting import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, preprocessing
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
import math
import pandas as pd

#escreve as correntes normalizadas no arquivo
def write_standards_currents(listA, listB, listC, classification):
    f = open('standard_results.txt', 'a')
    listA = [str(x) for x in listA]
    listB = [str(x) for x in listB]
    listC = [str(x) for x in listC]
    
    mat = np.array([listA, listB, listC,])
    mat = np.transpose(mat)
    print(mat)
    dataframe = pd.DataFrame(mat)
    dataframe.insert(3, 'classe', classification)
    print('dataframe')
    print(dataframe)
    dataframe.to_csv('standard_results.txt', index=False)
    f.close()
    
#retorna as colunas do arquivo
#recebe: titulo do arquivo
#retorna: uma matriz com as colunas
def get_columns(title):
    file = open(title)
    csvreader = csv.reader(file)

    header = next(csvreader)
    print(header)

    rows = []
    ia = []
    ib = []
    ic = []

    for row in csvreader:
        rows.append(row)

    for i in range(len(rows)):
        ia.append(rows[i][1])
        ib.append(rows[i][2])
        ic.append(rows[i][3])

    file.close()
    return [ia, ib, ic]

def get_columns_no_amostras(title):
    file = open(title)
    csvreader = csv.reader(file)

    header = next(csvreader)
    print(header)

    rows = []
    ia = []
    ib = []
    ic = []

    for row in csvreader:
        rows.append(row)

    for i in range(len(rows)):
        ia.append(rows[i][0])
        ib.append(rows[i][1])
        ic.append(rows[i][2])

    file.close()
    return [ia, ib, ic]

#normalizando valores
#argumentos: 1 vetor de valores
#retorno: vetor normalizado
def standard(list):
    list = [int(x) for x in list]
    mat = np.array([list])
    std = minmax_scale(list)
    std = [truncate(x,2) for x in std]
    return std


#transformada rapida de fourier
def fft_app(x):
    f = []
    f_len = []

    f = np.fft.rfft(x) / len(x)

    f = f[range(int(len(x) / 2))]

    tpCount = len(x)

    values = np.arange(int(tpCount / 2))

    timePeriod = tpCount / 100

    frequencies = values / timePeriod

    for i in range(len(f)):
        f_len.append(i)

    return f, f_len, frequencies


#trunca um número em 2 casas decimais
def truncate(number, decimals=0):
    # Retorna valor truncado com x casas
    if not isinstance(decimals, int):
        raise TypeError("A quantidade de casas deve ser um inteiro.")
    elif decimals < 0:
        raise ValueError("A quantidade de casas deve ser positiva.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def main():
    print('hello')
    if __name__ == "__main__":
        main()
    #file = open("DadosSimulacoes_09jun22a.csv")
    col = get_columns("DadosSimulacoes_09jun22a.csv")
    print(col);
    return 1;

def knn_comparison(data, k):
    x = data[['0', '1']].values
    y = data['classe'].astype(int).values
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y)
    plot_decision_regions(x,y,clf=knn)
    plt.xlabel('ia')
    plt.ylabel('ib')
    plt.title('K=' +str(k))
    plt.show()

#write_standards_currents(stdIa, stdIb, stdIc, 2)

col = get_columns_no_amostras("DadosSimulacoes_09jun225n.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

#write_standards_currents(stdIa, stdIb, stdIc, 3)

col = get_columns_no_amostras("DadosSimulacoes_09jun2210n.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

write_standards_currents(stdIa, stdIb, stdIc, 4)

col = get_columns_no_amostras("DadosSimulacoes_09jun2210ndesb.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

write_standards_currents(stdIa, stdIb, stdIc, 5)

col = get_columns_no_amostras("DadosSimulacoes_09jun2215n.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

write_standards_currents(stdIa, stdIb, stdIc, 6)

col = get_columns_no_amostras("DadosSimulacoes_09jun2215ndesb.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

write_standards_currents(stdIa, stdIb, stdIc, 7)

col = get_columns_no_amostras("DadosSimulacoes_09jun2220n.csv")
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])

write_standards_currents(stdIa, stdIb, stdIc, 8)

dataframe = pd.read_csv('draft2.txt')
x_data = dataframe.drop(['Classe', 'RPM'], axis =1)
y_data = dataframe['Classe']
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_MinMax = MinMaxScaler.fit_transform(x_data)
data = pd.DataFrame(X_data_MinMax, columns=['Ia', 'Ib', 'Ic'])


print(data.head())

X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y_data,test_size=0.15, random_state = 0)
X_train, val_inputs, y_train, val_outputs = model_selection.train_test_split(data, y_data, test_size=0.15, random_state = 0)

for i in range(1,1000):
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors= 1)
    knn_clf.fit(data,y_data)
    ypred=knn_clf.predict(data)
    scores =  accuracy_score(y_data, ypred)
    print(i)
    print(scores)

##################
