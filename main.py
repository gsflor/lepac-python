import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import minmax_scale
import math
import pandas as pd

#escreve as correntes normalizadas no arquivo
def write_standards_currents(listA, listB, listC, classification):
    f = open('standard_results.txt', 'w')
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
    dataframe.to_csv('teste.csv')
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


col = get_columns("DadosSimulacoes_09jun22a.csv")
stdIa = standard(col[0]) #trocar por minmax
stdIb = standard(col[1])
stdIc = standard(col[2])
print(stdIa)
print(stdIb)
print(stdIc)
write_standards_currents(stdIa, stdIb, stdIc, 'semCarga')
