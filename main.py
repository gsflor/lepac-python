import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
import math
import pandas as pd

def write_standards_currents(list):
    f = open('standard_results.txt', 'w')
    list = [str(x) for x in list]
    dataframe = pd.DataFrame(list)
    dataframe.to_csv('teste.csv')
    f.close()
    

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


def standard(list):
    list = [int(x) for x in list]
    std = preprocessing.scale(list)
    std = [truncate(x,2) for x in std]
    return std


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
stdIa = standard(col[0])
stdIb = standard(col[1])
stdIc = standard(col[2])
print(stdIa)
print(stdIb)
print(stdIc)
write_standards_currents(stdIa)
