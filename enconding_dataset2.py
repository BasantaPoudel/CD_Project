from pandas import read_csv
import pandas as pd
from dscharts import get_variable_types
from numpy import number
import datetime

file = ""

def data_encoding(filename):
    file = filename
    data = read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True)
    variable_encoding(data)

def variable_encoding(data):
    #Date enconded
    data_datetime_encoding = {}
    
    for n in range(len(data['date'])):
        # print(data['date'][n])
        data_datetime_encoding[n] = 10000*data['date'][n].year + 100*data['date'][n].month + data['date'][n].day
        # print(data_datetime_encoding[n])
        
    data['date'] =  data_datetime_encoding.values()

    data.to_csv('data/classification/datasets_for_further_analysis/dataset2/dataset2_variable_encoding.csv', index=False)


data_encoding('data/classification/drought.csv')
