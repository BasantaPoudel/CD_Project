import pandas as pd
import numpy as np
from numpy import nan
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from dscharts import bar_chart
from collections import defaultdict

# df = pd.read_csv('data/classification/diabetic_data.csv')

df = pd.read_csv('data/classification/drought.csv')

mvv = defaultdict()

'''if we count ? as a missing value:
    ? is replaced with nan for further processing'''

for j in df:
    s = 0
    for i in range(len(df)):
        value = df.loc[i,j]
        if pd.isna(value):
            s+=1
        elif value =='?':
            # df.loc[i,j] = nan
            s +=1
    mvv[j] = s
            
bar_chart(list(mvv.keys()), list(mvv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/data_dimensionality/dataset2/missing_values.png')