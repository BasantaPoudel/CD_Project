import pandas as pd
import numpy as np
from numpy import nan
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from dscharts import bar_chart
from collections import defaultdict

df = pd.read_csv('forecasting/data/classification/datasets_for_further_analysis/dataset1/diabetic_fill_columns_mv.csv')

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
            df.loc[i,j] = nan
            s +=1
    mvv[j] = s
            
bar_chart(list(mvv.keys()), list(mvv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)


'''figure already saved in previous work lab1'''
# savefig('C:/Users/20202199/stack2/International Semester/Data Science/Project/figures_lab2/missing_values.png')

# defines the number of records to discard entire columns
threshold = df.shape[0] * 0.90

missings = [c for c in mvv.keys() if mvv[c]>threshold]
data = df.drop(columns=missings, inplace=False)
# data.to_csv('disbetic_data_drop_columns_mv.csv', index=True)
print('Dropped variables', missings)

# defines the number of variables to discard entire records
threshold = df.shape[1] * 0.50

data = data.dropna(thresh=threshold, inplace=False)
# data.to_csv('drop_records_mv.csv', index=True)
print(df.shape)

'''drop payer and medical speciality?? They have many missing values'''
# data.drop(['payer_code', 'medical_specialty'],axis=1,inplace=True)

data.to_csv('data/classification/datasets_for_further_analysis/dataset1/drop_recs_cols_dataset1.csv')




























