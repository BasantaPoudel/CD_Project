import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, subplots
from ds_charts import bar_chart, get_variable_types
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from ds_charts import get_variable_types, choose_grid, HEIGHT, multiple_bar_chart

df = pd.read_csv('data/classification/diabetic_data.csv')
x = df.describe()

print(df.dtypes)




'''DATA GRANULARITY'''

variables = get_variable_types(df)['Numeric']
rows = len(variables)
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(df[variables[i]].values, bins=bins[j])
savefig('images/data_granularity/dataset1/granularity_study.png')
show()