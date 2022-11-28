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


'''_________________________Data dimensionality______________________________'''

'''records vs variables'''
figure(figsize=(4,2))
values = {'nr records': df.shape[0], 'nr variables': df.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/data_dimensionality/dataset1/records_variables.png')
show()


cat_vars = df.select_dtypes(include='object')
df[cat_vars.columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))


''' number of variables per category'''
variable_types = get_variable_types(df)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('images/data_dimensionality/dataset1/variable_types.png')
show()