from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from dscharts import get_variable_types
from numpy import nan
import pandas as pd
#import dataframe_image as dfi

data = pd.read_csv('data/classification/datasets_for_further_analysis/dataset1/diabetic_data_variable_enconding.csv',
                   index_col = 'encounter_id')

datareadmitted = data["readmitted"]

strategy = 1

if strategy == 1:
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']
    
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', fill_value=0, missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', fill_value=False, missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)
    
    df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    df.index = data.index
    #df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df["readmitted"] = datareadmitted
    df.to_csv('data/classification/datasets_for_further_analysis/dataset1/mv_filled_mean_dataset1.csv', index=True)
    describe = df.describe(include='all')
    #dfi.export(describe, 'images/missing_values_imputation/dataset1/describe_mean_fill.png',max_cols=(-1))
    #dfi.export(describe, 'images/missing_values_imputation/dataset1/describe_constant_fill.png',max_cols=(-1))

else:
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']
    
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)
    
    df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    df.index = data.index
    
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.to_csv('data/classification/datasets_for_further_analysis/dataset1/mv_filled_most_frequent_dataset1.csv', index=True)
    describe = df.describe(include='all')
    #dfi.export(describe, 'images/missing_values_imputation//dataset1/describe_most_frequent_fill.png', max_cols=(-1))
    
    
    
