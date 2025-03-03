from pandas import read_csv
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from dscharts import get_variable_types
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots, show
from matplotlib.pyplot import figure, savefig
import numpy as np

def scaling(data, dataset):
    register_matplotlib_converters()

    data = pd.read_csv(data, index_col='encounter_id')

    array_class = data.pop('readmitted')    

    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp, df_sb,  df_bool, array_class], axis=1)    
    norm_data_zscore.to_csv('data/classification/datasets_for_further_analysis/'+dataset+'/Scaling/'+dataset+'_dropna_replacing_outliers_scaled_zscore.csv', index=True)
    print(norm_data_zscore.describe())


    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_minmax = concat([tmp, df_sb,  df_bool, array_class], axis=1)
    norm_data_minmax.to_csv('data/classification/datasets_for_further_analysis/'+dataset+'/Scaling/'+dataset+'_dropna_replacing_outliers_scaled_minmax.csv', index=True)
    print(norm_data_minmax.describe())


    fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
    axs[0, 0].set_title('Original data')
    data.boxplot(ax=axs[0, 0])
    axs[0, 1].set_title('Z-score normalization')
    norm_data_zscore.boxplot(ax=axs[0, 1])
    axs[0, 2].set_title('MinMax normalization')
    norm_data_minmax.boxplot(ax=axs[0, 2])    
    image_location = 'images/data_scaling/' + dataset
    savefig(image_location+'/scaling_comparison_dataset1.png')
    show()

scaling('data/classification/datasets_for_further_analysis/dataset1/Scaling/dataset1_dropna_replacing_outliers.csv', "dataset1")
