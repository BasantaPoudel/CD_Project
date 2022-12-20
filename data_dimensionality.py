import pandas as pd
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from dscharts import get_variable_types, bar_chart


def data_dimensionality(filename, dataset, index_col, na_values):
    if (dataset == "dataset2"):
        df = pd.read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True)
    else:
        df = pd.read_csv(filename)
    x = df.describe()
    print(df.dtypes)
    records_variables(df, dataset)
    variable_distribution(df, dataset)


def records_variables(df, dataset):
    '''_________________________Data dimensionality______________________________'''

    '''records vs variables'''
    figure(figsize=(4,2))
    values = {'nr records': df.shape[0], 'nr variables': df.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    image_location = 'images/data_dimensionality/' + dataset
    savefig(image_location+'/records_variables.png')
    # show()


def variable_distribution(df, dataset):
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

    image_location = 'images/data_dimensionality/' + dataset
    savefig(image_location+'/variable_types.png')
    #show()


# data_dimensionality('data/classification/diabetic_data.csv', 'dataset1', 'encounter_id', '?')
# data_dimensionality('data/classification/drought.csv', 'dataset2', 'date', '')
data_dimensionality('data/classification/data_for_DT_RF/dataset2_minmax_test.csv', 'dataset2_minmax_test', 'date', '')
