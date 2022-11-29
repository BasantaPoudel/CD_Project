import pandas as pd
from matplotlib.pyplot import savefig, show, subplots
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from dscharts import get_variable_types, HEIGHT


def data_granularity(filename, dataset, index_col, na_values):
    '''DATA GRANULARITY'''
    if (dataset == "dataset2"):
        df = pd.read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True)
    else:
        df = pd.read_csv(filename)
    print(df.dtypes)
    variables = get_variable_types(df)['Date']
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
    image_location = 'images/data_granularity/' + dataset
    savefig(image_location+'/granularity_study_date.png')
    show()


# data_granularity('data/classification/diabetic_data.csv', 'dataset1', 'encounter_id', '?')
data_granularity('data/classification/drought.csv', 'dataset2', 'date', '')
