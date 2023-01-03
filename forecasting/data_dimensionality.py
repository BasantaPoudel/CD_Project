from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
import sys

def data_dimensionality(filename, dataset, index_col, na_values):
    data = read_csv(filename, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

    original_stdout = sys.stdout
    file_location = 'images/data_dimensionality/' + dataset
    with open(file_location+'/data_dimensionality_'+dataset+'.txt', 'w') as f:
        sys.stdout = f
        print("Nr. Records = ", data.shape[0])
        print("First timestamp", data.index[0])
        print("Last timestamp", data.index[-1])
        sys.stdout = original_stdout # Reset the standard output

    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(data, x_label='timestamp', y_label='consumption', title='ASHRAE')
    xticks(rotation = 45)

    image_location = 'images/data_dimensionality/' + dataset
    savefig(image_location+'/records_variables.png')
    show()


data_dimensionality('data/forecasting/glucose.csv', 'dataset1', 'Date', '')
data_dimensionality('data/forecasting/drought.forecasting_dataset.csv', 'dataset2', 'date', '')
