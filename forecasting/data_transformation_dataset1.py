from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT

def data_transofrmation_original(data, dataset, target):    
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(data[target], x_label='timestamp', y_label='glucose', title='Glucose original')    
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/original.png')
    show()

def data_transofrmation_original_multivariant(data, index_col,  dataset, target):    
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(data[target], x_label=index_col, y_label='glucose', title=target)
    plot_series(data['Insulin'])
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/original_multivariant.png')
    show()

def smothing_win_size_10(data, dataset, target):
    WIN_SIZE = 10
    rolling = data.rolling(window=WIN_SIZE)
    smooth_df = rolling.mean()
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/smothing_win_size_10.png')
    show()

def smothing_win_size_100(data, dataset, target):
    WIN_SIZE = 100
    rolling = data.rolling(window=WIN_SIZE)
    smooth_df = rolling.mean()
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/smothing_win_size_100.png')
    show()

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

def aggregate_hourly(data, dataset):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'h')
    plot_series(agg_df, title='Hourly glucose', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_hourly.png')
    show()

def aggregate_daily(data, dataset):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'D')
    plot_series(agg_df, title='Daily glucose', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_daily.png')
    show()

def aggregate_weekly(data, dataset):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'W')
    plot_series(agg_df, title='Weekly glucose', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_weekly.png')
    show()

def aggregate_monthly(data, dataset):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'M')
    plot_series(agg_df, title='Monthly glucose', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_monthly.png')
    show()

def differentiation(data, dataset):
    diff_df = data.diff()
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(diff_df, title='Differentiation', x_label='timestamp', y_label='glucose')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/differentiation.png')
    show()

index_col = 'Date'
dataset = 'dataset1'
target = 'Glucose'
data = read_csv('data/forecasting/glucose.csv',  index_col=index_col, sep=',', decimal='.', parse_dates=True,dayfirst=True,  infer_datetime_format=True)



data_transofrmation_original(data, dataset, target)
data_transofrmation_original_multivariant(data, index_col,  dataset, target)
smothing_win_size_10(data, dataset, target)
smothing_win_size_100(data, dataset, target)
aggregate_hourly(data, dataset)
aggregate_daily(data, dataset)
aggregate_weekly(data, dataset)
aggregate_monthly(data, dataset)
differentiation(data, dataset)



    