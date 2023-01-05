from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT

def data_transofrmation_original(data, dataset, target):    
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(data[target], x_label='timestamp', y_label='QV2M', title='Drought original')    
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/original.png')
    #show()

def data_transofrmation_original_multivariant(data, index_col,  dataset, target):    
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(data[target], x_label=index_col, y_label='QV2M', title=target)
    plot_series(data['PRECTOT'])
    plot_series(data['PS'])
    plot_series(data['T2M'])
    plot_series(data['T2MDEW'])
    plot_series(data['T2MWET'])
    plot_series(data['TS'])
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/original_multivariant.png')
    #show()

def smothing_win_size_10(data, dataset, target):
    WIN_SIZE = 10
    rolling = data.rolling(window=WIN_SIZE)
    smooth_df = rolling.mean()
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df[target], title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/smothing_win_size_10.png')
    #show()

def smothing_win_size_100(data, dataset, target):
    WIN_SIZE = 100
    rolling = data.rolling(window=WIN_SIZE)
    smooth_df = rolling.mean()
    figure(figsize=(3*HEIGHT, HEIGHT/2))
    plot_series(smooth_df[target], title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/smothing_win_size_100.png')
    #show()

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

def aggregate_hourly(data, dataset, target):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'h')
    datasetAggregateHourly = agg_df[target]
    datasetAggregateHourly.to_csv('data/forecasting/aggregation/'+dataset+'/aggregate_hourly.csv', index=True)
    plot_series(agg_df[target], title='Hourly drought', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_hourly.png')
    #show()

def aggregate_daily(data, dataset, target):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'D')
    datasetAggregateDaily = agg_df[target]
    datasetAggregateDaily.to_csv('data/forecasting/aggregation/'+dataset+'/aggregate_daily.csv', index=True)
    plot_series(agg_df[target], title='Daily drought', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_daily.png')
    #show()

def aggregate_weekly(data, dataset, target):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'W')
    datasetAggregateWeekly = agg_df[target]
    datasetAggregateWeekly.to_csv('data/forecasting/aggregation/'+dataset+'/aggregate_weekly.csv', index=True)
    plot_series(agg_df[target], title='Weekly drought', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_weekly.png')
    #show()

def aggregate_monthly(data, dataset, target):
    figure(figsize=(3*HEIGHT, HEIGHT))
    agg_df = aggregate_by(data, 'timestamp', 'M')
    datasetAggregateMonthly = agg_df[target]
    datasetAggregateMonthly.to_csv('data/forecasting/aggregation/'+dataset+'/aggregate_monthly.csv', index=True)
    plot_series(agg_df[target], title='Monthly drought', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/aggregation_monthly.png')
    #show()

def differentiation(data, dataset):
    diff_df = data.diff()
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(diff_df[target], title='Differentiation', x_label='timestamp', y_label='QV2M')
    xticks(rotation = 45)
    image_location = 'images/data_transformation/' + dataset
    savefig(image_location+'/differentiation.png')
    #show()




index_col = 'date'
dataset = 'dataset2'
target = 'QV2M'
data = read_csv('data/forecasting/drought.forecasting_dataset.csv',  index_col=index_col, sep=',', decimal='.', parse_dates=True,dayfirst=True,  infer_datetime_format=True)



data_transofrmation_original(data, dataset, target)
data_transofrmation_original_multivariant(data, index_col,  dataset, target)
smothing_win_size_10(data, dataset, target)
smothing_win_size_100(data, dataset, target)
aggregate_hourly(data, dataset, target)
aggregate_daily(data, dataset, target)
aggregate_weekly(data, dataset, target)
aggregate_monthly(data, dataset, target)
differentiation(data, dataset)



    