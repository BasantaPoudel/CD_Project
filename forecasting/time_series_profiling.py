from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from numpy import ones
from pandas import Series
import pandas

#dimensionality


def main(data, dataset):
    gran_hourly(data, dataset)
    gran_daily(data, dataset)
    gran_weekly(data, dataset)
    gran_monthly(data, dataset)
    gran_quarterly(data, dataset)
    stationarity_study_1(data, dataset)
    pass

# #granularity
def gran_daily(data, dataset):
    day_df = data.copy().groupby(data.index.date).mean()
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_series(day_df, title='Daily granularity '+dataset, x_label='timestamp', y_label='consumption')
    xticks(rotation=45)
    savefig('images/time_series/data_profiling/'+dataset+'/daily granularity.png')
    # show()

def gran_hourly(data, dataset):
    index = data.index.to_period('h')
    hour_df = data.copy().groupby(index).mean()
    hour_df['timestamp'] = index.drop_duplicates().to_timestamp()
    hour_df.set_index('timestamp', drop=True, inplace=True)
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_series(hour_df, title='Hourly granularity '+dataset, x_label='timestamp', y_label='consumption')
    xticks(rotation=45)
    savefig('images/time_series/data_profiling/' + dataset + '/Hourly granularity.png')
def gran_weekly(data, dataset):
    index = data.index.to_period('W')
    week_df = data.copy().groupby(index).mean()
    week_df['timestamp'] = index.drop_duplicates().to_timestamp()
    week_df.set_index('timestamp', drop=True, inplace=True)
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_series(week_df, title='Weekly granularity '+dataset, x_label='timestamp', y_label='consumption')
    xticks(rotation=45)
    savefig('images/time_series/data_profiling/' + dataset + '/weekly granularity.png')
    # show()

def gran_monthly(data, dataset):
    index = data.index.to_period('M')
    month_df = data.copy().groupby(index).mean()
    month_df['timestamp'] = index.drop_duplicates().to_timestamp()
    month_df.set_index('timestamp', drop=True, inplace=True)
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_series(month_df, title='Monthly granularity '+dataset, x_label='timestamp', y_label='consumption')
    savefig('images/time_series/data_profiling/' + dataset + '/monthly granularity.png')
    # show()

def gran_quarterly(data, dataset):
    index = data.index.to_period('Q')
    quarter_df = data.copy().groupby(index).mean()
    quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
    quarter_df.set_index('timestamp', drop=True, inplace=True)
    figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_series(quarter_df, title='Quarterly granularity '+dataset, x_label='timestamp', y_label='consumption')
    savefig('images/time_series/data_profiling/' + dataset + '/quarterly granularity.png')
    # show()


def stationarity_study_1(data, dataset):

    for var in data:

        dt_series = Series(data[var])

        mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
        series = {var: dt_series, 'mean': mean_line}
        figure(figsize=(3 * HEIGHT, HEIGHT))
        plot_series(series, x_label='timestamp', y_label='value', title='Stationary study', show_std=True)
        savefig('images/time_series/data_profiling/' + dataset + '/stationarity_study_1'+var+'.png')
        # show()





# data = read_csv('data/forecasting/glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, dayfirst=True, infer_datetime_format=True)
# onlytarget_1 = data.drop(['Insulin'], axis=1)
#
# main(onlytarget_1, 'dataset1')

data = read_csv('data/forecasting/drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, dayfirst=True, infer_datetime_format=True)
onlytarget_2 = pandas.DataFrame(data['QV2M'])
main(onlytarget_2, 'dataset2')


print('nr of records:', data.shape[0])
