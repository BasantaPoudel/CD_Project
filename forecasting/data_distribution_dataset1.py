from matplotlib.pyplot import subplots
from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
import sys


def data_distribution(filename, dataset, index_col, target, class_column):
    data = read_csv(filename, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

    index = data.index.to_period('D')
    period_df = data.copy().groupby(index).mean().round(2)
    period_df[index_col] = index.drop_duplicates().to_timestamp()
    period_df.set_index(index_col, drop=True, inplace=True)

    _, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
    axs[0].grid(False)
    axs[0].set_axis_off()
    axs[0].set_title('HOURLY', fontweight="bold")
    axs[0].text(0, 0, str(data.describe().round(2)))
    axs[1].grid(False)
    axs[1].set_axis_off()
    axs[1].set_title('DAILY', fontweight="bold")
    axs[1].text(0, 0, str(period_df.describe()))
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/5-NumberSummary_'+dataset+'.png')
    # show()

    _, axs = subplots(1, figsize=(2*HEIGHT, HEIGHT))
    axs.set_title('HOURLY', fontweight="bold")

    box_plot_most_atomic(data, dataset, index_col, target)
    variables_distribution_plot(data, dataset, index_col)

def box_plot_most_atomic(data, dataset, index_col, target):
    _, axs = subplots(1, figsize=(2 * HEIGHT, HEIGHT))
    axs.set_title('DAILY', fontweight="bold")
    data = data.drop("Insulin", axis="columns")
    data.boxplot(ax=axs)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location + '/boxplot_' + dataset + '.png')
    # show()


def variables_distribution_plot(data, dataset, index_col):
    data = data.drop("Insulin", axis="columns")

    bins = (10, 25, 50)
    _, axs = subplots(1, len(bins), figsize=(len(bins) * HEIGHT, HEIGHT))

    # hourly data histogram
    for j in range(len(bins)):
        axs[j].set_title('Histogram for hourly values %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data.values, bins=bins[j])
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_hourly_'+dataset+'.png')    

    # Daily data histogram
    index = data.index.to_period('D')
    period_df = data.copy().groupby(index).mean().round(2)
    period_df[index_col] = index.drop_duplicates().to_timestamp()
    period_df.set_index(index_col, drop=True, inplace=True)

    for j in range(len(bins)):
        axs[j].set_title('Histogram for daily values %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(period_df.values, bins=bins[j])
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_daily_'+dataset+'.png')

    # Weekly data histogram
    index = data.index.to_period('W')
    period_df = data.copy().groupby(index).mean().round(2)
    period_df[index_col] = index.drop_duplicates().to_timestamp()
    period_df.set_index(index_col, drop=True, inplace=True)
    for j in range(len(bins)):
        axs[j].set_title('Histogram for weekly values %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(period_df.values, bins=bins[j])
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_weekly_'+dataset+'.png')

    # show()


data_distribution('data/forecasting/glucose.csv', 'dataset1', "Date", "glucose", '')
