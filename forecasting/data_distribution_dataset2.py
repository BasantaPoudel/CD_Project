from matplotlib.pyplot import subplots
from pandas import read_csv
from matplotlib.pyplot import figure, xticks, savefig
from ts_functions import plot_series, HEIGHT
import sys


def data_distribution(filename, dataset, index_col, na_values, class_column):
    data = read_csv(filename, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

    # Weekly
    index = data.index.to_period('W')
    weekly_df = data.copy().groupby(index).sum()
    weekly_df[index_col] = index.drop_duplicates().to_timestamp()
    weekly_df.set_index(index_col, drop=True, inplace=True)

    # Monthly
    index = data.index.to_period('M')
    monthly_df = data.copy().groupby(index).sum()
    monthly_df[index_col] = index.drop_duplicates().to_timestamp()
    monthly_df.set_index(index_col, drop=True, inplace=True)

    # Daily - data
    _, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
    axs[0].grid(False)
    axs[0].set_axis_off()
    axs[0].set_title('DAILY', fontweight="bold")
    axs[0].text(0, 0, str(data.describe()))

    axs[1].grid(False)
    axs[1].set_axis_off()
    axs[1].set_title('WEEKLY', fontweight="bold")
    axs[1].text(0, 0, str(weekly_df.describe()))

    axs[2].grid(False)
    axs[2].set_axis_off()
    axs[2].set_title('MONTHLY', fontweight="bold")
    axs[2].text(0, 0, str(monthly_df.describe()))

    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/5-NumberSummary_'+dataset+'.png')
    # show()

    box_plot_most_atomic(data, dataset)

    variables_distribution_plot(data, dataset)


def box_plot_most_atomic(data, dataset):
    _, axs = subplots(1, figsize=(2 * HEIGHT, HEIGHT))
    axs.set_title('DAILY', fontweight="bold")
    data.boxplot(ax=axs)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location + '/boxplot_' + dataset + '.png')
    # show()


def variables_distribution_plot(data, dataset):
    bins = (10, 25, 50)
    _, axs = subplots(1, len(bins), figsize=(len(bins) * HEIGHT, HEIGHT))

    # Daily data histogram
    for j in range(len(bins)):
        axs[j].set_title('Histogram for daily records %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data.values, bins=bins[j])
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_daily_'+dataset+'.png')

    # Weekly data histogram
    index = data.index.to_period('W')
    weekly_df = data.copy().groupby(index).sum()
    for j in range(len(bins)):
        axs[j].set_title('Histogram for weekly records %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(weekly_df.values, bins=bins[j])
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_weekly_'+dataset+'.png')

    # Monthly data histogram
    index = data.index.to_period('M')
    weekly_df = data.copy().groupby(index).sum()
    for j in range(len(bins)):
        axs[j].set_title('Histogram for monthly records %d bins' % bins[j])
        axs[j].set_xlabel('consumption')
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data.values, bins=bins[j])

    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_monthly_'+dataset+'.png')
    # show()


data_distribution('data/forecasting/drought.forecasting_dataset.csv', 'dataset2', "date", "", '')
