from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes
from dscharts import get_variable_types, choose_grid, multiple_bar_chart, HEIGHT, multiple_line_chart, bar_chart
from seaborn import distplot
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm


def data_distribution(filename, dataset, index_col, na_values):
    register_matplotlib_converters()
    data = read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True, index_col=index_col, na_values=na_values)
    # print(data)
    if (dataset == "dataset2"):
        data_without_class = data.drop("class", axis='columns')
    else:
        data_without_class = data
    # print(data_without_class)
    print_summary5(data_without_class)
    #single_box_plot(data_without_class, dataset)
    #global_boxplot(data_without_class, dataset)
    #outliers_plot(data_without_class, dataset)
    #hist_plot(data_without_class, dataset)
    #best_fit_distribution(data_without_class, dataset)
    #hist_symbolic(data_without_class, dataset)


def print_summary5(data):
    summary5 = data.describe()
    print(summary5)


def global_boxplot(data, dataset):
    data.boxplot(rot=45)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/global_boxplot.png')
    #show()


# Numeric Variables distribution
# box-plot
def single_box_plot(data, dataset):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/single_boxplots.png')
    # show()


def outliers_plot(data, dataset):
    NR_STDEV: int = 2
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary5 = data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
            data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            data[data[var] > summary5[var]['mean'] + std].count()[var] +
            data[data[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/outliers.png')
    #show()


def hist_plot(data, dataset):
    numeric_vars = get_variable_types(data)['Numeric']    
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/single_histograms_numeric.png')
    #show()


def hist_trend(data, dataset):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histograms_trend_numeric.png')
    #show()


def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')


def best_fit_distribution(data, dataset):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histogram_numeric_distribution.png')
    #show()


def hist_symbolic(data, dataset):
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    image_location = 'images/data_distribution/' + dataset
    savefig(image_location+'/histograms_symbolic.png')
    #show()


def class_distribution(data, dataset):
    # TODO
    return True


data_distribution('data/classification/drought.csv', 'dataset2', 'date', '')
# data_distribution('data/classification/diabetic_data.csv', 'dataset1', "encounter_id", "?")

