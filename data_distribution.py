from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots
from dscharts import get_variable_types, choose_grid, HEIGHT


def data_distribution():
    register_matplotlib_converters()
    filename = 'data/classification/drought.csv'
    data = read_csv(filename, index_col='date', na_values='')
    # print(data)
    data_without_class = data.drop("class", axis='columns')
    # print(data_without_class)
    print_summary5(data_without_class)
    box_plot(data_without_class)


def print_summary5(data):
    summary5 = data.describe()
    print(summary5)


# Numeric Variables distribution
# box-plot
def box_plot(data):
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
    savefig('images/data_distribution/dataset2/single_boxplots.png')
    # show()


data_distribution()

