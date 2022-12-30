import numpy as np
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show, figure, title
from dscharts import get_variable_types, HEIGHT
from seaborn import heatmap
import matplotlib.pyplot as plt


def data_sparcity(filename, dataset, index_col, na_values, clas):
    register_matplotlib_converters()
    if (dataset == "dataset2"):         
        data = read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True, index_col=index_col)
    else:
        data = read_csv(filename, index_col=index_col, parse_dates=True, infer_datetime_format=True)
    # scatter_plot_sparcity_Numeric(data, dataset)
    # scatter_plot_sparcity_symbolic(data, dataset)

    # scatter_plot_sparcity_numeric_class(data, dataset, clas)
    # scatter_plot_sparcity_symbolic_class(data, dataset, clas)

    # correlation_analysis(data, dataset)
    correlation_analysis_class(data, dataset, clas)


def scatter_plot_sparcity_numeric(data, dataset):
    numeric_vars = get_variable_types(data)['Numeric']
    print(len(numeric_vars))
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = 15, 15
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(16):
        var1 = numeric_vars[i]
        for j in range(i+1, 16):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    image_location = 'images/data_sparcity/' + dataset
    savefig(image_location+'/sparsity_study_numeric.png')
    #show()

def scatter_plot_sparcity_numeric_class(data, dataset, clas):
    numeric_vars = get_variable_types(data)['Numeric']
    print(len(numeric_vars))
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = len(numeric_vars)-1,len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = clas
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    image_location = 'images/data_sparcity/' + dataset
    savefig(image_location+'/sparsity_study_numeric_class.png')
    #show()

def scatter_plot_sparcity_symbolic(data, dataset):
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    image_location = 'images/data_sparcity/' + dataset
    savefig(image_location+'/sparsity_study_symbolic.png')
    # show()

def scatter_plot_sparcity_symbolic_class(data, dataset, clas):
    symbolic_vars = get_variable_types(data)['Symbolic']
    print(len(symbolic_vars))
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = clas
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    image_location = 'images/data_sparcity/' + dataset
    savefig(image_location+'/sparsity_study_symbolic_class.png')
    show()



def correlation_analysis(data, dataset):
    corr_mtx = abs(data.corr())
    print(corr_mtx)
    if dataset == 'dataset1':
        fig = figure(figsize=[12, 12])
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    else:
        fig = figure(figsize=[51, 51])
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues',
                annot_kws={'fontsize':8}, square=True)
    title('Correlation analysis')
    image_location = 'images/data_sparcity/' + dataset
    savefig(image_location+'/correlation_analysis.png')
    # show()

def correlation_analysis_class(data, dataset, clas):
    corr = []
    vars = []

    numeric_vars = get_variable_types(data)['Numeric']
    numeric_vars = numeric_vars[:-1]
    print(numeric_vars)
    for var in numeric_vars:

        corr.append(np.corrcoef(data[var], data[clas])[0][1])
        vars.append(var)


    # print(, corr)
    plt.bar(vars, corr)
    plt.xticks(rotation=90)
    plt.title('corrlation of numeric variables against the class')

    image_location = 'images/data_sparcity/' + dataset
    plt.savefig(image_location + '/correlation_analysis_CLASS.png')
    plt.show()




def class_sparcity(data, dataset):
    # TODO
    return True

# data_sparcity('data/classification/diabetic_data.csv', 'dataset1', "encounter_id", "?", 'readmitted')
data_sparcity('data/classification/datasets_for_further_analysis/dataset1/diabetic_data_variable_enconding.csv',
              'dataset1', "encounter_id", "?", 'readmitted')
# data_sparcity('data/classification/drought.csv', 'dataset2', "date", '', 'class')
