from pandas import read_csv
from pandas import DataFrame
from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap
from dscharts import bar_chart, get_variable_types


def feature_engineering(filename, file, dataset, index_col):
    data = read_csv(filename,index_col=index_col)
    print('shape of original data:', data.shape)
    # data = drop_useless_vars(data, file)
    # data.to_csv('data/classification/datasets_for_further_analysis/'+dataset+'/'+dataset+'_drop_useless_vars_feature_engineering.csv')

    # for dataset1: there are no variables with correlation > 0.55
    # drop variables based on correlation - drop_redundant
    threshold = 0.95
    df = correlation_analysis(data, dataset, threshold)
    # print(df.shape)
    df.to_csv('data/classification/datasets_for_further_analysis/'+dataset+'/'+dataset+f'_drop_redundant_vars_{threshold}_feature_engineering.csv')

    # Drop on the basis of variance analysis
    # final_df, threshold_variance = variance_analysis(dataset, df2)
    # final_df.to_csv('data/classification/datasets_for_further_analysis/'+dataset+'/'+dataset+f'_{THRESHOLD}_{threshold_variance}_feature_engineering.csv')


def correlation_analysis(data, dataset, threshold):
    drop, corr_mtx = select_redundant(data.corr(), threshold)
    print(drop.keys())
    print(data.shape)
    try:
        plot_corrmatrix(corr_mtx, dataset, threshold)
        df = drop_redundant(data, drop)
    except:
        df = data
    return df


def select_redundant(corr_mtx, threshold: float) -> tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index


    return vars_2drop, corr_mtx


def plot_corrmatrix(corr_mtx, dataset, THRESHOLD):
    if corr_mtx.empty:
        raise ValueError('Matrix is empty.')

    figure(figsize=[10, 10])
    heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
    title('Filtered Correlation Analysis')
    savefig('images/feature_engineering/'+dataset+f'/filtered_correlation_analysis_{THRESHOLD}.png')
    # show()


def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df


def drop_useless_vars(data, dataset):  # drop all variables that have no numeric meaning for modeling

    if dataset == 'dataset1':
        useless_vars1 = ['payer_code', 'patient_nbr']
        for variable in useless_vars1:
            if variable in data:
                data.drop([variable], inplace=True, axis=1)
    else:  # data = dataset2
        useless_vars2 = ['fips']
        for variable in useless_vars2:
            if variable in data:
                data.drop([variable], inplace=True, axis=1)

    return data


def variance_analysis(dataset, df2):
    # drop variables based on variance analysis:
    threshold_variance = 0.1
    if 'Unnamed: 0' in df2:
        df2.drop(['Unnamed: 0'], inplace=True, axis=1)
    numeric = get_variable_types(df2)['Numeric']
    print('numeric variables are:', numeric)
    print(df2.var())
    vars2drop = select_low_variance(df2[numeric], threshold_variance, dataset)
    print('vars with low variance:', vars2drop)
    final_df = drop_low_var(df2, vars2drop)
    print(final_df.shape)
    return final_df, threshold_variance


def select_low_variance(data: DataFrame, threshold: float, dataset) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value <= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    if len(lst_variables) != 0:
        bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
        savefig('images/feature_engineering/'+dataset+'/filtered_variance_analysis.png')
        # show()
    return lst_variables


def drop_low_var(data, vars2drop):
    for var in vars2drop:
        if var in data:
            data.drop([var], inplace=True, axis=1)
    return data



# still need to select the right dataset for feature selection, SCALING dataaset should be used!:

# feature_engineering('data/classification/datasets_for_further_analysis/dataset1/dataset1_drop_outliers.csv',
#                                        'dataset1', 'dataset1', 'encounter_id')

# feature_engineering('forecasting/data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore.csv',
#                                         'dataset2', 'dataset2', 'date')
