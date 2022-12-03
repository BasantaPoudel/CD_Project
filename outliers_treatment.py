from pandas import read_csv, DataFrame
from pandas.plotting import register_matplotlib_converters
from dscharts import get_variable_types


def outliers_treatment(filename, file, dataset, index_col, na_values):
    register_matplotlib_converters()
    if (dataset == "dataset2"):
        data = read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True, index_col=index_col, na_values=na_values)
    else:
        data = read_csv(filename, index_col=index_col, na_values=na_values)

    print_summary5(data)
    drop_outliers(data, file)
    #replacing_outliers(data, file)
    #truncating_outliers(data, file)


def determine_outlier_thresholds(summary5: DataFrame, var: str):
    OUTLIER_PARAM: int = 1.5 # define the number of stdev to use or the IQR scale (usually 1.5)
    OPTION = 'iqr'  # or 'stdev'
    if 'iqr' == OPTION:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:  # OPTION == 'stdev'
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold


def drop_outliers(data, file):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    print('Original data:', data.shape)
    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    df.to_csv(f'data/{file}_drop_outliers.csv', index=True)
    print('data after dropping outliers:', df.shape)


def replacing_outliers(data, file):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        median = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    print('Original data:', data.shape)
    # print('data after replacing outliers:', df.describe())
    df.to_csv(f'data/{file}_replacing_outliers.csv', index=True)
    print('data after replacing outliers:', df.shape)



def truncating_outliers(data, file):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

    print('Original data:', data.shape)
    # print('data after truncating outliers:', df.describe())
    df.to_csv(f'data/{file}_truncate_outliers.csv', index=True)
    print('data after truncating outliers:', df.shape)


def print_summary5(data):
    print(data.describe())


# outliers_treatment('data/classification/drought.csv', 'drought', 'dataset2', 'date', '')
outliers_treatment('data/classification/diabetic_data.csv', 'diabetic_data', 'dataset1', 'encounter_id', '?')
