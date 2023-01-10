from pandas import read_csv, DataFrame
from pandas.plotting import register_matplotlib_converters
from dscharts import get_variable_types


def outliers_treatment(filename, file, dataset, index_col, na_values):
    register_matplotlib_converters()
    data = read_csv(filename, index_col=index_col, na_values=na_values)

    print_summary5(data)
    drop_outliers(data, dataset, file)
    replacing_outliers(data, dataset, file)
    truncating_outliers(data, dataset, file)


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


def drop_outliers(data, dataset, file):
    # numeric_vars = get_variable_types(data)['Numeric']
    numeric_vars = ['num_lab_procedures']
    # if [] == numeric_vars:
    #     raise ValueError('There are no numeric variables.')
    print('Original data:', data.shape)
    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
    output_location = 'data/classification/datasets_for_further_analysis/'+dataset+'/Outliers/'+file+'_drop_outliers.csv'
    df.to_csv(f'{output_location}', index=True)
    print('data after dropping outliers:', df.shape)


def replacing_outliers(data, dataset, file):
    # numeric_vars = get_variable_types(data)['Numeric']
    # if [] == numeric_vars:
    #     raise ValueError('There are no numeric variables.')
    numeric_vars = ['num_lab_procedures']

    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        median = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    print('Original data:', data.shape)
    # print('data after replacing outliers:', df.describe())
    output_location = 'data/classification/datasets_for_further_analysis/'+dataset+'/Outliers/'+file+'_replacing_outliers.csv'

    df.to_csv(f'{output_location}', index=True)
    print('data after replacing outliers:', df.shape)



def truncating_outliers(data, dataset, file):
    # numeric_vars = get_variable_types(data)['Numeric']
    # if [] == numeric_vars:
    #     raise ValueError('There are no numeric variables.')

    numeric_vars = ['num_lab_procedures']
    summary5 = data.describe(include='number')
    df = data.copy(deep=True)
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var)
        df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

    print('Original data:', data.shape)
    # print('data after truncating outliers:', df.describe())
    output_location = 'data/classification/datasets_for_further_analysis/'+dataset+'/Outliers/'+file+'_truncate_outliers.csv'

    #df.to_csv(f'{output_location}', index=True)
    print('data after truncating outliers:', df.shape)


def print_summary5(data):
    print(data.describe())


# outliers_treatment('data/classification/datasets_for_further_analysis/dataset1/mv_filled_mean_dataset1.csv', 'diabetic_data', 'dataset1', 'encounter_id', '')

outliers_treatment('data/classification/datasets_for_further_analysis/dataset1/MVI/mv_filled_mean_dataset1.csv', 'dataset1_filled_mean', 'dataset1', 'encounter_id', '')
outliers_treatment('data/classification/datasets_for_further_analysis/dataset1/MVI/diabetic_dropna_drop_columns_mv.csv', 'dataset1_dropna', 'dataset1', 'encounter_id', '')

# outliers_treatment('data/classification/diabetic_data.csv', 'diabetic_data', 'dataset1', 'encounter_id', '')
#outliers_treatment('data/classification/datasets_for_further_analysis/dataset2/dataset2_variable_encoding.csv', 'dataset2', 'dataset2', 'date', '')


