from pandas import read_csv, DataFrame, concat, Series
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import figure, savefig, show
from dscharts import bar_chart, multiple_bar_chart


def data_balancing(filename, file, dataset, index_col, class_col, na_values, ):
    # register_matplotlib_converters()
    if (dataset == "dataset2"):
        original = read_csv(filename, dayfirst=True, parse_dates=['date'], infer_datetime_format=True, index_col=index_col, na_values=na_values)
    else:
        original = read_csv(filename, index_col=index_col, na_values=na_values)

    target_count = original[class_col].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    savefig(f'images/{file}_balance.png')
    show()

    # undersampling(file, class_col, original, positive_class, negative_class, values)
    #oversampling(file, class_col, original, positive_class, negative_class, values)
    smote(file, class_col, original, positive_class, negative_class, values)


def undersampling(file, class_col, original, positive_class, negative_class, values):
    df_positives = original[original[class_col] == positive_class]
    df_negatives = original[original[class_col] == negative_class]

    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    df_under.to_csv(f'data/{file}_under.csv', index=False)
    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')


def oversampling(file, class_col, original, positive_class, negative_class, values):
    df_positives = original[original[class_col] == positive_class]
    df_negatives = original[original[class_col] == negative_class]

    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(f'data/{file}_over.csv', index=False)
    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('Minority class=', positive_class, ':', len(df_pos_sample))
    print('Majority class=', negative_class, ':', len(df_negatives))
    print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')


#oversampling hybrid technique
def smote(file, class_col, original, positive_class, negative_class, values):
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = original.pop(class_col).values
    X = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [class_col]
    df_smote.to_csv(f'data/{file}_smote.csv', index=False)

    smote_target_count = Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

    figure()
    multiple_bar_chart([positive_class, negative_class], values, title='Target', xlabel='frequency', ylabel='Class balance')
    show()

# data_balancing('data/classification/diabetic_data.csv', 'diabetic_data', 'dataset1', "encounter_id", 'readmitted', "?")
data_balancing('data/classification/drought.csv', 'drought', 'dataset2', 'date', 'class', '')

