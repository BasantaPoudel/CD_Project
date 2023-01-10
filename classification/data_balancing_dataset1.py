from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show
from dscharts import bar_chart, multiple_bar_chart
from pandas import concat, DataFrame
from pandas import Series
from imblearn.over_sampling import SMOTE

filename = 'data/classification/datasets_for_further_analysis/dataset1/Balancing/dataset1_0.2_0.1_feature_engineering_diabetes_train.csv'

file = "balanced"
scaling = "dataset1_0.2_0.1"

original = read_csv(filename, sep=',', decimal='.')
class_var = 'readmitted'
target_count = original[class_var].value_counts()
positive_class = 1 #'<30'#target_count.idxmin()
negative_class = 0 #'NO'#target_count.idxmax()
intermediate_class = 2 #'>30'

print('Minority class=', positive_class, ':', target_count[positive_class])
print('Majority class=', negative_class, ':', target_count[negative_class])
print('Intermediate class=', intermediate_class, ':', target_count[intermediate_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')

values = {'Original': [target_count[positive_class], target_count[negative_class], target_count[intermediate_class]] }

figure()
bar_chart(target_count.index, target_count.values, title='Class balance')
savefig(f'images/data_balancing/dataset1/diabetes'+scaling+'_resumen_unbalance.png')
show()

#Set positives, negatives, intermediates from original diabetes dataset
df_positives = original[original[class_var] == positive_class]
df_negatives = original[original[class_var] == negative_class]
df_intermediate = original[original[class_var] == intermediate_class]

print(len(df_positives))
print(len(df_negatives))
print(len(df_intermediate))


######### UNDERSAMPLING
df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
df_inter_sample = DataFrame(df_intermediate.sample(len(df_positives)))
df_under = concat([df_positives, df_neg_sample, df_inter_sample], axis=0)
df_under.to_csv(f'data/classification/datasets_for_further_analysis/dataset1/Balancing/diatebes_'+scaling+'_'+file+'_under.csv', index=False)
values['UnderSample'] = [len(df_positives), len(df_neg_sample), len(df_inter_sample)]
print('Minority class=', positive_class, ':', len(df_positives))
print('Majority class=', negative_class, ':', len(df_neg_sample))
print('Intermediate class=', intermediate_class, ':', len(df_inter_sample))
print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')


######### OVERSAMPLING
df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_inter_sample = DataFrame(df_intermediate.sample(len(df_negatives), replace=True))
df_over = concat([df_pos_sample, df_negatives, df_inter_sample], axis=0)
df_over.to_csv(f'data/classification/datasets_for_further_analysis/dataset1/Balancing/diatebes_'+scaling+'_'+file+'_over.csv', index=False)
values['OverSample'] = [len(df_pos_sample), len(df_negatives), len(df_inter_sample)]
print('Minority class=', positive_class, ':', len(df_pos_sample))
print('Majority class=', negative_class, ':', len(df_negatives))
print('Intermediate class=', intermediate_class, ':', len(df_inter_sample))
print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')


######### SMOTE
#RANDOM_STATE = 42
#smote = SMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE)
#y = original.pop(class_var).values
#X = original.values
#smote_X, smote_y = smote.fit_resample(X, y)
df_neg_sample_smote = DataFrame(df_negatives.sample(len(df_intermediate)))
df_pos_sample_smote = DataFrame(df_positives.sample(len(df_intermediate), replace=True))
df_smote = concat([df_pos_sample_smote, df_neg_sample_smote, df_intermediate], axis=0)
df_smote.to_csv(f'data/classification/datasets_for_further_analysis/dataset1/Balancing/diatebes_'+scaling+'_'+file+'_smote.csv', index=False)
values['Smote'] = [len(df_pos_sample_smote), len(df_neg_sample_smote), len(df_intermediate)]
print('Minority class=', positive_class, ':', len(df_pos_sample_smote))
print('Majority class=', negative_class, ':', len(df_neg_sample_smote))
print('Intermediate class=', intermediate_class, ':', len(df_intermediate))
print('Proportion:', round(len(df_pos_sample_smote) / len(df_neg_sample_smote), 2), ': 1')


#PRINT BALANCE
figure()
multiple_bar_chart([positive_class, negative_class, intermediate_class], values, title='Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/data_balancing/dataset1/diabetes'+scaling+'_'+file+'resumen_balance.png')
show()




