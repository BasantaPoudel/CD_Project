import variable_encoding_dataset2 as ve
import outliers_treatment_dataset2 as ot
import train_test_split_dataset2 as tr
import knn_dataset2 as knn
import nb_dataset2 as nb
import scaling_dataset2 as sc
import data_balancing_dataset2 as db

# # #Variable Encoding
# ve.data_encoding('data/classification/drought.csv')
# #
# # #Outliers Treatment
# ot.outliers_treatment('data/classification/datasets_for_further_analysis/dataset2/dataset2_variable_encoding.csv', 'dataset2_variable_encoding', 'dataset2', '')
# #
# # #Train Test Splitting
# tr.training_split('dataset2', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_drop_outliers.csv', 'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'drop_outliers')
# tr.training_split('dataset2', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_replace_outliers.csv', 'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'replace_outliers')
# tr.training_split('dataset2', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_truncate_outliers.csv', 'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'truncate_outliers')

#
# #Evaluate the outliers treatment approaches
# knn.knn_variants('dataset2_drop_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_drop_outliers', 'class', 'dataset2', '')
# knn.knn_variants('dataset2_replace_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_replace_outliers', 'class', 'dataset2', '')
# knn.knn_variants('dataset2_truncate_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_truncate_outliers', 'class', 'dataset2', '')
#
# nb.nb_variants('dataset2_drop_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_drop_outliers', 'class', 'dataset2', '')
# nb.nb_variants('dataset2_replace_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_replace_outliers', 'class', 'dataset2', '')
# nb.nb_variants('dataset2_truncate_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_truncate_outliers', 'class', 'dataset2', '')

#
# #Scaling - Run after analysing and selecting correct approach for outliers treatment
# file_location = 'data/classification/datasets_for_further_analysis/dataset2/'
# best_approach_file = 'dataset2_drop_outliers.csv'
# sc.scaling(file_location+best_approach_file, "dataset2")

# #Train Test Splitting
# tr.training_split('dataset2', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_minmax.csv', 'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'scaled_minmax')
# tr.training_split('dataset2', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore.csv', 'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'scaled_zscore')

#
#Evaluate the scaling approaches
# knn.knn_variants('dataset2_scaled_minmax', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_minmax', 'class', 'dataset2', '')
# knn.knn_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', '')
#
# nb.nb_variants('dataset2_scaled_minmax', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_minmax', 'class', 'dataset2', '')
# nb.nb_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', '')

# Balancing dataset
# file_location = 'data/classification/datasets_for_further_analysis/dataset2/'
# best_approach_file = 'dataset2_scaled_zscore'
# db.data_balancing(file_location + f'{best_approach_file}_train.csv', best_approach_file, 'dataset2', 'date', 'class', '')


#
# #Evaluate the balancing approaches
# knn.knn_variants(best_approach_file, file_location+best_approach_file, 'class', 'dataset2', 'oversampling')

# knn.knn_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'oversampling')
# knn.knn_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'undersampling')
# knn.knn_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'smote')

# nb.nb_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'oversampling')
# nb.nb_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'undersampling')
# nb.nb_variants('dataset2_scaled_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'smote')