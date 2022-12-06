import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import dscharts as ds
from sklearn.model_selection import train_test_split

def training_split(file_tag, data, target, positive, negative, urlfiles, scalingtype):
    file_tag = file_tag #'diabetes'
    data: DataFrame = read_csv(data) #'data/diabetic_scaled_minmax.csv')
    target = target #'readmitted'
    positive = positive #1
    negative = negative #0
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    #Data_distribution_per_dataset
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(urlfiles+'/'+scalingtype+'_'+file_tag+'_train.csv', index=False)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(urlfiles+'/'+scalingtype+'_'+file_tag+'_test.csv', index=False)
    values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
    values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

    plt.figure(figsize=(12,4))
    ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
    plt.show()

<<<<<<< HEAD
training_split('diabetes','data/classification/lab2_datasets/dataset1/dataset1_scaled_minmax.csv', 
                'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'minmax')
training_split('diabetes','data/classification/lab2_datasets/dataset1/dataset1_scaled_zscore.csv', 
                  'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'zscore')
training_split('diabetes','data/classification/lab2_datasets/dataset1/mv_filled_mean_dataset1.csv', 
                  'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'zscore')
training_split('diabetes','data/classification/lab2_datasets/dataset1/mv_filled_most_frequent_dataset1.csv', 
                  'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'zscore')
=======
#training_split('diabetes','data/classification/lab2_datasets/dataset1/dataset1_scaled_minmax.csv', 
#               'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'minmax')
#training_split('diabetes','data/classification/lab2_datasets/dataset1/dataset1_scaled_zscore.csv', 
#                 'readmitted', 1, 0, 'data/classification/lab2_datasets/dataset1', 'zscore')

training_split('drought','data/classification/lab2_datasets/dataset2/dataset2_scaled_minmax.csv', 
               'class', 1, 0, 'data/classification/lab2_datasets/dataset2', 'minmax')
training_split('drought','data/classification/lab2_datasets/dataset2/dataset2_scaled_zscore.csv', 
                 'class', 1, 0, 'data/classification/lab2_datasets/dataset2', 'zscore')
>>>>>>> ed51d24eedc1f20588e15b4b07b70484fddaed2f

