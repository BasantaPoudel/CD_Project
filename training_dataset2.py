import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import dscharts as ds
from sklearn.model_selection import train_test_split

def training_split(file_tag, data, target, positive, negative, urlfiles, scalingtype):
    data: DataFrame = read_csv(data)
    target = target
    positive = positive #0
    negative = negative #1
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values

    labels: np.ndarray = unique(y)
    labels.sort()

    #Data_distribution_per_dataset
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(urlfiles+'/'+file_tag+'_'+scalingtype+'_train.csv', index=False)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(urlfiles+'/'+file_tag+'_'+scalingtype+'_test.csv', index=False)
    values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
    values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

    plt.figure(figsize=(20, 6))
    ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset', percentage=False)
    plt.show()
    image_location = 'images/data_splitting/dataset2_scaled_' + scalingtype
    plt.savefig(image_location+'/train_test')

training_split('dataset2','data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_minmax.csv',
                'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'minmax')
training_split('dataset2','data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore.csv',
                  'class', 0, 1, 'data/classification/datasets_for_further_analysis/dataset2', 'zscore')
