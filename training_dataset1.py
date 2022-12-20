import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import dscharts as ds
from sklearn.model_selection import train_test_split

def training_split(file_tag, data, target, urlfiles, scalingtype):
    print(data.shape)
    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    #Data_distribution_per_dataset
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    #print("trny",trnY)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(urlfiles+'/'+scalingtype+'_'+file_tag+'_train.csv', index=False)
    #print(train.shape)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(urlfiles+'/'+scalingtype+'_'+file_tag+'_test.csv', index=False)
    #print(test.shape)

    return train, test

data: DataFrame = read_csv('data/classification/datasets_for_further_analysis/dataset1/diabetic_fill_columns_mv.csv')

target = 'readmitted'
positive = 1 #"<30" 
negative =  0 #"NO"  
intermediate = 2 #">30" 

values = {'Original': [len(data[data[target] == negative]), len(data[data[target] == intermediate]), len(data[data[target] == positive])]}

train, test = training_split('diabetes', data, 'readmitted', 'data/classification/datasets_for_further_analysis/dataset1', 'noscaling')

values['train'] = [len(train[train[target] == negative]), len(train[train[target] == intermediate]), len(train[train[target] == positive])] 
values['test'] =  [len(test[test[target] == negative]), len(test[test[target] == intermediate]), len(test[test[target] == positive])]

plt.figure(figsize=(9, 11))
#plt.ylabel(r'$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$')
#plt.xlabel(r'$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$', fontsize=50)
ds.multiple_bar_chart([negative, intermediate, positive], values, title='Data distribution per dataset')
#plt.savefig('images/data_splitting/dataset1/diabetes_split.pdf', bbox_inches = 'tight')
plt.show()



