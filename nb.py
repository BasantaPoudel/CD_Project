from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from dscharts import plot_evaluation_results, bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

def nb_variants(file_tag, filename, target, dataset, method):
    file_tag = file_tag #'diabetes'
    filename = filename #'data/diabetes'
    target = target #'readmitted'

    train: DataFrame = read_csv(f'{filename}_train.csv')
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename}_test.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    image_location = 'images/nb/' + dataset
    savefig(image_location+'/'+method+'_'+file_tag+'_nb_best.png')
    #show()

    estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))

    figure()
    bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    savefig(image_location+'/'+method+'_'+file_tag+'_nb_study.png')
    #how()

#Last Paramenter correspond to MVI or scaling (Lab2 --> First part is MVI and Second part --> Scaling)
nb_variants('diabetes', 'data/classification/lab2_datasets/dataset1/minmax_diabetes', 'readmitted', 'dataset1', 'minmax_scaling')
#if you are going to run the zscore put comment MultinomialNB line 34
nb_variants('diabetes', 'data/classification/lab2_datasets/dataset1/zscore_diabetes', 'readmitted', 'dataset1', 'zscore_scaling')


#nb_variants('drought', 'data/classification/lab2_datasets/dataset2/minmax_drought', 'class', 'dataset2', 'minmax_scaling')
#if you are going to run the zscore put comment MultinomialNB line 34
#nb_variants('drought', 'data/classification/lab2_datasets/dataset2/zscore_drought', 'class', 'dataset2', 'zscore_scaling')