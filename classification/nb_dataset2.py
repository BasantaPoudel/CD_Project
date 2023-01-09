from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from dscharts import plot_evaluation_results, bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score


def nb_variants(file_tag, filename, target, dataset, method):
    file_tag = file_tag
    filename = filename
    target = target
    if method == '':
        train: DataFrame = read_csv(f'{filename}_train.csv')
    else:
        #Running over balanced
        train: DataFrame = read_csv(f'{filename}'+'_'+method+'.csv')

    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename}_test.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values


    estimators = {'GaussianNB': GaussianNB(),
                  # 'MultinomialNB': MultinomialNB(),
                  'BernoulliNB': BernoulliNB()
                  # 'CategoricalNB': CategoricalNB
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
    #preparation steps
    # image_location = 'images/nb/preparation_steps/' + dataset
    #nb_study
    image_location = 'images/nb/nb_study/' + dataset
    if method == '':
        savefig(image_location+'/'+file_tag+'_nb_study.png')
    else:
        #Running over balanced
        savefig(image_location+'/'+file_tag+'_'+method+'_nb_study.png')

    nb_best(dataset, file_tag, image_location, labels, trnX, trnY, tstX, tstY, method)
    #show()


def nb_best(dataset, file_tag, image_location, labels, trnX, trnY, tstX, tstY, method):
    clf = GaussianNB()
    print(f'Best Classifier from the study: {clf}')
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    if method == '':
        savefig(image_location+'/'+file_tag+'_nb_best.png')
    else:
        #Running over balanced
        savefig(image_location+'/'+file_tag+'_'+method+'_nb_best.png')    # show()


# TODO - Change the image_location variable and other parameters
# nb_variants('dataset2_scaled_zscore_model',
#             'forecasting/data/classification/datasets_for_further_analysis/dataset2/dataset2_scaled_zscore', 'class', 'dataset2', 'undersampling')
# nb_variants('dataset2_feature_engineering', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_feature_engineering', 'class', 'dataset2', 'undersampling')
