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

    # train: DataFrame = read_csv(f'{filename}_train.csv')
    train: DataFrame = read_csv(f'{filename}_train_oversampling.csv')
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
    savefig(image_location+'/'+file_tag+'_nb_best.png')
    show()

    estimators = {'GaussianNB': GaussianNB(),
              # 'MultinomialNB': MultinomialNB(),
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
    savefig(image_location+'/'+file_tag+'_nb_study.png')
    #how()


# nb_variants('dataset2_minmax', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_minmax', 'class', 'dataset2', 'minmax')
# nb_variants('dataset2_minmax_balanced', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_minmax', 'class', 'dataset2', 'minmax_balanced')

# nb_variants('dataset2_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_zscore', 'class', 'dataset2', 'zscore')
nb_variants('dataset2_zscore_balanced', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_zscore', 'class', 'dataset2', 'zscore_balanced')
