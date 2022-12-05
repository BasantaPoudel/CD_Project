from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from dscharts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score


def knn_variants(file_tag, filename, target, dataset, method):
    train: DataFrame = read_csv(f'{filename}_train.csv')
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename}_test.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    eval_metric = accuracy_score
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        y_tst_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
        values[d] = y_tst_values

    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    image_location = 'images/knn/' + dataset
    savefig(image_location+'/'+method+'_'+file_tag+'_knn_study.png')
    #show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    #Knn Best
    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(image_location+'/'+method+'_'+file_tag+'_knn_best.png')
    #show()

    #Overfitting
    d = 'euclidean'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        prd_trn_Y = knn.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(dataset, method, nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))


def plot_overfitting_study(dataset, method, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    image_location = 'images/knn/' + dataset
    savefig(image_location+'/overfitting_'+name+'_'+method+'.png')


#Last Paramenter correspond to MVI or scaling (Lab2 --> First part is MVI and Second part --> Scaling)
knn_variants('diabetes', 'data/classification/lab2_datasets/dataset1/minmax_diabetes', 'readmitted', 'dataset1', 'minmax_scaling')
knn_variants('diabetes', 'data/classification/lab2_datasets/dataset1/zscore_diabetes', 'readmitted', 'dataset1', 'zscore_scaling')