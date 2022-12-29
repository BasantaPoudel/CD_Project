from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from dscharts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score


def knn_variants(file_tag, filename, target, dataset, method):
    #Running over unbalanced
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
    if method == '':
        savefig(image_location+'/'+file_tag+'_knn_study.png')
    else:
        #Running over balanced
        savefig(image_location+'/'+file_tag+'_'+method+'_knn_study.png')

    #show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    knn_best(best, file_tag, image_location, labels, trnX, trnY, tstX, tstY, method)

    # overfitting_study(best, dataset, method, n, nvalues, trnX, trnY, tstX, tstY)


def knn_best(best, file_tag, image_location, labels, trnX, trnY, tstX, tstY, method):
    # Knn Best
    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    if method == '':
        savefig(image_location+'/'+file_tag+'_knn_best.png')
    else:
        #Running over balanced
        savefig(image_location+'/'+file_tag+'_'+method+'_knn_best.png')
    # show()


def overfitting_study(best, dataset, method, n, nvalues, trnX, trnY, tstX, tstY):
    # #Overfitting
    d = best[1]  # 'euclidean'
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
    plot_overfitting_study(dataset, method, nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K',
                           ylabel=str(eval_metric))


def plot_overfitting_study(dataset, method, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    image_location = 'images/knn/' + dataset
    savefig(image_location+'/'+dataset+'_'+method+'_overfitting_'+name+'.png')



# (file_tag, filename, target, dataset, method)
#Running for data preparation step - outliers treatment
# knn_variants('dataset2_drop_outliers', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_drop_outliers', 'class', 'dataset2', 'drop_outliers')

#Running for data preparation step - scaling
# knn_variants('dataset2_scaled_minmax', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_minmax', 'class', 'dataset2', 'scaled_minmax')

#Running for classification steps
#Running over unbalanced - Change the train file in the knn_variants function
# knn_variants('dataset2_minmax', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_minmax', 'class', 'dataset2', 'minmax')

#Running over balanced - Change the train file in the knn_variants function
# knn_variants('dataset2_minmax_oversampling', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_minmax', 'class', 'dataset2', 'minmax_oversampling')

#Running over unbalanced - Change the train file in the knn_variants function
# knn_variants('dataset2_zscore', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_zscore', 'class', 'dataset2', 'zscore')

#Running over balanced - Change the train file in the knn_variants function
# knn_variants('dataset2_zscore_oversampling', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_zscore', 'class', 'dataset2', 'zscore_oversampling')
