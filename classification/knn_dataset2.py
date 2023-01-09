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
    # nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    #knn study with higher values
    nvalues = [1, 3, 5, 7, 9] #best k conclusion
    n = 9
    # dist = ['manhattan', 'euclidean', 'chebyshev']
    dist = ['manhattan'] #best distance conclusion
    values = {}
    best = (9, 'manhattan')
    # best = (0, '')
    # last_best = 0
    # for d in dist:
    #     y_tst_values = []
    #     for n in nvalues:
    #         knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    #         knn.fit(trnX, trnY)
    #         prd_tst_Y = knn.predict(tstX)
    #         y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    #         if y_tst_values[-1] > last_best:
    #             best = (n, d)
    #             last_best = y_tst_values[-1]
    #     values[d] = y_tst_values
    #
    # figure()
    # multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    #preparation steps
    # image_location = 'images/knn/preparation_steps/' + dataset
    #knn study steps
    image_location = 'images/knn/knn_study/' + dataset
    # if method == '':
    #     savefig(image_location+'/'+file_tag+'_knn_study.png')
    # else:
    #     #Running over balanced
    #     savefig(image_location+'/'+file_tag+'_'+method+'_knn_study.png')

    #show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    knn_best(best, file_tag, image_location, labels, trnX, trnY, tstX, tstY, method)

    overfitting_study(best, file_tag, dataset, method, image_location, n, nvalues, trnX, trnY, tstX, tstY)


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


def overfitting_study(best, file_tag, dataset, method, image_location, n, nvalues, trnX, trnY, tstX, tstY):
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
    plot_overfitting_study(file_tag, dataset, method, image_location, nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K',
                           ylabel=str(eval_metric))


def plot_overfitting_study(file_tag, dataset, method, image_location, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(image_location+'/'+file_tag+'_'+method+'_overfitting_'+name+'.png')



# TODO - Change the image_location variable and other parameters
# (file_tag, filename, target, dataset, method)
#Running for classification steps
knn_variants('dataset2_feature_engineering', 'data/classification/datasets_for_further_analysis/dataset2/dataset2_feature_engineering', 'class', 'dataset2', 'undersampling')
