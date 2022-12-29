from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from dscharts import plot_evaluation_results_dataset1, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

file_tag = 'diabetes_over_noscaled'

def plot_overfitting_study(best_1, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig('images/knn/dataset1/'+file_tag+'_'+best_1+'_overfitting_'+name+'.png')


filenametrain = f'data/classification/datasets_for_further_analysis/dataset1/minmax_diabetes_train.csv'
filenametest = f'data/classification/datasets_for_further_analysis/dataset1/minmax_diabetes_test.csv'
target = 'readmitted'

train: DataFrame = read_csv(filenametrain)
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(filenametest)
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values
eval_metric = accuracy_score
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')


def knn_study():  
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
    savefig('images/knn/dataset1/'+file_tag+'_'+best[1]+'_overfitting_knn_study.png')
    show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))
    return best[0], best[1]

def knn_best(best_0,best_1):
    clf = knn = KNeighborsClassifier(n_neighbors=best_0, metric=best_1)
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results_dataset1(labels, trnY, prd_trn, tstY, prd_tst)
    savefig('images/knn/dataset1/'+file_tag+'_'+best_1+'_knn_best.png')
    show()

    d = best_1 #'euclidean'
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
    plot_overfitting_study(best_1, nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))
    show()

best_0, best_1 = knn_study()
knn_best(best_0, best_1)
#knn_best(5, 'manhattan')
#knn_best(best_0, 'euclidean')
#knn_best(best_0, 'chebyshev')
