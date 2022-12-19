from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.tree import DecisionTreeClassifier
from dscharts import plot_evaluation_results_Maribel, multiple_line_chart, plot_evaluation_results
from sklearn.metrics import accuracy_score
from sklearn import tree
from numpy import argsort, arange
from dscharts import horizontal_bar_chart
from matplotlib.pyplot import Axes
from dscharts import plot_overfitting_study
import sklearn



def decision_tree(filename, file, dataset, clas):
    target = clas

    train: DataFrame = read_csv(f'{filename}_train.csv')
    train=train.head(10000)
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()
    print(labels)

    test: DataFrame = read_csv(f'{filename}_test.csv')
    test=test.head(10000)
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('', 0, 0.0)
    last_best = 0
    best_model = None

    figure()
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                print(imp)
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    savefig('images/decision_tree/'+dataset+'/_dt_study.png')
    # show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
    best[0], best[1], best[2], last_best))

    # plot the tree
    labels = [str(value) for value in labels]
    print(labels)
    sklearn.tree.plot_tree(best_model, feature_names=train.columns, class_names=labels)
    savefig('images/decision_tree/'+dataset+'/_dt_best_tree.png')

    # plot evaluation results
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    if dataset == 'dataset1':
        plot_evaluation_results_Maribel(labels, trnY, prd_trn, tstY, prd_tst)
        savefig('images/decision_tree/'+dataset+'/_dt_best.png')
        show()
    else:
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        savefig('images/decision_tree/' + dataset + '/_dt_best.png')
        show()


    # plot feature importances
    variables = train.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance',
                         ylabel='variables')
    savefig('images/decision_tree/'+dataset+'/_dt_ranking.png')

    # plot overfitting study
    imp = 0.0001
    f = 'entropy'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for d in max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(trnX, trnY)
        prdY = tree.predict(tstX)
        prd_tst_Y = tree.predict(tstX)
        prd_trn_Y = tree.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth',
                           ylabel=str(eval_metric))
    savefig('images/decision_tree/' + dataset + '/_overfitting_study.png')


decision_tree('data/classification/data_for_DT_RF/minmax_diabetes',
              'dataset1', 'dataset1', 'readmitted')

# decision_tree('data/classification/data_for_DT_RF/dataset2_minmax',
#               'dataset2', 'dataset2', 'class')

