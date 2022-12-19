from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score


def random_forest(filename, file, dataset, clas):
    target = clas

    train: DataFrame = read_csv(f'{filename}_train.csv')
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()
    print(labels)

    test: DataFrame = read_csv(f'{filename}_test.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10, 25]
    max_features = [.3, .5, .7, 1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    savefig('images/decision_tree/'+dataset+'/_rf_study.png')
    show()
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f' % (
    best[0], best[1], best[2], last_best))



# random_forest('data/classification/data_for_DT_RF/minmax_diabetes',
#               'dataset1', 'dataset1', 'readmitted')

random_forest('data/classification/data_for_DT_RF/dataset2_minmax',
              'dataset2', 'dataset2', 'class')