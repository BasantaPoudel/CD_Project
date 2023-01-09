from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots, savefig, show
from dscharts import multiple_line_chart, HEIGHT
from ts_functions import split_temporal_data, PREDICTION_MEASURES, plot_evaluation_results
from sklearn.neural_network import MLPRegressor
from dscharts import plot_overfitting_study
import numpy
from numpy import ndarray

def mlp(filename, file, dataset, clas):
    target = clas

    if dataset == 'dataset2':
         train: DataFrame = read_csv(f'{filename}_undersampling.csv')
    # train=train.head(10000)
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = numpy.unique(trnY)
    labels.sort()
    print(labels)

    test: DataFrame = read_csv(f'{filename}_test.csv')
    # test=test.head(10000)
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values
    # trnX, tstX, trnY, tstY = split_temporal_data(data, clas, trn_pct=0.7)

    lr_type = ['constant']  # , 'invscaling', 'adaptive'] - only used if optimizer='sgd'
    learning_rate = [.9, .6, .3, .1]
    max_iter = [100, 150, 250, 500, 1000]
    max_iter_warm_start = [max_iter[0]]
    for el in max_iter[1:]:
        max_iter_warm_start.append(max_iter_warm_start[-1] + el)

    measure = 'R2'
    flag_pct = False
    best = ('', 0, 0.0)
    last_best = -10000
    best_model = None
    ncols = len(lr_type)

    fig, axs = subplots(1, ncols, figsize=(ncols * HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(lr_type)):
        tp = lr_type[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            warm_start = False
            for n in max_iter:
                print(f'MLP - lr type={tp} learning rate={lr} and nr_episodes={n}')
                pred = MLPRegressor(
                    learning_rate=tp, learning_rate_init=lr, max_iter=n,
                    activation='relu', warm_start=warm_start, verbose=False)
                pred.fit(trnX, trnY)
                prdY = pred.predict(tstX)
                yvalues.append(PREDICTION_MEASURES[measure](tstY, prdY))
                warm_start = True
                if yvalues[-1] > last_best:
                    best = (tp, lr, n)
                    last_best = yvalues[-1]
                    best_model = pred
            values[lr] = yvalues

        multiple_line_chart(
            max_iter_warm_start, values, ax=axs[0, k], title=f'MLP with lr_type={tp}', xlabel='mx iter', ylabel=measure,
            percentage=flag_pct)
    savefig('images/mlp/'+dataset+'_ts_mlp_study.png')
    show()
    print(
        f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter ==> measure={last_best:.2f}')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plot_evaluation_results(trnY, prd_trn, tstY, prd_tst, 'ts_mlp_best')
    savefig('images/mlp/'+dataset+'_ts_mlp_best.png')

    y_tst_values = []
    y_trn_values = []
    warm_start = False
    for n in max_iter:
        # print(f'MLP - lr type={best[1]} learning rate={best[0]} and nr_episodes={n}')
        MLPRegressor(
            learning_rate=best[0], learning_rate_init=best[1], max_iter=n,
            activation='relu', warm_start=warm_start, verbose=False)
        pred.fit(trnX, trnY)
        prd_tst_Y = pred.predict(tstX)
        prd_trn_Y = pred.predict(trnX)
        y_tst_values.append(PREDICTION_MEASURES[measure](tstY, prd_tst_Y))
        y_trn_values.append(PREDICTION_MEASURES[measure](trnY, prd_trn_Y))
        warm_start = True
    plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'ts_NN_{best[0]}_{best[1]}',
                           xlabel='nr episodes', ylabel=measure, pct=flag_pct)
    savefig('images/mlp/' + dataset + 'mlp_overfitting.png')

# data = read_csv('data/classification/datasets_for_further_analysis/dataset1/diabetic_data_variable_enconding.csv')
# mlp(data, 'dataset1', 'dataset1', 'readmitted')


mlp('data/classification/datasets_for_further_analysis/dataset2/dataset2_feature_engineering', 'dataset2', 'dataset2', 'class')