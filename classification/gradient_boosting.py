from pandas import read_csv, DataFrame
from matplotlib.pyplot import savefig, show, subplots
from dscharts import multiple_line_chart, HEIGHT
from ts_functions import split_temporal_data, PREDICTION_MEASURES, plot_evaluation_results
from sklearn.ensemble import GradientBoostingRegressor
from ts_functions import plot_evaluation_results
from dscharts import plot_overfitting_study

def gradient_boosting(data, file, dataset, clas):
    trnX, tstX, trnY, tstY = split_temporal_data(data, clas, trn_pct=0.7)
    learning_rate = [.1, .5, .9]
    max_depths = [5, 10, 15, 25]
    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]

    measure = 'R2'
    flag_pct = False
    best = ('', 0, 0.0)
    last_best = -10000
    best_model = None
    ncols = len(learning_rate)

    fig, axs = subplots(1, ncols, figsize=(ncols * HEIGHT, HEIGHT), squeeze=False)
    for nr_lr in range(len(learning_rate)):
        lr = learning_rate[nr_lr]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                print(f'GB - learning rate={lr} depth={d} and nr_estimators={n}')
                pred = GradientBoostingRegressor(n_estimators=n, max_depth=d, learning_rate=lr, loss='absolute_error')
                pred.fit(trnX, trnY)
                prdY = pred.predict(tstX)
                yvalues.append(PREDICTION_MEASURES[measure](tstY, prdY))
                print(yvalues)
                if yvalues[-1] > last_best:
                    best = (lr, d, n)
                    last_best = yvalues[-1]
                    best_model = pred

            values[d] = yvalues
        multiple_line_chart(
            n_estimators, values, ax=axs[0, nr_lr], title=f'Gradient Boosting with {lr} learning rate',
            xlabel='nr estimators', ylabel=measure, percentage=flag_pct)
    savefig('images/gradient_boosting/'+dataset+'_ts_gb_study.png')
    show()
    print(
        f'Best results achieved with {best[0]} learning rate, depth={best[1]} and nr estimators={best[2]} ==> measure={last_best:.2f}')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    plot_evaluation_results(trnY, prd_trn, tstY, prd_tst, 'ts_gb_best.png')
    savefig('images/gradient_boosting/' + dataset + '_ts_gb_best.png')

    y_tst_values = []
    y_trn_values = []
    for k in n_estimators:
        pred = GradientBoostingRegressor(n_estimators=best[2], max_depth=best[1], learning_rate=best[0])
        pred.fit(trnX, trnY)
        prd_tst_Y = pred.predict(tstX)
        prd_trn_Y = pred.predict(trnX)
        y_tst_values.append(PREDICTION_MEASURES[measure](tstY, prd_tst_Y))
        y_trn_values.append(PREDICTION_MEASURES[measure](trnY, prd_trn_Y))
    plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'ts_rf_{best[0]}_{best[1]}',
                           xlabel='nr estimators', ylabel=measure, pct=flag_pct)
    savefig('images/gradient_boosting/' + dataset + 'overfitting_study.png')

''''Still need to select the final dataset'''
# data = read_csv('data/classification/datasets_for_further_analysis/dataset1/diabetic_data_variable_enconding.csv')
# gradient_boosting(data, 'dataset1', 'dataset1', 'readmitted')


data = read_csv('data/classification/datasets_for_further_analysis/dataset2/dataset2_variable_encoding.csv')
gradient_boosting(data, 'dataset2', 'dataset2', 'class')