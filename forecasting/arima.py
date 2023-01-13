from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pyplot import subplots, savefig
from dscharts import multiple_line_chart
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, split_dataframe
from statsmodels.tsa.arima.model import ARIMA
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

#DEFINE dataframe

# dataset = 'dataset1'
# granularity = '_standard'
# index = 'Date'
# target = 'Glucose'
# data = read_csv('data/forecasting/glucose.csv', index_col=index, sep=',', decimal='.', parse_dates=True, dayfirst=True, infer_datetime_format=True)
# data.drop(['Insulin'], axis=1, inplace=True)
# data = data.sort_values('Date')

# print(data.head())

dataset = 'dataset2'
granularity = '_standard'
index = 'date'
target = 'QV2M'
data = read_csv('data/forecasting/drought.forecasting_dataset.csv', index_col=index, sep=',', decimal='.', parse_dates=True, dayfirst=True, infer_datetime_format=True)
data.drop(["PRECTOT", "PS", "T2M", "T2MDEW", "T2MWET", "TS"], axis=1, inplace=True)
data.index.freq = 'D'
data = data.sort_values('date')

# print(data.head())

train, test = split_dataframe(data, trn_pct=0.75)

# ARIMA Study with different values
# pred = ARIMA(train, order=(p, d, q))
# pred = ARIMA(train, order=(1, 0, 3))
# model = pred.fit(method_kwargs={'warn_convergence': False})
# prd_tst = model.forecast(steps=len(test), signal_only=False)
# prd_trn = model.predict(start=0, end=len(train)-1)

# model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))
# savefig(f'images/arima/'+dataset+'/arima_diagnostics'+granularity+'.png')

measure = 'R2'
flag_pct = False
last_best = -100
best = ('',  0, 0.0)
best_model = None

d_values = (0, 1, 2)
params = (1, 2, 3, 5)
ncols = len(d_values)

fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)

for der in range(len(d_values)):
    d = d_values[der]
    values = {}
    for q in params:
        yvalues = []
        for p in params:
            pred = ARIMA(train, order=(p, d, q))
            model = pred.fit(method_kwargs={'warn_convergence': False})
            prd_tst = model.forecast(steps=len(test), signal_only=False)
            yvalues.append(PREDICTION_MEASURES[measure](test,prd_tst))
            if yvalues[-1] > last_best:
                best = (p, d, q)
                last_best = yvalues[-1]
                best_model = model
        values[q] = yvalues
    multiple_line_chart(
        params, values, ax=axs[0, der], title=f'ARIMA d={d}', xlabel='p', ylabel=measure, percentage=flag_pct)
savefig(f'images/arima/{dataset}/{dataset}_ts_arima_study.png')
print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')

prd_trn = best_model.predict(start=0, end=len(train)-1)
prd_tst = best_model.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'arima_eval_'+granularity)
savefig(f'images/arima/'+dataset+'/arima_eval'+granularity+'.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, 'arima_plots_'+granularity, x_label= str(index), y_label=str(target))
savefig(f'images/arima/'+dataset+'/arima_plots'+granularity+'.png')
