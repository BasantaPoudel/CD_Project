from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
from matplotlib.pyplot import show, savefig


def main(data, dataset, granularity, train, test, prd_trn, prd_tst, index_col, target):
    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, 'persistence_eval_'+granularity)
    savefig(f'images/time_series/persistance_model/'+dataset+'/persistence_eval'+granularity+'.png')

    plot_forecasting_series(train, test, prd_trn, prd_tst, 'persistence_plots+'+granularity,
                            x_label=index_col,
                            y_label=target)
    savefig(f'images/time_series/persistance_model/'+dataset+'/persistence_plots'+granularity+'.png')
    show()

class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd


def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test


eval_results = {}
measure = 'R2'


data = read_csv('data/forecasting/aggregation/dataset2/smothing_100_daily.csv', index_col='timestamp', sep=',', decimal='.', parse_dates=True, dayfirst=True, infer_datetime_format=True)
# data.drop(['Insulin'], axis=1, inplace=True)
data = data.iloc[100:,:]
train, test = split_dataframe(data, trn_pct = 0.7)

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)


main(data, 'dataset2', 'smoothing_100_daily', train, test, prd_trn, prd_tst, 'timestamp', 'Glucose')
