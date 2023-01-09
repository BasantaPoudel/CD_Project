from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots, show, savefig
from ts_functions import HEIGHT, split_dataframe
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

#dataset1 - drop the insulin col
dataset = 'dataset1'
file_tag = 'glucose'
index_col = "Date"
target = 'Glucose'
data = read_csv('data/forecasting/glucose.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data = data.drop("Insulin", axis='columns')


#dataset2
# dataset = 'dataset2'
# file_tag = 'drought'
# index_col = 'date'
# target = 'QV2M'
# data = read_csv('data/forecasting/drought.forecasting_dataset.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

print(data.head())

#train & test spliting
def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.75)

measure = 'R2'
flag_pct = False
eval_results = {}

#simpleAvg
class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/simpleAvg/dataset2/{file_tag}_simpleAvg_eval.png')
savefig(f'images/simpleAvg/{dataset}/{file_tag}_simpleAvg_eval.png')
# show()

figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_forecasting_series(train, test, prd_trn, prd_tst, f'images/simpleAvg/dataset2/{file_tag}_simpleAvg_plots.png', x_label=index_col, y_label=target)
savefig(f'images/simpleAvg/{dataset}/{file_tag}_simpleAvg_plots.png')
# show()
