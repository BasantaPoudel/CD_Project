from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from dscharts import bar_chart


register_matplotlib_converters()
file = 'diabetic'
#filename = 'data/classification/diabetic_data.csv'
filename = 'forecasting/data/classification/datasets_for_further_analysis/dataset1/diabetic_data_variable_encoding.csv'
data = read_csv(filename, index_col='encounter_id', na_values='?', parse_dates=True, infer_datetime_format=True)

# defines the number of records to discard entire columns
threshold = data.shape[0] * 0.90

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

missings = [c for c in mv.keys() if mv[c]>threshold]
df = data.drop(columns=missings, inplace=False)

if 'weight' in df:
    df = df.drop("weight", axis='columns')

#if 'payer_code' in df:
    #df = df.drop("payer_code", axis='columns')

if 'medical_specialty' in df:
    df = df.drop("medical_specialty", axis='columns')

if 'examide' in df:
    df = df.drop("examide", axis='columns')

if 'citoglipton' in df:
    df = df.drop("citoglipton", axis='columns')

if 'metformin-rosiglitazone' in df:
    df = df.drop("metformin-rosiglitazone", axis='columns')

df = df.dropna()

df.to_csv(f'data/classification/datasets_for_further_analysis/dataset1/{file}_drop_columns_mv.csv', index=True)
print('Dropped variables', missings)

filename = f'data/classification/datasets_for_further_analysis/dataset1/{file}_drop_columns_mv.csv'
data = read_csv(filename, index_col='encounter_id', na_values='?', parse_dates=True, infer_datetime_format=True)
#data["payer_code"] = data["payer_code"].fillna('NN')
#data["medical_specialty"] = data["medical_specialty"].fillna('NN')

#data = data.dropna()

#data.to_csv(f'data/classification/datasets_for_further_analysis/dataset1/{file}_fill_columns_mv.csv', index=True)

