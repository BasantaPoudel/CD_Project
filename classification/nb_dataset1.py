from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from dscharts import plot_evaluation_results_dataset1, bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score


file_tag = 'dataset1_0.2_0.1_balanced_over'
filenametrain = f'data/classification/datasets_for_further_analysis/dataset1/Balancing/diatebes_dataset1_0.2_0.1_balanced_over.csv'
filenametest = f'data/classification/datasets_for_further_analysis/dataset1/Balancing/dataset1_0.2_0.1_feature_engineering_diabetes_test.csv'
target = 'readmitted'

train: DataFrame = read_csv(filenametrain)
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(filenametest)
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

clf = GaussianNB()
print(f'Best Classifier from the study: {clf}')
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results_dataset1(labels, trnY, prd_trn, tstY, prd_tst)
savefig('images/nb/dataset1/'+file_tag+'_nb_best.png')
#show()

''' estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/nb/dataset1/'+file_tag+'_nb_study.png')
show() '''