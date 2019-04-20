# imports, config
import pandas as pd
import numpy as np

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df, oversample

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score

# data
db = database.MongoDB()
df = get_df(list(db.get_articles()))

# experiment
OVERSAMPLE_FRAC = 2.5
OVERSAMPLE_N = 100

skf = StratifiedKFold(n_splits=5)
scores = []
for train_index, test_index in skf.split(df, df['label']):
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]

    clf = pipelines.make_best_lr()

    X_train = oversample(X_train, frac=OVERSAMPLE_FRAC)
    # X_train = oversample(X_train, n=OVERSAMPLE_N)

    clf.fit(X_train, X_train['label'])

    y_pred = clf.predict(X_test)

    current_acc = accuracy_score(X_test['label'], y_pred)
    scores.append(current_acc)

clf = pipelines.make_best_lr()

cv_results = cross_validate(clf, df, df['label'], cv=5, scoring='accuracy')

print('oversampled acc: ', np.average(scores))
print('default acc: ', np.average(cv_results['test_score']))
