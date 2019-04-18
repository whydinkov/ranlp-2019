# imports, config
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from data_retrieval.helpers import database
from classifier.sklearn import pipelines
from evaluation.compare import compare_classifiers

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

load_dotenv()
db = database.MongoDB()

# data retrieval
df = pd.DataFrame(list(db.get_articles()))

# preprocesing
# X = df['text'].apply(lambda x: ' text_'.join(x.split(' '))) + \
#     df['title'].apply(lambda x: ' title_'.join(x.split(' ')))
X = df['text'] + df['title']
y = df['label']

# models


def oversample(X, y, count):
    df = X.to_frame().join(y.to_frame())

    labels = y.unique()
    result = pd.Series()

    for label in labels:
        samples = df[df['label'] == label].sample(
            count, replace=True, random_state=0)
        result = result.append(samples)

    return result[0], result['label']


# evaulate
kf = KFold(n_splits=5)
results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    svc = pipelines.make(LinearSVC(random_state=0))
    X_train, y_train = oversample(X_train, y_train, 150)
    svc.fit(X_train, y_train)
    results.append(accuracy_score(y_test, svc.predict(X_test)))
print(np.average(results))
