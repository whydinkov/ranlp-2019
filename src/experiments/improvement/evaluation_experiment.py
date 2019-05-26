# imports, config
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.preprocessing.transformator import get_df, oversample
from src.evaluation.confusion_matrix import plot

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# data
db = database.MongoDB()
df = get_df(list(db.get_articles()))

# experiment
X_train, X_test, y_train, y_test = train_test_split(
    df,
    df['label'],
    test_size=0.2,
    random_state=0
)

clf = pipelines.make_best_lr()

X_train = oversample(X_train, n=500)

clf.fit(X_train, X_train['label'])

y_pred = clf.predict(X_test)

plot(y_test,
     y_pred,
     classes=clf.classes_)

plt.show()
