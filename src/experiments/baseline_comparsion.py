# imports, config
import pandas as pd
from numpy.random import seed
from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC

seed(0)

db = database.MongoDB()

df = get_df(list(db.get_articles()))

# models
baseline = pipelines.make(DummyClassifier(strategy="most_frequent"))
svc = pipelines.make(LinearSVC())
nb = pipelines.make(BernoulliNB())
lr = pipelines.make(LogisticRegression(multi_class="auto", solver='lbfgs'))

# evaluation
models = [
    ('baseline', baseline),
    ('svc', svc),
    ('nb', nb),
    ('lr', lr)
]

compare_classifiers(models, df, df['label'], silent=False, plot=True)
