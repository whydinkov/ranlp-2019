# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC

db = database.MongoDB()

df = get_df(list(db.get_articles()))

pipeline_options = {
    'lsa_text': 1,
    'lsa_title': 1,
    'bert_text': 0,
    'bert_title': 0,
    'meta_article': 0,
    'meta_media': 0
}

# models
baseline = pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), pipeline_options)
svc = pipelines.make(LinearSVC(random_state=0), pipeline_options)
nb = pipelines.make(BernoulliNB(), pipeline_options)
lr = pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    solver='lbfgs'), pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('svc', svc),
    ('nb', nb),
    ('lr', lr)
]

compare_classifiers(models, df, df['label'], silent=False, plot=True)
