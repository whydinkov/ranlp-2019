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


feat_setups = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1]
]

for setup in feat_setups:
    pipeline_options = {
        'lsa_title': setup[0],
        'lsa_text': setup[1],
        'bert_title': setup[2],
        'bert_text': setup[3],
        'meta_article': setup[4],
        'meta_media': setup[5]
    }

    # evaluation
    models = [
        ('clf', pipelines.make(LogisticRegression(
            random_state=0,
            multi_class="auto",
            solver='lbfgs'),
            pipeline_options)),
    ]

    compare_classifiers(models, df, df['label'], silent=False)
