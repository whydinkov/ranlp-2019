# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df, oversample

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

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

oversampled_df = oversample(df)

# models
baseline = pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), pipeline_options)

lr = pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    solver='lbfgs'), pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', lr)
]

compare_classifiers(models, df, df['label'], silent=False)

print('with oversample')
compare_classifiers(models, oversampled_df, oversampled_df['label'],
                    silent=False)
