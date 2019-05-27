import numpy as np
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import ranlp_pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.linear_model import LogisticRegression

db = database.MongoDB()

df = get_df(list(db.get_articles()))

clf = LogisticRegression(
    C=0.05,
    random_state=0,
    multi_class="auto",
    solver='liblinear',
    max_iter=20000)

feature_sets = ['bg_bert', 'bg_xlm', 'en_use', 'en_nela', 'en_bert', 'en_elmo']

# feature_sets = ['en_use']

for feature_set in feature_sets:
    models = [
        (feature_set + '_title',
         ranlp_pipelines.make(clf, [feature_set + '_title'])),
        (feature_set + '_text',
         ranlp_pipelines.make(clf, [feature_set + '_text'])),
        (feature_set + '_title_text', ranlp_pipelines.make(clf,
                                                           [feature_set + '_title', feature_set + '_text'])),
        (feature_set + '_title_text_cos', ranlp_pipelines.make(clf,
                                                               [feature_set + '_title', feature_set + '_text', feature_set + '_cos']))
    ]

    compare_classifiers(models, df, df['label'], silent=False, plot=False)
