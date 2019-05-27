import numpy as np
import pandas as pd
import warnings

from src.data_retrieval.helpers import database
from src.classifier.sklearn import ranlp_pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

db = database.MongoDB()

df = get_df(list(db.get_articles()))

clf = LogisticRegression(
    C=0.05,
    random_state=0,
    multi_class="auto",
    solver='liblinear',
    max_iter=20000)

feature_sets = ['bg_bert', 'bg_xlm', 'bg_styl',
                'en_use', 'en_nela', 'en_bert', 'en_elmo']

all_feats = []
for feature_set in feature_sets:
    all_feats.append(feature_set + '_title')
    all_feats.append(feature_set + '_text')
    if feature_set != 'bg_styl':
        all_feats.append(feature_set + '_cos')
all_feats.append('meta_media')

models = []

models.add(('meta_media', ranlp_pipelines.make(clf, ['meta_media'])))
models.add(('all', ranlp_pipelines.make(clf, all_feats)))

for feature_set in feature_sets:
    title_model = (feature_set + '_title',
                   ranlp_pipelines.make(clf, [feature_set + '_title']))
    text_model = (feature_set + '_text',
                  ranlp_pipelines.make(clf, [feature_set + '_text']))
    title_text_model = (feature_set + '_title_text', ranlp_pipelines.make(clf,
                                                                          [feature_set + '_title', feature_set + '_text']))

    title_text_cos_model = (feature_set + '_title_text_cos', ranlp_pipelines.make(clf,
                                                                                  [feature_set + '_title', feature_set + '_text', feature_set + '_cos']))

    models.add(title_model)
    models.add(text_model)
    models.add(title_text_model)

    if feature_set != 'bg_styl':
        models.add(title_text_cos_model)

if __name__ == '__main__':
    compare_classifiers(models, df, df['label'], silent=False, plot=False)
