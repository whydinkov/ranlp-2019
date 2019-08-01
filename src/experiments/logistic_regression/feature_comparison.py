import numpy as np
import pandas as pd
import warnings

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
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

feature_sets = ['bg_bert', 'bg_xlm', 'bg_styl', 'bg_lsa',
                'en_use', 'en_nela', 'en_bert', 'en_elmo']

all_feats = []
for feature_set in feature_sets:
    all_feats.append(feature_set + '_title')
    all_feats.append(feature_set + '_text')
    if feature_set not in ['bg_styl', 'bg_lsa']:
        all_feats.append(feature_set + '_cos')
all_feats.append('meta_media')
bg_feats = [x for x in all_feats if x.startswith('bg_')] + ['meta_media']
en_feats = [x for x in all_feats if x.startswith('en_')] + ['meta_media']
models = []


for feature_set in feature_sets:
    title_model = (feature_set + '_title',
                   pipelines.make(clf, [feature_set + '_title']))
    text_model = (feature_set + '_text',
                  pipelines.make(clf, [feature_set + '_text']))
    title_text_model = (feature_set + '_title_text', pipelines.make(clf,
                                                                          [feature_set + '_title', feature_set + '_text']))

    title_text_cos_model = (feature_set + '_title_text_cos', pipelines.make(clf,
                                                                                  [feature_set + '_title', feature_set + '_text', feature_set + '_cos']))

    models.append(title_model)
    models.append(text_model)
    models.append(title_text_model)

    if feature_set not in ['bg_styl', 'bg_lsa']:
        models.append(title_text_cos_model)
    else:
        models.append(
            ('meta_media', pipelines.make(clf, ['meta_media'])))

models.append(('all', pipelines.make(clf, all_feats)))
models.append(('all_bg', pipelines.make(clf, bg_feats)))
models.append(('all_en', pipelines.make(clf, en_feats)))

if __name__ == '__main__':
    compare_classifiers(models, df, df['label'], silent=False, plot=False)
