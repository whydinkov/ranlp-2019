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
    C=1.5,
    random_state=0,
    multi_class="auto",
    solver='liblinear',
    max_iter=20000)

features = [
    'bg_bert_title',
    'bg_bert_text',
    'bg_xlm_title',
    'bg_xlm_text',
    'meta_media',
    'bg_styl_title',
    'bg_styl_text',
    'en_use_title',
    'en_use_text',
    'en_nela_title',
    'en_nela_text',
    'en_bert_title',
    'en_bert_text',
    'en_elmo_title',
    'en_elmo_text',
]


models = []

for feature in features:
    name = f'{feature}_pred'
    models.append((name, ranlp_pipelines.make(clf, [name])))

all_features = [f'{x}_pred' for x in features]
models.append(('pred_all', ranlp_pipelines.make(clf, all_features)))

compare_classifiers(models, df, df['label'], silent=False, plot=False)
