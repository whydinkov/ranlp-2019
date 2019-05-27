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

features = [
    'bg_bert_title',
    'bg_bert_text',
    'bg_xlm_title',
    'bg_xlm_text',
    'en_use_title',
    'article_meta',
    'bg_styl_title',
    'bg_styl_text',
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
    name = f'pred_{feature}'
    models.append((name, ))

models.append(('pred_all', ranlp_pipelines.make(clf,
                                                [f'pred_{x}' fox in features])))

compare_classifiers(models, df, df['label'], silent=False, plot=False)
