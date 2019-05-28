import numpy as np
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import ranlp_pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

db = database.MongoDB()

articles = list(db.get_articles())
df = get_df(articles)

clf = LogisticRegression(
    C=0.05,
    random_state=0,
    multi_class="auto",
    solver='liblinear',
    max_iter=20000)

# feature_sets = ['bg_bert', 'bg_xlm', 'en_use',
#                 'en_nela', 'en_bert', 'en_elmo', 'bg_styl']

feature_sets = ['bg_lsa']

all_feats = []
for feature_set in feature_sets:
    all_feats.append(feature_set + '_title')
    all_feats.append(feature_set + '_text')
    if feature_set not in ['bg_styl', 'bg_lsa']:
        all_feats.append(feature_set + '_cos')
all_feats.append('meta_media')

for feature_set in all_feats:
    model = ranlp_pipelines.make(clf, [feature_set])
    pred = cross_val_predict(model,
                             df,
                             df['label'],
                             cv=5,
                             method='predict_proba')

    for article, article_pred in zip(articles, pred):
        if 'predictions' not in article:
            article['predictions'] = {}

        article['predictions'][feature_set] = article_pred.tolist()

        db.save_article(article)
    
    print(f'Done for {feature_set}')
