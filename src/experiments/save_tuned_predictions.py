import numpy as np
import pandas as pd

from src.data_retrieval.helpers import in_memory
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df
from sklearn.model_selection import cross_val_predict, GridSearchCV

from sklearn.linear_model import LogisticRegression

#db = database.MongoDB()

#articles = list(db.get_articles())
articles = in_memory.get_articles()
df = get_df(articles)

clf = LogisticRegression()

feature_sets = [
    #'bg_bert',
     'bg_xlm',
    # 'bg_styl',
    # 'bg_lsa',
    # 'en_use',
    # 'en_nela',
    # 'en_bert',
    # 'en_elmo'
]

all_feats = []
for feature_set in feature_sets:
    all_feats.append(feature_set + '_title')
    all_feats.append(feature_set + '_text')
#all_feats.append('meta_media')


param_grid = {
    'clf__tol': [1e-10, 1e-8, 1e-4, 1e-2, 1e-1],  # 1e-4
    'clf__C': [0.05, 0.15, 0.25, 0.35, 0.50, 0.75, 1, 1.25, 1.5, 2],  # 1,
    'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # lbfgs
}

print('All features count: ', len(all_feats))
for feature_set in all_feats:
    model = pipelines.make(clf, [feature_set])

    gs = GridSearchCV(model,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=5,
                      error_score=-1,
                      verbose=1,
                      n_jobs=-1,
                      iid=False,
                      return_train_score=True)

    gs.fit(df, df['label'])

    print(f"{feature_set} | BEST SCORE: {gs.best_score_}")
    print(f"{feature_set} | BEST PARAMS: {gs.best_params_}")

 #   pred = cross_val_predict(gs.best_estimator_,
 #                            df,
 #                            df['label'],
 #                            cv=5,
 #                            method='predict_proba')
 #
 #   for article, article_pred in zip(articles, pred):
 #       if 'tuned_predictions' not in article:
 #           article['tuned_predictions'] = {}
 #
 #       article['tuned_predictions'][feature_set] = article_pred.tolist()
 #
 #       db.save_article(article)

    print(f'Done for {feature_set}')
