# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

db = database.MongoDB()

df = get_df(list(db.get_articles()))

pipeline_options = {
    'lsa_text': 1,
    'lsa_title': 1,
    'bert_text': 0,
    'bert_title': 0,
    'meta_article': 1,
    'meta_media': 0
}

# models
svc = pipelines.make(LinearSVC(random_state=0), pipeline_options)

# evaluation
param_grid = {
    'feats__lsa_title__vect__max_df': [0.5, 1, 2],
    'feats__lsa_title__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'feats__lsa_text__vect__max_df': [0.5, 1, 2],
    'feats__lsa_text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__penalty': ['l1', 'l2'],
    'clf__tol': [1e-10, 1e-8, 1e-4, 1e-2, 1e-1],
    'clf__C': [0.1, 0.5, 1, 2, 5],
}
gs = GridSearchCV(svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  error_score=-1,
                  verbose=1,
                  n_jobs=4,
                  iid=False,
                  return_train_score=True)

gs.fit(df, df['label'])

pd.DataFrame(gs.cv_results_).to_excel('gs.xlsx')
