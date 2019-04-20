# imports, config
import pandas as pd
import warnings

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

db = database.MongoDB()

df = get_df(list(db.get_articles()))

# models
model = pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    solver='lbfgs'
))

# evaluation
param_grid = {
    # 'feats__lsa_title__vect__min_df': [1, 2],  # 1
    # 'feats__lsa_title__vect__ngram_range': [(1, 1), (1, 2)],  # (1, 1)
    # 'feats__lsa_text__vect__min_df': [1, 2, 5],  # 1
    # 'feats__lsa_text__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

    'clf__penalty': ['l1', 'l2'],  # l2
    'clf__tol': [1e-10, 1e-8, 1e-4, 1e-2, 1e-1],  # 1e-4
    'clf__C': [0.1, 0.5, 1, 5, 10],  # 1,
    'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

gs = GridSearchCV(model,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  error_score=-1,
                  verbose=2,
                  n_jobs=-1,
                  iid=False,
                  return_train_score=True)

gs.fit(df, df['label'])

pd.DataFrame(gs.cv_results_).to_excel('gs.xlsx')

print(f"BEST SCORE: {gs.best_score_}")
print(f"BEST SCORE: {gs.best_params_}")

# models
baseline = pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', model),
    ('lr_tuned', gs.best_estimator_)
]

compare_classifiers(models, df, df['label'], silent=False)
