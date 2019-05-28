# imports, config
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing.transformator import get_df
from src.evaluation.compare import compare_classifiers
from src.classifier.sklearn import ranlp_pipelines
from src.data_retrieval.helpers import database
import pandas as pd


def warn(*args, **kwargs):
    pass


warnings.warn = warn


db = database.MongoDB()

df = get_df(list(db.get_articles()))

# models

feature_list = ['en_bert_title', 'en_bert_text', 'en_bert_cos']
oversampler = None
model = ranlp_pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    solver='lbfgs'
), oversampler, feature_list)

# evaluation
param_grid = {
    'clf__penalty': ['l1', 'l2'],  # l2
    'clf__tol': [1e-10, 1e-8, 1e-4, 1e-2, 1e-1],  # 1e-4
    'clf__C': [0.1, 0.05, 0.25, 0.5, 0.75, 1, 5, 10],  # 1,
    'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

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

pd.DataFrame(gs.cv_results_).to_excel('gs.xlsx')

print(f"BEST SCORE: {gs.best_score_}")
print(f"BEST SCORE: {gs.best_params_}")

# models
baseline = ranlp_pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), oversampler, feature_list)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', model),
    ('lr_tuned', gs.best_estimator_)
]

compare_classifiers(models, df, df['label'], silent=False, plot=False)
