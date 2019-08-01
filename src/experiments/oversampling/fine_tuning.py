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

features = [
    #('bg_bert_title_text', ['bg_bert_title', 'bg_bert_text']),
    #('bg_xlm_title_text', ['bg_xlm_title', 'bg_xlm_text']),
    #('bg_styl_title_text', ['bg_styl_title', 'bg_styl_text']),
    #('bg_lsa_title_text', ['bg_lsa_title', 'bg_lsa_text']),
    #('all_bg', bg_feats),
    #('en_use_title_text', ['en_use_title', 'en_use_text']),
    #('en_nela_title_text', ['en_nela_title', 'en_nela_text']),
    #('en_bert_title_text', ['en_bert_title', 'en_bert_text']),
    #('en_elmo_title_text', ['en_elmo_title', 'en_elmo_text']),
    # ('all_en', en_feats),
      ('all', all_feats)
]
oversampler = None

# evaluation
param_grid = {
    # 'clf__tol': [1e-2], # [1e-10, 1e-8, 1e-4, 1e-2, 1e-1],  # 1e-4
    'clf__C': [0.05, 0.15, 0.25, 0.35, 0.50, 0.75, 1, 1.25, 1.5, 2],  # 1,
    'clf__solver': ['liblinear'] #['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

for name, feature_list in features:
    model = ranlp_pipelines.make(LogisticRegression(
        random_state=0,
        multi_class="auto",
        max_iter=1000
    ), feature_list)

    print(name)

    gs = GridSearchCV(model,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=5,
                      error_score=-1,
                      verbose=10000,
                      n_jobs=-1,
                      iid=False,
                      return_train_score=True)

    gs.fit(df, df['label'])

    pd.DataFrame(gs.cv_results_).to_excel(f'{name}_gs.xlsx')

    print(f"{name} | BEST SCORE: {gs.best_score_}")
    print(f"{name} | BEST PARAMS: {gs.best_params_}")

    # evaluation
    models = [
        (f'{name}', gs.best_estimator_)
    ]

    compare_classifiers(models, df, df['label'], silent=False, plot=False)
