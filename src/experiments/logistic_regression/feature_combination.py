# imports, config
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocessing.transformator import get_df
from src.evaluation.compare import compare_classifiers
from src.classifier.sklearn import pipelines
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
features = [
    ('top_1', ['bg_lsa_title', 'bg_lsa_text']),
    ('top_2', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text']),
    ('top_3', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text']),
    ('top_4', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text']),
    ('top_5', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text', 'bg_bert_title', 'bg_bert_text']),
    ('top_6', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text', 'bg_bert_title', 'bg_bert_text', 'meta_media']),
    ('top_7', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text', 'bg_bert_title', 'bg_bert_text', 'meta_media', 'bg_xlm_title', 'bg_xlm_text']),
    ('top_8', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text', 'bg_bert_title', 'bg_bert_text', 'meta_media', 'bg_xlm_title', 'bg_xlm_text', 'en_nela_title', 'en_nela_text']),
    ('top_9', ['bg_lsa_title', 'bg_lsa_text', 'en_elmo_title', 'en_elmo_text', 'en_use_title', 'en_use_text', 'en_bert_title', 'en_bert_text', 'bg_bert_title', 'bg_bert_text', 'meta_media', 'bg_xlm_title', 'bg_xlm_text', 'en_nela_title', 'en_nela_text', 'bg_styl_title', 'bg_styl_textx']),
]

oversampler = None


models = []
for name, feature_list in features:
    clf = LogisticRegression()
    clf_params = {'clf__C': 1.5, 'clf__solver': 'liblinear', 'clf__tol': 0.01}
    model = pipelines.make(clf, feature_list, clf_params=clf_params)

    # evaluation
    models.append((f'{name}', model))

compare_classifiers(models, df, df['label'], silent=False, plot=False)
