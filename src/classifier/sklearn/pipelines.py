import os
from os.path import join
import json

from imblearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.layers import Dense, Dropout
# from keras.models import Sequential

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(join(dir_path, 'stopwords.json'), 'rb') as f:
    bg_stopwords = json.load(f)


def get_column(column_name):
    return FunctionTransformer(lambda x: x[column_name].tolist(),
                               validate=False)


def _pipe_column(name):
    return (name, Pipeline([
        ('selector', get_column(name))
    ]))


_meta_media = ('meta_media', FeatureUnion([
    ('category', Pipeline([
        ('selector', get_column('media_cat')),
        ('vect', DictVectorizer())
    ])),
    ('numerical', Pipeline([
        ('selector', get_column('media_num')),
        ('vect', DictVectorizer()),
        ('norm', Normalizer()),
    ]))
]))

__lsa_title = ('lsa_title', Pipeline([
    ('selector', get_column('title')),
    ('vect', TfidfVectorizer(stop_words=bg_stopwords)),
    ('dim_red', TruncatedSVD(15, random_state=0)),
    ('norm', Normalizer())
]))

__lsa_text = ('lsa_text', Pipeline([
    ('selector', get_column('text')),
    ('vect', TfidfVectorizer(stop_words=bg_stopwords)),
    ('dim_red', TruncatedSVD(200, random_state=0)),
    ('norm', Normalizer())
]))


def make(classifier, columns, oversampler=None, clf_params={}):
    feat_pipes = []

    for column in columns:
        if column == 'meta_media':
            feat_pipes.append(_meta_media)
        elif column == 'bg_lsa_title':
            feat_pipes.append(__lsa_title)
        elif column == 'bg_lsa_text':
            feat_pipes.append(__lsa_text)
        else:
            feat_pipes.append(_pipe_column(column))

    default_params = {
        'clf__max_iter': 1000,
        'clf__random_state': 0,
        'clf__multi_class': "auto"
    }

    if oversampler:
        pipeline = Pipeline([
            ('feats', FeatureUnion(feat_pipes)),
            ('oversampler', oversampler),
            ('clf', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('feats', FeatureUnion(feat_pipes)),
            ('clf', classifier)
        ])

    params = {**default_params, **clf_params}
    pipeline.set_params(**params)

    return pipeline
