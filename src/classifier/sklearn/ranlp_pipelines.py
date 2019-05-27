from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import DictVectorizer

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


def make(classifier, columns):
    feat_pipes = []

    for column in columns:
        if column == 'meta_media':
            feat_pipes.append(_meta_media)
        else:
            feat_pipes.append(_pipe_column(column))

    return Pipeline([
        ('feats', FeatureUnion(feat_pipes)),
        ('clf', classifier)
    ])
