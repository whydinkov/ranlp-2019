from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion


def get_column(column_name):
    return FunctionTransformer(lambda x: x[column_name].tolist(),
                               validate=False)


def _pipe_column(name):
    return (name, Pipeline([
        ('selector', get_column(name))
    ]))


def make(classifier, columns):
    feat_pipes = []

    for column in columns:
        feat_pipes.append(_pipe_column(column))

    return Pipeline([
        ('feats', FeatureUnion(feat_pipes)),
        ('clf', classifier)
    ])
