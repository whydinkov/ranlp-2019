from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


def make(classifier):
    return Pipeline([
        ('feats', FeatureUnion([
            ('tfidf', Pipeline([
                ('vect', TfidfVectorizer(stop_words=None)),
                ('dim_red', TruncatedSVD(300)),
                ('norm', Normalizer())
            ]))
        ])),
        ('clf', classifier)
    ])
