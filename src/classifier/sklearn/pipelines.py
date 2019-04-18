import json
import os
from os.path import join

from src.features import media_meta, text_meta
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(join(dir_path, 'stopwords.json'), 'rb') as f:
    bg_stopwords = json.load(f)


def get_column(column_name):
    return FunctionTransformer(lambda x: x[column_name].tolist(),
                               validate=False)


__lsa_title = ('lsa_title', Pipeline([
    ('selector', get_column('title')),
    ('vect', TfidfVectorizer(stop_words=bg_stopwords)),
    ('dim_red', TruncatedSVD(15, random_state=0)),
    ('norm', Normalizer())
]))

__lsa_text = ('lsa_text', Pipeline([
    ('selector', get_column('text')),
    ('vect', TfidfVectorizer(stop_words=bg_stopwords)),
    ('dim_red', TruncatedSVD(300, random_state=0)),
    ('norm', Normalizer())
]))

__bert_title = ('bert_title', Pipeline([
    ('selector', get_column('bert_title'))
]))

__bert_text = ('bert_text', Pipeline([
    ('selector', get_column('bert_text'))
]))

__meta_article = ('meta_article', Pipeline([
    ('selector', get_column('article')),
    ('to_dict', FunctionTransformer(
        lambda X: [text_meta.get_stats(x) for x in X], validate=False)),
    ('vect', DictVectorizer()),
    ('norm', Normalizer())
]))


__meta_media = ('meta_media', Pipeline([
    ('selector', get_column('media')),
    ('to_dict', FunctionTransformer(
        lambda X: [media_meta.get_stats(x) for x in X], validate=False)),
    ('vect', DictVectorizer()),
    ('norm', Normalizer())
]))


def make(classifier, pipeline_options):
    feat_pipes = []
    if pipeline_options['lsa_title']:
        feat_pipes.append(__lsa_title)

    if pipeline_options['lsa_text']:
        feat_pipes.append(__lsa_text)

    if pipeline_options['bert_title']:
        feat_pipes.append(__bert_title)

    if pipeline_options['bert_text']:
        feat_pipes.append(__bert_text)

    if pipeline_options['meta_article']:
        feat_pipes.append(__meta_article)

    if pipeline_options['meta_media']:
        feat_pipes.append(__meta_media)

    return Pipeline([
        ('feats', FeatureUnion(feat_pipes)),
        ('clf', classifier)
    ])
