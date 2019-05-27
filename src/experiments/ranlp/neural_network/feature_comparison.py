import numpy as np
import pandas as pd
import warnings

from src.data_retrieval.helpers import database
from src.classifier.sklearn import ranlp_pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

db = database.MongoDB()

df = get_df(list(db.get_articles()))

feature_sets = ['bg_bert', 'bg_xlm', 'bg_styl',
                'en_use', 'en_nela', 'en_bert', 'en_elmo']
all_feats = []
for feature_set in feature_sets:
    all_feats.append(feature_set + '_title')
    all_feats.append(feature_set + '_text')
    if feature_set != 'bg_styl':
        all_feats.append(feature_set + '_cos')
all_feats.append('meta_media')

models = [
    # bg_bert
    ('bg_bert_title', ranlp_pipelines.make_nn(768, ['bg_bert_title'])),
    ('bg_bert_text', ranlp_pipelines.make_nn(768, ['bg_bert_text'])),
    ('bg_bert_title_text', ranlp_pipelines.make_nn(1536, ['bg_bert_title', 'bg_bert_text'])),
    ('bg_bert_title_text_cos', ranlp_pipelines.make_nn(1537, ['bg_bert_title', 'bg_bert_text', 'bg_bert_cos'])),
    
    # bg_xlm
    ('bg_xlm_title', ranlp_pipelines.make_nn(1024, ['bg_xlm_title'])),
    ('bg_xlm_text', ranlp_pipelines.make_nn(1024, ['bg_xlm_text'])),
    ('bg_xlm_title_text', ranlp_pipelines.make_nn(2048, ['bg_xlm_title', 'bg_xlm_text'])),
    ('bg_xlm_title_text_cos', ranlp_pipelines.make_nn(2049, ['bg_xlm_title', 'bg_xlm_text', 'bg_xlm_cos'])),

    # bg_xlm
    ('bg_styl_title', ranlp_pipelines.make_nn(6, ['bg_styl_title'])),
    ('bg_styl_text', ranlp_pipelines.make_nn(9, ['bg_styl_text'])),
    ('bg_styl_title_text', ranlp_pipelines.make_nn(15, ['bg_styl_title', 'bg_styl_text'])),
    ('meta_media', ranlp_pipelines.make_nn(8, ['meta_media'])),

    # en_use
    ('en_use_title', ranlp_pipelines.make_nn(512, ['en_use_title'])),
    ('en_use_text', ranlp_pipelines.make_nn(512, ['en_use_text'])),
    ('en_use_title_text', ranlp_pipelines.make_nn(1024, ['en_use_title', 'en_use_text'])),
    ('en_use_title_text_cos', ranlp_pipelines.make_nn(1025, ['en_use_title', 'en_use_text', 'en_use_cos'])),
    
    # en_nela
    ('en_nela_title', ranlp_pipelines.make_nn(129, ['en_nela_title'])),
    ('en_nela_text', ranlp_pipelines.make_nn(129, ['en_nela_text'])),
    ('en_nela_title_text', ranlp_pipelines.make_nn(258, ['en_nela_title', 'en_nela_text'])),
    ('en_nela_title_text_cos', ranlp_pipelines.make_nn(259, ['en_nela_title', 'en_nela_text', 'en_nela_cos'])),

    # en_bert
    ('en_bert_title', ranlp_pipelines.make_nn(768, ['en_bert_title'])),
    ('en_bert_text', ranlp_pipelines.make_nn(768, ['en_bert_text'])),
    ('en_bert_title_text', ranlp_pipelines.make_nn(1536, ['en_bert_title', 'en_bert_text'])),
    ('en_bert_title_text_cos', ranlp_pipelines.make_nn(1537, ['en_bert_title', 'en_bert_text', 'en_bert_cos'])),
    
    # en_elmo
    ('en_elmo_title', ranlp_pipelines.make_nn(1024, ['en_elmo_title'])),
    ('en_elmo_text', ranlp_pipelines.make_nn(1024, ['en_elmo_text'])),
    ('en_elmo_title_text', ranlp_pipelines.make_nn(2048, ['en_elmo_title', 'en_elmo_text'])),
    ('en_elmo_title_text_cos', ranlp_pipelines.make_nn(2049, ['en_elmo_title', 'en_elmo_text', 'en_elmo_cos'])),

    # all_feats
    ('all', ranlp_pipelines.make_nn(8479, all_feats))
]

if __name__ == '__main__':
    compare_classifiers(models, df, df['label'], silent=False, plot=False)
