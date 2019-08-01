import numpy as np
import pandas as pd
import warnings

from src.data_retrieval.helpers import database
from src.classifier.sklearn import ranlp_pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

db = database.MongoDB()

clf = LogisticRegression(
    C=1.5,
    tol=0.01,
    random_state=0,
    multi_class="auto",
    solver='liblinear',
    max_iter=20000)

df = get_df(list(db.get_articles()))

for oversampler in [None, SMOTE(), ADASYN(), RandomOverSampler(random_state=0)]:
    models = [
        ('bg_lsa', ranlp_pipelines.make(clf, ['bg_lsa_title', 'bg_lsa_text'], oversampler=oversampler))
        #('bg_bert_title_text', ranlp_pipelines.make(clf, ['bg_bert_title', 'bg_bert_text'], oversampler=oversampler)),
#        ('bg_xlm_title_text', ranlp_pipelines.make(clf, oversampler, ['bg_xlm_title', 'bg_xlm_text', 'bg_xlm_cos'])),
#        ('bg_styl_title_text', ranlp_pipelines.make(clf, oversampler, ['bg_styl_title', 'bg_styl_text'])),
#        ('meta_media', ranlp_pipelines.make(clf, oversampler, ['meta_media'])),
#        ('en_use_title_text', ranlp_pipelines.make(clf, oversampler, ['en_use_title', 'en_use_text', 'en_use_cos'])),
#        ('en_nela_title_text', ranlp_pipelines.make(clf, oversampler, ['en_nela_title', 'en_nela_text', 'en_nela_cos'])),
#        ('en_bert_title_text', ranlp_pipelines.make(clf, oversampler, ['en_bert_title', 'en_bert_text', 'en_bert_cos'])),
#        ('en_elmo_title_text', ranlp_pipelines.make(clf, oversampler, ['en_elmo_title', 'en_elmo_text', 'en_elmo_cos'])),
    ]
    print(f'Oversampler: { oversampler}')
    compare_classifiers(models, df, df['label'], silent=False, plot=False)
