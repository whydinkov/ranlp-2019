# imports, config
import pandas as pd
import numpy as np

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.svm import LinearSVC

db = database.MongoDB()

bert_types = ['CLS_TOKEN',
              'REDUCE_MEAN',
              'REDUCE_MAX',
              'REDUCE_MEAN_MAX',
              'SEP_TOKEN']
keys = []
for bert_title in bert_types:
    for bert_text in bert_types:

        df = get_df(list(db.get_articles()), transformation_options={
            'bert_title': bert_title,
            'bert_text': bert_text
        })

        for p_options in [[0, 1], [1, 0], [1, 1]]:
            pipeline_options = {
                'lsa_text': 0,
                'lsa_title': 0,
                'bert_text': p_options[0],
                'bert_title': p_options[1],
                'meta_article': 0,
                'meta_media': 0
            }

            # models
            svc = pipelines.make(LinearSVC(random_state=0,
                                           max_iter=20000),
                                 pipeline_options)

            # evaluation
            models = [
                ('svc', svc),
            ]

            key = ""
            if p_options[0]:
                key += bert_title
            else:
                key += "NONE"
            if p_options[1]:
                key += f"\t{bert_text}"
            else:
                key += "\tNONE"
            if key in keys:
                continue

            acc = compare_classifiers(models, df, df['label'], silent=True)[0]

            keys.append(key)

            print(key, np.average(acc), flush=True)
