# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df, oversample

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# data
db = database.MongoDB()
df = get_df(list(db.get_articles()))

unified_df = oversample(df, group_size=125)
oversampled_df = oversample(df, preserve_distribution=True)

# models
baseline = pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), pipeline_options)

lr = pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    C=10,
    penality='l1',
    tol=1e-10,
    solver='saga'), pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', lr)
]

print('default:')
compare_classifiers(models, df, df['label'], silent=False)
print('oversampled:')
compare_classifiers(models, oversampled_df, oversampled_df['label'],
                    silent=False)
print('oversampled - unified groups:')
compare_classifiers(models, unified_df, unified_df['label'], silent=False)
