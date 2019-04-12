# imports, config
import pandas as pd
from numpy.random import seed
from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC

seed(0)

db = database.MongoDB()

# data retrieval
df = pd.DataFrame(list(db.get_articles()))

# preprocesing
data = df['text'] + df['title']
# data= df['text'].apply(lambda x: ' text_'.join(x.split(' '))) + \
#     df['title'].apply(lambda x: ' title_'.join(x.split(' ')))
y = df['label']

# models
baseline = pipelines.make(DummyClassifier(strategy="most_frequent"))
svc = pipelines.make(LinearSVC())
nb = pipelines.make(BernoulliNB())
lr = pipelines.make(LogisticRegression(multi_class="auto", solver='lbfgs'))

# evaluation
models = [
    ('baseline', baseline),
    ('svc', svc),
    ('nb', nb),
    ('lr', lr)
]

compare_classifiers(models, data, y, silent=False)
