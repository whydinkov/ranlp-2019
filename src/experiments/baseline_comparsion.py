# imports, config
import pandas as pd
from dotenv import load_dotenv
from data_retrieval.helpers import database
from classifier.sklearn import pipelines
from evaluation.compare import compare_classifiers

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

load_dotenv()
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
svc = pipelines.make(LinearSVC(random_state=0))
nb = pipelines.make(MultinomialNB(fit_prior=True, class_prior=None))
lr = pipelines.make(LogisticRegression(
    random_state=0,
    solver='sag', max_iter=200, multi_class='auto'))

# evaluation
models = [
    ('baseline', baseline),
    ('svc', svc),
    ('nb', nb),
    ('lr', lr)
]

compare_classifiers(models, data, y, silent=False)
