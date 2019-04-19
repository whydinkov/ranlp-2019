# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from keras.wrappers.scikit_learn import KerasClassifier

db = database.MongoDB()

df = get_df(list(db.get_articles()))

pipeline_options = {
    'lsa_text': 1,
    'lsa_title': 1,
    'bert_text': 0,
    'bert_title': 0,
    'meta_article': 0,
    'meta_media': 0
}

# models
baseline = pipelines.make(DummyClassifier(
    random_state=0,
    strategy="most_frequent"), pipeline_options)

lr = pipelines.make(LogisticRegression(
    random_state=0,
    multi_class="auto",
    solver='lbfgs'), pipeline_options)


def create_model():
    model = Sequential()

    # model arch
    # model.add(Dense())
    # model.add(Dropout())

    model.compile(optimizer='',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


nn_model = pipelines.make(, pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', lr),
    ('nn', nn_model)
]

compare_classifiers(models, df, df['label'], silent=False, plot=True)
