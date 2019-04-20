# imports, config
import pandas as pd

from src.data_retrieval.helpers import database
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout

# data
db = database.MongoDB()
df = get_df(list(db.get_articles()))

# models
baseline = pipelines.make_baseline()
lr = pipelines.make_best_lr()


def create_model():
    model = Sequential()

    # model arch
    model.add(Dense(32, activation='relu', input_dim=330))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

nn_clf = KerasClassifier(build_fn=create_model)
nn_model = pipelines.make(, pipeline_options)

# evaluation
models = [
    ('baseline', baseline),
    ('lr', lr),
    ('nn', nn_model)
]

compare_classifiers(models, df, df['label'], silent=False, plot=True)
