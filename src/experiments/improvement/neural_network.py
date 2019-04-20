# imports, config
import pandas as pd
import numpy as np

from src.data_retrieval.helpers import in_memory
from src.classifier.sklearn import pipelines
from src.evaluation.compare import compare_classifiers
from src.preprocessing.transformator import get_df

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data
df = get_df(list(in_memory.get_articles()))

# models
baseline = pipelines.make_baseline()
lr = pipelines.make_best_lr()


def create_model():
    model = Sequential()

    # model arch
    model.add(Dense(32, activation='relu', input_dim=230))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


nn_clf = KerasClassifier(build_fn=create_model, verbose=0)
nn_model = pipelines.make(nn_clf)

skf = StratifiedKFold(n_splits=5)
scores = []
for train_index, test_index in skf.split(df, df['label']):
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]

    clf = pipelines.make(nn_clf)

    clf.fit(X_train, X_train['label'])

    y_pred = clf.predict(X_test)

    current_acc = accuracy_score(X_test['label'], y_pred)
    scores.append(current_acc)

print('NN acc: ', np.average(scores))
