from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import DictVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential


def get_column(column_name):
    return FunctionTransformer(lambda x: x[column_name].tolist(),
                               validate=False)


def _pipe_column(name):
    return (name, Pipeline([
        ('selector', get_column(name))
    ]))


_meta_media = ('meta_media', FeatureUnion([
    ('category', Pipeline([
        ('selector', get_column('media_cat')),
        ('vect', DictVectorizer())
    ])),
    ('numerical', Pipeline([
        ('selector', get_column('media_num')),
        ('vect', DictVectorizer()),
        ('norm', Normalizer()),
    ]))
]))


def make(classifier, columns):
    feat_pipes = []

    for column in columns:
        if column == 'meta_media':
            feat_pipes.append(_meta_media)
        else:
            feat_pipes.append(_pipe_column(column))

    return Pipeline([
        ('feats', FeatureUnion(feat_pipes)),
        ('clf', classifier)
    ])


def _mlp_arch(input_dim):
    model = Sequential()

    # model arch
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.4))
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def make_nn(dim, columns):
    feat_pipes = []

    for column in columns:
        if column == 'meta_media':
            feat_pipes.append(_meta_media)
        else:
            feat_pipes.append(_pipe_column(column))

    classifier = KerasClassifier(build_fn=_mlp_arch,
                                 epochs=50,
                                 batch_size=12,
                                 verbose=0,
                                 input_dim=dim)

    return Pipeline([
        ('feats', FeatureUnion(feat_pipes)),
        ('clf', classifier)
    ])
