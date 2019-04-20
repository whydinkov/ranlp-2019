import pickle
from os import environ
from src.data_retrieval.helpers import constant


def get_articles():
    if constant.DB_FILE not in environ:
        raise Exception('Missing DB_FILE')

    with open(constant.DB_FILE, 'rb') as f:
        return pickle.load(f)
