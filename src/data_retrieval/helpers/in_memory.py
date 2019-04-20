import pickle
from os import environ
from src.data_retrieval.helpers import constant
from dotenv import load_dotenv

load_dotenv()


def get_articles():
    if constant.DB_FILE not in environ:
        raise Exception('Missing DB_FILE')

    with open(environ[constant.DB_FILE], 'rb') as f:
        return pickle.load(f)
