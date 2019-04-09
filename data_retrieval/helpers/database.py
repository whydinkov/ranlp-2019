from pymongo import MongoClient
from bson.objectid import ObjectId
from os import environ
from . import constant


class MongoDB:
    def __init__(self):
        if constant.MONGO_DB not in environ:
            error_msg = (f"{constant.MONGO_DB} environment variable missing."
                         "Cannot establish connection.")
            raise KeyError(error_msg)

        if constant.DB_NAME not in environ:
            error_msg = (f"{constant.DB_NAME} environment variable. missing"
                         "Cannot establish connection.")

        client = MongoClient(environ[constant.MONGO_DB])
        database = client[environ[constant.DB_NAME]]

        self.articles_collection = database.articles

    def save_article(self, article):
        query_filter = {
            'link': article['link']
        }

        self.__add_or_update(self.articles_collection, query_filter, article)

    def get_articles(self, options={}):
        return self.articles_collection.find(options)

    def __add_or_update(self, collection, query_filter, entry):
        result = collection.find_one_and_replace(query_filter, entry)

        if not result:
            collection.insert_one(entry)
