from src.data_retrieval.helpers import database
from bert_serving.client import BertClient

_client = BertClient(timeout=5000, output_fmt='list', check_length=False)


def _get_bert(text):
    return _client.encode([text])[0]


def main():
    pooling = 'SEP_TOKEN'  # CLS_TOKEN, REDUCE_MEAN,
    # REDUCE_MAX, REDUCE_MEAN_MAX, SEP_TOKEN - should be same as in server

    db = database.MongoDB()

    articles = list(db.get_articles())
    for article in articles:
        try:
            title = _get_bert(article['title'])
            text = _get_bert(article['text'])

            article['features']['bg']['BERT']['title'][pooling] = title
            article['features']['bg']['BERT']['text'][pooling] = text

            db.save_article(article)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
