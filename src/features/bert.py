import pickle
from bert_serving.client import BertClient
from os import environ
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    bc = BertClient(timeout=5000, output_fmt='list')

    articles_path = environ['articles_path']
    with open(articles_path, 'rb') as f:
        articles = pickle.load(f)

    pooling = 'SEP_TOKEN'
    print(f'Generating for {pooling}')
    for iteration, article in enumerate(articles):
        title = article['translation']['title']
        text = article['translation']['text']

        if 'BERT' not in article['features']['en']:
            article['features']['en']['BERT'] = {'title': {}, 'text': {}}

        article['features']['en']['BERT']['title'][pooling] = bc.encode([title])[0]
        article['features']['en']['BERT']['text'][pooling] = bc.encode([text])[0]

    with open(articles_path, 'wb') as f:
        pickle.dump(articles, f)
