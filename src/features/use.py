import tensorflow as tf
import tensorflow_hub as hub
import pickle
from os import environ
from dotenv import load_dotenv

load_dotenv()

embed = hub.Module(module_url)

def __tokenize(sentence):
    parts = sentence.lower().split(' ')

    return ' '.join([p.strip() for p in parts][:300])

def get_use_embeddings(inputs):
    with tf.Session() as session:
        tokenized = [__tokenize(i) for i in inputs]

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embed(tokenized))


def batches(l, n=10):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

if __name__ == '__main__':
    articles_path = environ['articles_path']

    with open(articles_path, 'rb') as f:
        articles = pickle.load(f)
    
    results = []
    for i_batch, articles_batch in enumerate(batches(articles, 30)):
        data = []

        for article in articles_batch:
            data.append(article['translation']['title'])
            data.append(article['translation']['text'])

        feat_results = get_use_embeddings(data)

        for i, article in enumerate(articles_batch):
            title_feature = feat_results[i * 2]
            text_feature  = feat_results[i * 2 + 1]

            if 'en' not in article['features']:
                article['features']['en'] = {}

            article['features']['en']['USE'] = {
                'title': title_feature,
                'text': text_feature
            }

            results.append(article)

        print(f'done batch {i_batch}')
        
    with open(articles_path, 'wb') as f:
        pickle.dump(results, f)
