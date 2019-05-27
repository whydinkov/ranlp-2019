import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from os import environ
from dotenv import load_dotenv

load_dotenv()

module_url = "https://tfhub.dev/google/elmo/2"

elmo = hub.Module(module_url)

def __tokenize(sentence):
    parts = sentence.lower().split(' ')

    return ' '.join([p.strip() for p in parts][:300])

def generate_elmo_embeddings(inputs):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])

        tokenized = [__tokenize(i) for i in inputs]

        embeddings_tensor = elmo(
            tokenized,
            signature="default",
            as_dict=True)["elmo"]

        results = []
        for embedding in session.run(embeddings_tensor):
            results.append(np.mean(embedding, axis=0).tolist())

        return results

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

        feat_results = generate_elmo_embeddings(data)

        for i, article in enumerate(articles_batch):
            title_feature = feat_results[i * 2]
            text_feature  = feat_results[i * 2 + 1]

            if 'en' not in article['features']:
                article['features']['en'] = {}

            article['features']['en']['ELMO'] = {
                'title': title_feature,
                'text': text_feature
            }

            results.append(article)

        print(f'done batch {i_batch}')
        
    with open(articles_path, 'wb') as f:
        pickle.dump(results, f)