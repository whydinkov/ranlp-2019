import pandas as pd


def get_df(articles):
    results = []

    for article in articles:
        results.append([
            article['title'],
            article['text'],
            {
                'title': article['title'],
                'text': article['text']
            },
            article['features']['bg']['BERT']['title']['REDUCE_MEAN'],
            article['features']['bg']['BERT']['text']['REDUCE_MEAN'],
            article['media_info'],
            article['label']
        ])

    return pd.DataFrame(results, columns=[
        'title',
        'text',
        'article',
        'bert_title',
        'bert_text',
        'media',
        'label'
    ])
