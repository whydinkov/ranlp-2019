import pandas as pd


def get_df(articles, transformation_options={
    'bert_title': 'REDUCE_MAX',
    'bert_text': 'CLS_TOKEN'
}):
    results = []

    bert_title = transformation_options['bert_title']
    bert_text = transformation_options['bert_text']

    for article in articles:
        results.append([
            article['title'],
            article['text'],
            {
                'title': article['title'],
                'text': article['text']
            },
            article['features']['bg']['BERT']['title'][bert_title],
            article['features']['bg']['BERT']['text'][bert_text],
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
