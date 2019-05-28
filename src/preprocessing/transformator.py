import pandas as pd
from numpy import dot
from numpy.linalg import norm
from src.features.stylometry import get_stats_text, get_stats_title
from src.features.media import get_stats_cat, get_stats_num


def _cos_sim(a, b):
    return [dot(a, b)/(norm(a)*norm(b))]


def get_df(articles, transformation_options={
    'bert_title': 'REDUCE_MAX',
    'bert_text': 'CLS_TOKEN'
}):
    results = []

    bert_title = transformation_options['bert_title']
    bert_text = transformation_options['bert_text']

    for article in articles:
        bg_bert_title = article['features']['bg']['BERT']['title'][bert_title]
        bg_bert_text = article['features']['bg']['BERT']['text'][bert_text]
        bg_bert_cos = _cos_sim(bg_bert_title, bg_bert_text)

        bg_xlm_title = article['features']['bg']['XLM']['title']
        bg_xlm_text = article['features']['bg']['XLM']['text']
        bg_xlm_cos = _cos_sim(bg_xlm_title, bg_xlm_text)

        en_use_title = article['features']['en']['USE']['title']
        en_use_text = article['features']['en']['USE']['text']
        en_use_cos = _cos_sim(en_use_title, en_use_text)

        en_bert_title = article['features']['en']['BERT']['title'][bert_title]
        en_bert_text = article['features']['en']['BERT']['text'][bert_text]
        en_bert_cos = _cos_sim(en_bert_title, en_bert_text)

        en_nela_title = article['features']['en']['NELA']['title']
        en_nela_text = article['features']['en']['NELA']['text']
        en_nela_cos = _cos_sim(en_nela_title, en_nela_text)

        en_elmo_title = article['features']['en']['ELMO']['title']
        en_elmo_text = article['features']['en']['ELMO']['text']
        en_elmo_cos = _cos_sim(en_elmo_title, en_elmo_text)

        bg_styl_title = list(get_stats_title(article['title']).values())
        bg_styl_text = list(get_stats_text(article['text']).values())

        media_cat = get_stats_cat(article['media_info'])
        media_num = get_stats_num(article['media_info'])

        bg_bert_title_pred = article['predictions']['bg_bert_title']
        bg_bert_text_pred = article['predictions']['bg_bert_text']
        bg_xlm_title_pred = article['predictions']['bg_xlm_title']
        bg_xlm_text_pred = article['predictions']['bg_xlm_text']
        meta_media_pred = article['predictions']['meta_media']
        bg_styl_title_pred = article['predictions']['bg_styl_title']
        bg_styl_text_pred = article['predictions']['bg_styl_text']
        bg_lsa_title_pred = article['predictions']['bg_lsa_title']
        bg_lsa_text_pred = article['predictions']['bg_lsa_text']
        en_use_title_pred = article['predictions']['en_use_title']
        en_use_text_pred = article['predictions']['en_use_text']
        en_nela_title_pred = article['predictions']['en_nela_title']
        en_nela_text_pred = article['predictions']['en_nela_text']
        en_bert_title_pred = article['predictions']['en_bert_title']
        en_bert_text_pred = article['predictions']['en_bert_text']
        en_elmo_title_pred = article['predictions']['en_elmo_title']
        en_elmo_text_pred = article['predictions']['en_elmo_text']

        results.append([
            article['title'],
            article['text'],
            {
                'title': article['title'],
                'text': article['text']
            },
            bg_bert_title,
            bg_bert_text,
            bg_bert_cos,
            bg_xlm_title,
            bg_xlm_text,
            bg_xlm_cos,
            bg_styl_title,
            bg_styl_text,
            en_use_title,
            en_use_text,
            en_use_cos,
            en_nela_title,
            en_nela_text,
            en_nela_cos,
            en_elmo_title,
            en_elmo_text,
            en_elmo_cos,
            en_bert_title,
            en_bert_text,
            en_bert_cos,
            media_cat,
            media_num,
            bg_bert_title_pred,
            bg_bert_text_pred,
            bg_xlm_title_pred,
            bg_xlm_text_pred,
            meta_media_pred,
            bg_styl_title_pred,
            bg_styl_text_pred,
            bg_lsa_title_pred,
            bg_lsa_text_pred,
            en_use_title_pred,
            en_use_text_pred,
            en_nela_title_pred,
            en_nela_text_pred,
            en_bert_title_pred,
            en_bert_text_pred,
            en_elmo_title_pred,
            en_elmo_text_pred,
            article['media_info'],
            article['label']
        ])

    return pd.DataFrame(results, columns=[
        'title',
        'text',
        'article',
        'bg_bert_title',
        'bg_bert_text',
        'bg_bert_cos',
        'bg_xlm_title',
        'bg_xlm_text',
        'bg_xlm_cos',
        'bg_styl_title',
        'bg_styl_text',
        'en_use_title',
        'en_use_text',
        'en_use_cos',
        'en_nela_title',
        'en_nela_text',
        'en_nela_cos',
        'en_elmo_title',
        'en_elmo_text',
        'en_elmo_cos',
        'en_bert_title',
        'en_bert_text',
        'en_bert_cos',
        'media_cat',
        'media_num',
        'bg_bert_title_pred',
        'bg_bert_text_pred',
        'bg_xlm_title_pred',
        'bg_xlm_text_pred',
        'meta_media_pred',
        'bg_styl_title_pred',
        'bg_styl_text_pred',
        'bg_lsa_title_pred',
        'bg_lsa_text_pred',
        'en_use_title_pred',
        'en_use_text_pred',
        'en_nela_title_pred',
        'en_nela_text_pred',
        'en_bert_title_pred',
        'en_bert_text_pred',
        'en_elmo_title_pred',
        'en_elmo_text_pred',
        'media',
        'label'
    ])


def oversample(df, n=None, frac=None):
    labels = df['label'].unique()

    oversampled = None

    for label in labels:
        if frac:
            samples = df[df['label'] == label].sample(frac=frac,
                                                      random_state=0,
                                                      replace=True)
        elif n:
            samples = df[df['label'] == label].sample(n=n,
                                                      random_state=0,
                                                      replace=True)
        else:
            raise 'n/frac is required.'

        if oversampled is None:
            oversampled = samples
        else:
            oversampled = oversampled.append(samples)

    return oversampled.sample(frac=1).reset_index(drop=True)
