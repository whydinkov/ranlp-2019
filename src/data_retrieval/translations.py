from helpers import database
from googletrans import Translator
import time

db = database.MongoDB()

articles = list(db.get_articles())


def _split_by_n(seq, n):
    while seq:
        yield seq[:n]
        seq = seq[n:]


def _translate(input):
    translator = Translator()

    parts = _split_by_n(input, 1000)
    translated_parts = []

    for p in parts:
        translated_part = translator.translate(p, src='bg', dest='en').text
        translated_parts.append(translated_part)

    return "".join(translated_parts)

if __name__ == '__main__':
    for index, article in enumerate(articles):
        try:
            print(f"{index + 1} / {len(articles)}")

            if 'translation' in article:
                continue

            article['translation'] = {
                'title': _translate(article['title']),
                'text': _translate(article['text'])
            }

            db.save_article(article)
            time.sleep(5)
        except Exception as e:
            print(e)
            continue
