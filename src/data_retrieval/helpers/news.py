from urllib import urlparse
from newspaper import Article


def get_host(url):
    parsed_uri = urlparse(url)

    return parsed_uri.netloc


def download_article(url):
    article = Article(url)
    article.download()
    article.parse()

    article.nlp()

    return {
        'origin': get_host(url),
        'title': article.title,
        'text': article.text,
        'authors': article.authors,
        'publish_date': article.publish_date,
        'tags': list(article.tags),
        'keywords': article.keywords,
        'summary': article.summary
    }
