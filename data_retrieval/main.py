from helpers import markup, media_eye
from helpers.db import MongoDB
from dotenv import load_dotenv

url = 'https://mediascan.gadjokov.com/?page='

load_dotenv()
db = MongoDB()


def download_article(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()

    return {
        'title': article.title,
        'text': article.text,
        'authors': article.authors,
        'publish_date': article.publish_date,
        'tags': list(article.tags)
    }


has_results = True
page_number = 1
results = []
added_links = []
while has_results:
    current_page = markup.get_page(f'{url}{page_number}')
    current_page_results = list(get_results(current_page))

    for page_result in current_page_results:
        for t in page_result['tags']:
            if 'google' in t['link']:
                continue

            link = t['link']

            if link in added_links:
                continue

            if link.endswith(('.bg/', '.com/', '.eu/', '.net/', '.org/',
                              '.bg', '.com', '.eu', '.net', '.org',
                              '.info', '.info/', '.cc', '.cc/', '.бг', '.бг/',
                              '.host', '.host/')):
                continue

            if 'xn--e1atdf' in link:
                continue

            if 'webnovinar' in link:
                continue

            if 'tangranews' in link:
                continue

            if 'bpost.bg' in link:
                continue

            if 'dailypress.bg' in link:
                continue

            if 'fakto.bg' in link:
                continue

            added_links.append(link)

            results.append(t)

    has_results = len(current_page_results) > 0
    page_number += 1
    print(f'Processed page: {page_number}')

articles = []
results_len = len(results)
for index, r in enumerate(
        results):
    try:
        current_article = download_article(r['link'])
        current_article['link'] = r['link']
        current_article['label'] = r['tag']

        articles.append(current_article)
        print(f'{index}/{results_len}')
    except:
        pass

for article in articles:
    try:
        db.save_article(media)
    except Exception as ex:
        print(ex)
        pass
