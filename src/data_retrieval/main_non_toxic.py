from helpers import markup, media_eye, database, news
from urllib.parse import urlparse, urljoin
from newspaper import Article, build

url = 'https://mediascan.gadjokov.com/?page='

db = database.MongoDB()

ARTICLES_COUNT = 100
ARTICLES_PER_PAPER = 5
NON_TOXIC_LABEL = 'нетоксичен'

has_results = True
page_number = 1
results = []
added_links = []
while has_results:
    current_page = markup.get_page(f'{url}{page_number}')
    current_page_results = list(media_eye.get_results(current_page))

    for page_result in current_page_results:
        if any(page_result['tags']):
            continue  # contains at least one tag -> toxic

        current_paper = build(f"https://{page_result['link']}")

        for article_url in current_paper.article_urls()[:ARTICLES_PER_PAPER]:
            results.append({
                'link': article_url,
                'media_info': page_result['media_info']
            })

    if len(results) >= ARTICLES_COUNT:
        break

    print(f"Found articles links: {len(results)}")
    print(f'Processed page: {page_number}')
    has_results = len(current_page_results) > 0
    page_number += 1

results_len = len(results)
for index, r in enumerate(results):
    try:
        current_article = news.download_article(r['link'])
        current_article['link'] = r['link']
        current_article['label'] = NON_TOXIC_LABEL
        current_article['media_info'] = r['media_info']

        db.save_article(current_article)

        print(f'{index + 1}/{results_len}')
    except Exception as e:
        print(e)
        pass
