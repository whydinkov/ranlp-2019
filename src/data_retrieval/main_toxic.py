from helpers import markup, media_eye, database
from dotenv import load_dotenv
from urllib.parse import urlparse
from newspaper import Article

url = 'https://mediascan.gadjokov.com/?page='

load_dotenv()
db = database.MongoDB()


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


has_results = True
page_number = 1
results = []
added_links = []
while has_results:
    current_page = markup.get_page(f'{url}{page_number}')
    current_page_results = list(media_eye.get_results(current_page))

    for page_result in current_page_results:
        for t in page_result['tags']:
            if t == 'анонимен':
                continue

            if 'google' in t['link']:
                continue

            link = t['link']

            t['media_info'] = page_result['media_info']

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
for index, r in enumerate(results):
    try:
        current_article = download_article(r['link'])
        current_article['link'] = r['link']
        current_article['label'] = r['tag']
        current_article['media_info'] = r['media_info']

        db.save_article(current_article)

        print(f'{index}/{results_len}')
    except Exception as e:
        print(e)
        pass

    for page_result in current_page_results:
        if not any(page_result)
        for t in page_result['tags']:
            if 'google' in t['link']:
                print(t['link'])
                continue

            link = t['link']

            t['media_info'] = page_result['media_info']

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
    # print(f'Processed page: {page_number}')

# articles = []
# results_len = len(results)
# for index, r in enumerate(results):
#     try:
#         current_article = download_article(r['link'])
#         current_article['link'] = r['link']
#         current_article['label'] = r['tag']
#         current_article['media_info'] = r['media_info']

#         db.save_article(current_article)

#         print(f'{index}/{results_len}')
#     except Exception as e:
#         print(e)
#         pass
