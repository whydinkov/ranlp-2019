from helpers import markup, media_eye, database, new

url = 'https://mediascan.gadjokov.com/?page='

db = database.MongoDB()


has_results = True
page_number = 1
results = []
added_links = []
while has_results:
    current_page = markup.get_page(f'{url}{page_number}')
    current_page_results = list(media_eye.get_results(current_page))

    for page_result in current_page_results:
        if not any(page_result)
        for t in page_result['tags']:
            if t == 'анонимен':
                continue  # don't support this tag, as it's not toxic

            if 'google' in t['link']:
                continue  # not an actual article, but an pdf document

            link = t['link']

            t['media_info'] = page_result['media_info']

            if link in added_links:
                continue

            # current link is not an article, but direct url to newspaper
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

    print(f'Processed page: {page_number}')
    has_results = len(current_page_results) > 0
    page_number += 1


results_len = len(results)
for index, r in enumerate(results):
    try:
        current_article = news.download_article(r['link'])
        current_article['link'] = r['link']
        current_article['label'] = r['tag']
        current_article['media_info'] = r['media_info']

        db.save_article(current_article)

        print(f'{index + 1}/{results_len}')
    except Exception as e:
        print(e)
        pass
