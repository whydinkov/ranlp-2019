from urllib.parse import urljoin
from .markup import get_page

__MEDIASCAN_BASE_URL = 'https://mediascan.gadjokov.com/'


def set_tag(tag):
    if tag.endswith('\n'):
        return tag[:-1]

    return tag


def get_media_info(media_link_node):
    url = urljoin(__MEDIASCAN_BASE_URL, media_link_node['href'])
    page = get_page(url)

    rs = page.find_all(class_='single-view-cell-other')

    info = {
        "editor": rs[0].getText().strip(),
        "responsible_person": rs[1].getText().strip(),
        "established": rs[2].getText().strip(),
        "popularity": rs[3].getText().strip(),
        "server_in": rs[4].getText().strip(),
        "domain_owner": rs[5].getText().strip(),
        "domain_responsible_person": rs[6].getText().strip()
    }

    return info


def get_results(page):
    table = page.find('table', class_='differentTable')
    body = table.find('tbody')
    rows = body.find_all('tr')

    for row in rows:
        media_link = row.find('a')
        article_links = row.find_all('a', {'data-toggle': 'tooltip'})

        yield {
            'link': media_link.getText().strip(),
            'media_info': get_media_info(media_link),
            'tags': [
                {
                    'tag': tag.getText().strip(),
                    'link': set_tag(tag['href'])
                }
                for tag
                in article_links]
        }
