def set_tag(tag):
    if tag.endswith('\n'):
        return tag[:-1]

    return tag


def get_results(page):
    table = page.find('table', class_='differentTable')
    body = table.find('tbody')
    rows = body.find_all('tr')

    for row in rows:
        yield {
            'link': row.find('a').getText().strip(),
            'tags': [{
                'tag': tag.getText().strip(),
                'link': set_tag(tag['href'])
            }
                for tag
                in row.find_all('a', {'data-toggle': 'tooltip'})
                if tag.getText().strip() != 'анонимен']
        }
