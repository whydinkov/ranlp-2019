import requests
from bs4 import BeautifulSoup
from urllib import parse
import json
from . import constant
from os import environ

__USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\
               (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'


def get_page(base_url, page_url=''):
    try:
        url = parse.urljoin(base_url, page_url)
        markup = requests.get(url,
                              headers={'User-Agent': __USER_AGENT},
                              timeout=10).text
        return BeautifulSoup(markup, 'html.parser')
    except Exception as err:
        print(err)
        return None
