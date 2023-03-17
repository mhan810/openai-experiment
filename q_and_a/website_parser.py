import logging
import os
import re as regex
import requests
from bs4 import BeautifulSoup
from collections import deque
from typing import Self
from urllib.parse import urlparse

HTTP_URL_PATTERN = r'^http[s]*://.+'


class Webpage:
    links = []
    output: str

    def __init__(self, url: str):
        self.url = url
        self.local_domain = urlparse(url).netloc

    def process_link(self, link: str):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if regex.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == self.local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith('/'):
                link = link[1:]
            elif link.startswith('#') or link.startswith('mailto:'):
                return
            clean_link = 'https://' + self.local_domain + '/' + link

        if clean_link is not None:
            if clean_link.endswith('/'):
                clean_link = clean_link[: -1]
            if 'sitemap' not in clean_link:
                self.links.append(clean_link)

    def parse(self) -> Self:
        # connect to url and retrieve content
        try:
            response = requests.get(self.url)
            self.status_code = response.status_code

            if self.status_code != 200:
                return self

            soup = BeautifulSoup(response.text, 'html.parser')

            # grab links save them for future processing. delete them from tree
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None:
                    self.process_link(href)
                link.extract()

            # remove scripts from tree
            for script in soup(['script', 'style']):
                script.extract()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split('  '))

            self.output = '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            print(e)

        return self


class WebsiteParser:

    file_names = []

    def __init__(self, output_path: str, valid_paths=['product', 'about', 'services', 'partners'],
                 max_pages=20):
        self.output_path = output_path
        self.max_pages = max_pages
        self.valid_paths = valid_paths

    def parse(self, url: str) -> None:
        self.local_domain = urlparse(url).netloc

        os.makedirs(self.output_path + '/text/' +
                    self.local_domain, exist_ok=True)

        queue = deque([url])
        completed = set([url])

        while (len(queue) != 0) and (self.max_pages > 0):
            url = queue.pop()

            logging.info('Processing: ' + url)

            completed.add(url)

            webpage = Webpage(url).parse()

            if (webpage.status_code != 200):
                continue

            self.max_pages -= 1

            file_name = self.output_path + '/text/' + self.local_domain + \
                '/' + url[8:].replace('/', '_') + '.txt'

            self.file_names.append(file_name)

            with open(file_name, 'w', encoding='UTF-8') as file:
                file.write(webpage.output)

            for link in webpage.links:
                if (link not in completed):
                    for path in self.valid_paths:
                        if (path in link):
                            queue.append(link)
                        else:
                            logging.debug('Skipping ' + link)
