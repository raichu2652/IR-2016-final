#!/usr/bin/env python3
from bs4 import BeautifulSoup
import argparse
import requests
import urllib.request

def searchImages(query, numImages=100, outputDir='./images/', outputFileFormat='.jpg'):
    URL_GOOGLE = 'https://www.google.com/'
    html = requests.get(URL_GOOGLE + 'search?hl=en&tbm=isch' + '&q=' + query).text
    soup = BeautifulSoup(html, 'html.parser')
#    print soup.prettify().encode('utf-8')
#    print soup.find_all('a', string='Next')
    
    nextPage = soup.find('span', string='Next').parent.get('href')
    imageURLs = []
    while len(imageURLs) < numImages:
        imageURLs.extend([img.get('src') for img in soup.find_all('img')])

        html = requests.get(URL_GOOGLE + nextPage).text
        soup = BeautifulSoup(html, 'html.parser')
        span = soup.find('span', string='Next')
        if not span == None:
            nextPage = span.parent.get('href')

        print (len(imageURLs), numImages)
#        print nextPage

    for i, url in enumerate(imageURLs):
        print ('downloading[' + str(i) + '] '  + url)
        urllib.request.urlretrieve(url, outputDir + 'image' + str(i) + outputFileFormat)

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', default='Information Retrieval')
parser.add_argument('-n', '--num-results', default='100')
parser.add_argument('-o', '--directory', default='./images/')
options = parser.parse_args()

searchImages(options.query, int(options.num_results), options.directory)
