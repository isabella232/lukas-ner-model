''' Read wikidata article '''
import sys
import re
import pandas
import json
import re
import numpy as np
from bs4 import BeautifulSoup

#import matplotlib
#matplotlib.use('agg')


def split_by_special_token(article):
    str_tokens = re.sub(r'[^a-zåäöÅÄÖ\- ]', ' ', article['body'].lower())
    tokens = ' '.join(w for w in str_tokens.split(' ') if w != '')
    article['body'] = tokens
    return article


def remove_stop_words(article):
    stop_words = open('stopwords.txt', 'r').read().split('\n')
    for word in stop_words:
        article['body'] = article['body'].replace(' ' + word + ' ', ' ')
    return article


def ngram(article, g=3):
    tokens = article['body'].split(' ')
    res = []
    for n in range(1, min(len(tokens), g + 1), 1):
        sub_tokens = []
        for i in range(0, len(tokens) - n + 1, 1):
            sub_tokens.append(' '.join(tokens[i:i+n]))
        res.append(sub_tokens)
    return res

def get_articles(articles_file):
    with open(articles_file, 'r') as f_handle:
        articles_arr = []
        regexp = re.compile('\\[.*?=\".*?\"\\]')
        for article_str in f_handle:
            article = json.loads(article_str)
            text = BeautifulSoup(article.get('body', '')).get_text().replace('\n', '. ').strip()
            article['text'] = ' '.join(regexp.split(text))
            articles_arr.append(article)
    return articles_arr

def print_ngrams(articles):
    for article in articles:
        article_special = split_by_special_token(article)
        article_no_stop_words = remove_stop_words(article_special)
        article_ngram = ngram(article_no_stop_words, g=10)
        print(article_ngram)

def print_article_body_info(articles):
    article_lengths = articles['body'].apply(lambda x: len(x))
    print(article_lengths.mean())
    print(article_lengths.max())
    print(article_lengths.min())
    max_length = article_lengths.max()
    print(articles.iloc[article_lengths.idxmax()])
    print(articles.iloc[article_lengths.idxmin()])
    print(articles.iloc[300]['body'])
    frequency, bins = np.histogram(article_lengths, bins=300)

    print(articles)
    for i, (b, f) in enumerate(zip(bins[1:], frequency)):
        if f == 1:
            print(round(b, 1), '1/one')
        else:
            print(round(b, 1), ' '.join(np.repeat('*', 20 * f / max(frequency))))
        # print(round(b, 1), ' '.join(np.repeat('*', f)))


def main(articles_file):
    articles = pandas.DataFrame(get_articles(articles_file))

if __name__ == '__main__':
    args = ['data/articles.csv']
    args[0] = sys.argv[1] or args[0]
    main(args[0])
