from transformers import pipeline
from parse_articles import get_articles
import re
nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
articles = get_articles('data/articles.csv')
for article in articles:
    sentences = article['text'].split('.')
    article_entities = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(sentence) > 3000:
            print(sentence)
        entities = nlp(sentence.strip() + '.')
        article_entities += entities
    print('-' * 50)
    print(article['text'])
    print(article_entities)
