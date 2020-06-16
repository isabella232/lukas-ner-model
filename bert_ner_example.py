from transformers import pipeline
from parse_articles import get_articles
import re
import json
import jsonlines

nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
articles = get_articles('data/small.json')
article_cnt = 0
json_output = []
failed_articles = []

for article in articles:
    sentences = article['content_text'].split('.')
    article_entities = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(sentence) > 3000:
            print(sentence)
            
        try:
            entities = nlp(sentence.strip() + '.')
            article_entities += entities
        except IndexError:
            # 1541 max length sentence
            failed_articles += [ article['content_text'] ]

    formatted_entities = []
    no_subtokens = 1
    for token in article_entities:
        if no_subtokens > 1 and not token['word'].startswith('##'):
            formatted_entities[-1]['score'][-1] = formatted_entities[-1]['score'][-1]/no_subtokens
            no_subtokens = 1

        if formatted_entities and formatted_entities[-1]['entity'] == token['entity'] and formatted_entities[-1]['index'] == token['index'] - 1:
            formatted_entities[-1]['index'] = token['index']
            if token['word'].startswith('##'):
                no_subtokens += 1
                formatted_entities[-1]['word'] += token['word'][2:]
                formatted_entities[-1]['score'][-1] += token['score']
            else:
                formatted_entities[-1]['word'] += ' ' + token['word']
                formatted_entities[-1]['score'] += [token['score']]
        else:
            token['score'] = [token['score']]
            formatted_entities += [token]
    
    json_output += {'article': article, 'entities': formatted_entities},
    article_cnt += 1
    print(article_cnt)


with jsonlines.open('data/output.jsonl', mode='w') as writer:
    for article in json_output:
        writer.write(article)

with jsonlines.open('data/failed.jsonl', mode='w') as writer:
    for article in failed_articles:
        writer.write(article)

# TODO: hist för antal entiteter/artikel (baserat på artikellängd) samt för mest frekventa entiteter/(topp)kategori, 17 st