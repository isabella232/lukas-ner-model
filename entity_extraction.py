from transformers import pipeline
import re
import json
import jsonlines


def get_articles(articles_file):
    with open(articles_file, 'r') as f_handle:
        articles_arr = []
        articles = json.load(f_handle)
        for article in articles:
            articles_arr.append(article)
    return articles_arr


def format_entities(raw_entities):
    formatted_entities = []
    no_subtokens = 1
    
    for i, current in enumerate(raw_entities):
        if '[UNK]' in current['word']: current['word'] = current['word'].replace('[UNK]', '').strip()
        if not current['word']:
            current['entity'] = 'NA'
            continue

        if formatted_entities:
            previous = formatted_entities[-1]
            adjacent = previous['index'] == current['index'] - 1
            same_entity = previous['entity'] == current['entity']
        
        if formatted_entities and current['word'].startswith('##'):
            if adjacent:
                previous['index'] = current['index']
                previous['word'] += current['word'][2:]
                previous['score'][-1] += current['score']
                no_subtokens += 1
                if not same_entity:
                    if not isinstance(previous['entity'], list): previous['entity'] = [previous['entity']]
                    previous['entity'] += [{'type': current['entity'], 'subtoken': current['word']}]
            else:
                current['entity'] = 'NA'
        elif formatted_entities and adjacent and same_entity:
            previous['index'] = current['index']
            previous['word'] += ' ' + current['word']
            previous['score'] += [current['score']]
            no_subtokens = 1
        else:
            current['score'] = [current['score']]
            formatted_entities += [current.copy()]
            no_subtokens = 1

        end_of_article = i + 1 >= len(raw_entities)
        if not end_of_article:
            subseq = raw_entities[i + 1]
            end_of_sentence = subseq['index'] < current['index']
            end_of_entity_series = no_subtokens > 1 and not subseq['word'].startswith('##')

        if (end_of_article or end_of_sentence) or end_of_entity_series:
            formatted_entities[-1]['score'][-1] = formatted_entities[-1]['score'][-1] / no_subtokens
            no_subtokens = 1
    return formatted_entities
    

nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
articles = get_articles('data/small.json')

article_cnt = 0
json_output = []
failed_articles = []

for count, article in enumerate(articles):
    sentences = article['content_text'].split('.')
    article_entities = []
    for sentence in sentences:
        if re.search('<.*>', sentence):
            failed_articles += [article]
            break
        elif not sentence.strip():
            continue
        try:
            entities = nlp(sentence.strip() + '.')
            article_entities += entities
        except IndexError:  # 1541 max length sentence
            failed_articles += [article]
            continue

    formatted_entities = format_entities(article_entities)
    json_output += [{'article': article, 'entities': formatted_entities}]
    print(count + 1)

    for entity in formatted_entities:
        # if isinstance(entity['entity'], list): print('mixed:', entity)
        for score in entity['score']:
            if score > 1:
                print('failed:', entity)
                exit()

with jsonlines.open('data/results.jsonl', mode='w') as writer:
    for article in json_output:
        writer.write(article)

with jsonlines.open('data/failed.jsonl', mode='w') as writer:
    for article in failed_articles:
        writer.write(article)
