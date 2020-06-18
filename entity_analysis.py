import pandas as pd
import jsonlines
import numpy as np
import matplotlib.pyplot as plt

def calculate_average_score(df):
    tot_score = 0
    no_scores = 0
    for ind in df.index:
        scores = df['score'][ind]
        tot_score += sum(scores)
        no_scores += len(scores)
    return tot_score / no_scores

with jsonlines.open('data/results.jsonl') as reader:
    articles_list = []
    entities_list = []
    for obj in reader:
        articles_list += [obj['article']]
        article_id = obj['article']['id']
        for entity in obj['entities']:
            entities_list += [ entity ]
            entities_list[-1]['article_id'] = article_id


articles_df = pd.DataFrame(articles_list)
entities_df = pd.DataFrame(entities_list)
entities_df['word'] = entities_df['word'].str.lower()
ambigous_df = entities_df[entities_df['entity'].apply(lambda x: type(x) == list)]
ambigous_share = len(ambigous_df.index) / len(entities_df.index)
entities_df = entities_df[entities_df['entity'].apply(lambda x: type(x) != list)]
essentials_df = entities_df[(entities_df['entity'] == 'PER') | (entities_df['entity'] == 'ORG') | (entities_df['entity'] == 'LOC')]

print('-' * 200)
print('ARTICLES')
print(articles_df)
print('-' * 200)
print('ENTITIES')
print(entities_df)
print('-' * 200)
print('AMBIGOUS ENTITIES')
print(ambigous_df)
print('-' * 200)
print('ESSENTIAL ENTITIES')
print(essentials_df)
print('-' * 200)

tot_len = len(entities_df.index)
avg_score = calculate_average_score(entities_df)
avg_no_entities = tot_len / len(articles_df.index)
ambigous_score = calculate_average_score(ambigous_df)
ambigous_no = len(ambigous_df) / len(articles_df)
essentials_score = calculate_average_score(essentials_df)
essentials_no = len(essentials_df) / len(articles_df)
essentials_share = len(essentials_df.index) / tot_len

print('Total:\t\taverage score =', avg_score, '| average number of entities per article =', avg_no_entities)
print('Ambigous:\taverage score =', ambigous_score, '| average number of entities per article =', ambigous_no, '| share =', ambigous_share)
print('Essentials:\taverage score =', essentials_score, '| average number of entities per article =', essentials_no, '| share =', essentials_share)
print('-' * 200)

ent_types = ['PER', 'ORG', 'LOC', 'TME', 'MSR', 'WRK', 'EVN', 'OBJ']
for ent_type in ent_types:
    filtered = entities_df[entities_df['entity'] == ent_type]
    share = len(filtered) / len(entities_df.index)
    avg_score = calculate_average_score(filtered)
    print(ent_type, ': share =', share, '\t| average score =', avg_score)
print('-' * 200)

print('NUMBER OF OCURRENCES PER ENTITY')
unique_entities = pd.DataFrame(essentials_df.groupby(['word'])['article_id'].nunique()).reset_index()
unique_entities = unique_entities.rename(columns={'article_id': 'no_occurences'}).sort_values(by=['no_occurences'], ascending=False)
print(unique_entities.head(20))
count = pd.DataFrame(unique_entities.groupby('no_occurences').size()).reset_index().rename(columns={0: 'no_words'})
count = count[count['no_words'] > 1]
plt.plot(count['no_occurences'], count['no_words'])
#plt.show()

categories_list = []
for ind in articles_df.index:
    for tag in articles_df['tags'][ind]:
        if tag['category'].startswith('RYF-'):
            categories_list += [{'category': tag['category'], 'article_id': articles_df['id'][ind]}]
categories_df = pd.DataFrame(categories_list)
print(categories_df.groupby(['category']).sum())