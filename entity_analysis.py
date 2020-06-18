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

with jsonlines.open('data/output.jsonl') as reader:
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
print(articles_df)
print(entities_df)

avg_score = calculate_average_score(entities_df)
avg_no_entities = len(entities_df.index) / len(articles_df.index)
print('Average score: ', avg_score)
print('Average number of entities per article: ', avg_no_entities)

unique_entities = pd.DataFrame(entities_df.groupby(['word'])['article_id'].nunique())
unique_entities = unique_entities.sort_values(by=['article_id'], ascending=False)
print(unique_entities.reset_index())
count = pd.DataFrame(unique_entities.groupby('article_id').size())
print(count)
#hist = count.reset_index().hist(bins=5)
#plt.show()

tot_score = 0
no_scores = 0
ent_types = ['PER', 'ORG', 'LOC', 'TME', 'MSR', 'WRK', 'EVN', 'OBJ']
for ent_type in ent_types:
    filtered = entities_df.loc[entities_df['entity'] == ent_type]
    share = len(filtered) / len(entities_df.index)
    avg_score = calculate_average_score(filtered)
    print(ent_type, ': share =', share, '| average score =', avg_score)