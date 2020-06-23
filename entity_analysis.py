import pandas as pd
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def create_data_frames():
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
    entities_df = entities_df[entities_df['word'] != 's']
    #entities_df['word'] = entities_df['word'].str.lower()
    ambiguous_df = entities_df[entities_df['entity'].apply(lambda x: type(x) == list)]
    entities_df = entities_df[entities_df['entity'].apply(lambda x: type(x) != list)]
    essentials_df = entities_df[entities_df['entity'].apply(lambda x: x in ['PER', 'ORG', 'LOC', 'EVN'])]

    return articles_df, entities_df, ambiguous_df, essentials_df


def calculate_average_score(df):
    tot_score = 0
    no_scores = 0
    for ind in df.index:
        scores = df['score'][ind]
        tot_score += sum(scores)
        no_scores += len(scores)
    return tot_score / no_scores


def initial_analysis(articles_df, entities_df, ambiguous_df, essentials_df):

    print('-' * 200)
    print('ARTICLES')
    print(articles_df)
    print('-' * 200)
    print('ENTITIES')
    print(entities_df)
    print('-' * 200)
    print('AMBIGUOUS ENTITIES')
    print(ambiguous_df)
    print('-' * 200)
    print('ESSENTIAL ENTITIES')
    print(essentials_df)
    print('-' * 200)

    tot_len = len(entities_df.index)
    avg_score = calculate_average_score(entities_df)
    avg_no_entities = tot_len / len(articles_df.index)
    ambiguous_score = calculate_average_score(ambiguous_df)
    ambiguous_no = len(ambiguous_df) / len(articles_df)
    ambiguous_share = len(ambiguous_df.index) / (len(ambiguous_df.index) + len(entities_df.index))
    essentials_score = calculate_average_score(essentials_df)
    essentials_no = len(essentials_df) / len(articles_df)
    essentials_share = len(essentials_df.index) / tot_len

    print('Total:\t\taverage score =', avg_score, '| average number of entities per article =', avg_no_entities)
    print('Ambiguous:\taverage score =', ambiguous_score, '| average number of entities per article =', ambiguous_no, '| share =', ambiguous_share)
    print('Essentials:\taverage score =', essentials_score, '| average number of entities per article =', essentials_no, '| share =', essentials_share)
    print('-' * 200)

    ent_types = ['PER', 'ORG', 'LOC', 'TME', 'MSR', 'WRK', 'EVN', 'OBJ']
    for ent_type in ent_types:
        filtered = entities_df[entities_df['entity'] == ent_type]
        share = len(filtered) / len(entities_df.index)
        avg_score = calculate_average_score(filtered)
        print(ent_type, ': share =', share, '\t| average score =', avg_score)
    print('-' * 200)


def merge_entities(df):
    to_be_removed = []
    for i in df.index:
        i_w = df['word'][i]
        if len(i_w) < 3: continue
        if i % 1000 == 0: print(i, i_w.lower()[0])
        for j in df.index[i+1:]:
            j_w = df['word'][j]
            if not i_w.lower()[0] == j_w.lower()[0]: break
            if i_w == j_w.lower() or i_w == j_w[:-1]:
                df.at[i, 'no_occurrences'] += df.at[j, 'no_occurrences']
                # print('To be merged:', i_w, j_w)
                to_be_removed += [df['word'][j]]
    return df[df['word'].apply(lambda x: x not in to_be_removed)]


def link_entities_to_categories(articles_df, essentials_df):

    categories_list = []
    for ind in articles_df.index:
        is_mittmedia = articles_df['brand'][ind] == 'MITTMEDIA'
        for tag in articles_df['tags'][ind]:
            if not is_mittmedia or (is_mittmedia and tag['category'].startswith('RYF-')):
                categories_list += [{'category': tag['name'], 'article_id': articles_df['id'][ind]}]
    categories_df = pd.DataFrame(categories_list)
    categories_df = categories_df.groupby('category')['article_id'].apply(list).reset_index(name='article_ids')
    categories_df['len'] = categories_df['article_ids'].str.len()
    categories_df = categories_df.sort_values(by='len', ascending=False).drop(columns='len')

    new_column = []
    for i in categories_df.index:
        cat_ent = []
        cat_cnt = []
        for article_id in categories_df['article_ids'][i]:
            filtered = essentials_df[essentials_df['article_id'] == article_id]
            for j in filtered.index:
                if filtered['word'][j] in cat_ent:
                    cat_cnt[cat_ent.index(filtered['word'][j])] += 1
                else:
                    cat_ent += [filtered['word'][j]]
                    cat_cnt += [1]
        new_column += [sorted(zip(cat_cnt,cat_ent), reverse=True)]

    categories_df['entities'] = new_column

    most_frequent = categories_df.head(5)
    for ind in most_frequent.index:
        print('\n', most_frequent['category'][ind])
        for entity in most_frequent['entities'][ind]:
            if entity[0] > 10: print(entity)

    categories_df['no_unique_entities'] = categories_df['entities'].str.len()

    tot_no = []
    for ind in categories_df.index:
        tot_no += [0]
        for i, entity in enumerate(categories_df['entities'][ind]):
            if entity: tot_no[-1] += entity[0]

    categories_df['tot_no_entities'] = tot_no

    return categories_df


data_frames = create_data_frames()
articles_df = data_frames[0]
entities_df = data_frames[1]
ambiguous_df = data_frames[2]
essentials_df = data_frames[3]

initial_analysis(articles_df, entities_df, ambiguous_df, essentials_df)

print('NUMBER OF OCCURRENCES PER ENTITY')
unique_entities = pd.DataFrame(essentials_df.groupby('word')['article_id'].nunique()).reset_index()
unique_entities = unique_entities.rename(columns={'article_id': 'no_occurrences'})
merged_entities = merge_entities(unique_entities)
merged_entities = merged_entities.sort_values(by=['no_occurrences'], ascending=False)
print(merged_entities)
count = pd.DataFrame(merged_entities.groupby('no_occurrences').size()).reset_index().rename(columns={0: 'no_words'})
count = count[count['no_words'] > 1]
count.plot(x='no_occurrences', y='no_words')

per_article = []
grouped_entities = entities_df.groupby(['article_id']).count().reset_index()
for ind in articles_df.index:
    no_entities = grouped_entities[grouped_entities['article_id'] == articles_df['id'][ind]]['index']
    no_entities = no_entities.item() if len(no_entities) > 0 else 0
    article_len = len(articles_df['content_text'][ind])
    per_article += [{'no_entities': no_entities, 'article_len': article_len}]

per_article_df = pd.DataFrame(per_article).sort_values(by=['no_entities'], ascending=False)
per_article_df.plot(x='no_entities', y='article_len')

categories_df = link_entities_to_categories(articles_df, essentials_df)

x = categories_df['no_unique_entities'].values.reshape(-1, 1)
y = categories_df['tot_no_entities'].values.reshape(-1, 1)
linear_regressor = LinearRegression().fit(x, y)
y_pred = linear_regressor.predict(x)
print(linear_regressor.coef_)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
hist = categories_df.hist(bins=70)
plt.show()

