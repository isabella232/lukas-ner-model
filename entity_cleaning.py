import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
import lemmy
from help_functions import write_df_to_file


def create_data_frames():
    with jsonlines.open('data/new_results.jsonl') as reader:
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
    ambiguous_df = entities_df[entities_df['entity'].apply(lambda x: type(x) == list)]
    entities_df = entities_df[entities_df['entity'].apply(lambda x: type(x) != list)]
    essentials_df = entities_df[entities_df['entity'].apply(lambda x: x in ['PER', 'ORG', 'LOC', 'EVN'])]

    return articles_df, entities_df, ambiguous_df, essentials_df


def merge_entities(df):
    #stemmer = SnowballStemmer('swedish')
    lemmatizer = lemmy.load('sv')
    to_be_removed = []
    #merges = []
    for i in df.index:
        i_w = df['word'][i]
        if len(i_w) < 3: continue
        #i_s = stemmer.stem(i_w)
        i_l = lemmatizer.lemmatize('PROPN', i_w)[0].lower()
        if i % 1000 == 0: print(i, i_w.lower()[0])
        for j in df.index[i+1:]:
            j_w = df['word'][j]
            if not i_w.lower()[0] == j_w.lower()[0]: break
            #j_s = stemmer.stem(j_w)
            j_l = lemmatizer.lemmatize('PROPN', j_w)[0].lower()
            if i_l == j_l or i_w == j_l[0]:
                df.at[i, 'article_ids'] += df.at[j, 'article_ids']
                #merges += [{'1st': i_w, '2nd': j_w}]
                to_be_removed += [j_w]
    return df[df['word'].apply(lambda x: x not in to_be_removed)]
    #return merges


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


dfs = create_data_frames()
#initial_analysis(dfs[0], dfs[1], dfs[2], dfs[3])

unique_entities = pd.DataFrame(dfs[3].groupby('word')['article_id'].apply(list).reset_index(name='article_ids'))
merged_entities = merge_entities(unique_entities.copy())
merged_entities['no_occurrences'] = merged_entities['article_ids'].str.len()
merged_entities = merged_entities.sort_values(by=['no_occurrences'], ascending=False)
cols = merged_entities.columns.tolist()
cols = cols[0:1] + cols[-1:] + cols[1:-1]
merged_entities = merged_entities[cols]

write_df_to_file(dfs[0], 'data/articles_df.jsonl')
write_df_to_file(dfs[1], 'data/unambiguous_entities_df.jsonl')
write_df_to_file(merged_entities, 'data/merged_entities_df.jsonl')
