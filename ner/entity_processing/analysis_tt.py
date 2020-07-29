from collections import Counter

import numpy as np
import pandas as pd

from ..utils.file_handling import write_df_to_file, read_df_from_file


def link_entities_to_categories(articles, entities):

    categories_list = []

    for i in articles.index:
        aid = articles["id"][i]
        for category in articles["categories"][i]:
            categories_list += [{"category": category, "article_id": aid}]
    categories = pd.DataFrame(categories_list)
    categories = (
        categories.groupby(categories["category"].map(tuple))["article_id"]
        .apply(list)
        .reset_index(name="article_ids")
    )
    categories["no_uses"] = categories["article_ids"].str.len()

    entity_column = []

    for i in categories.index:
        ents = []
        cnts = []

        for aid in categories["article_ids"][i]:
            filtered = entities[entities["article_ids"].apply(lambda x: aid in x)]

            for j in filtered.index:
                ent = filtered["word"][j]
                cnt = filtered["article_ids"][j].count(aid)

                if ent in ents:
                    cnts[ents.index(ent)] += cnt
                else:
                    ents += [ent]
                    cnts += [cnt]

        entity_column += [sorted(zip(cnts, ents), reverse=True)]

    categories["entities"] = entity_column
    categories["no_unique_entities"] = categories["entities"].str.len()

    # most_frequent = categories_df.head(5)
    # print(most_frequent)
    # for i in most_frequent.index:
    #     print('\n', most_frequent['category'][i])
    #     for entity in most_frequent['entities'][i]:
    #         if entity[0] > 10: print(entity)

    tot_no_column = []

    for i in categories.index:
        tot_no_column += [0]

        for entity in categories["entities"][i]:
            if entity:
                tot_no_column[-1] += entity[0]

    categories["tot_no_entities"] = tot_no_column

    return categories


def link_categories_to_entities(articles, entities):
    linked = []

    for i in entities.index:
        ent_cats = []
        entity = entities.iloc[i]
        aids = entity["article_ids"]

        for aid in aids:
            cats = articles[articles["id"] == aid]["categories"].tolist()[0]
            ent_cats += [tuple(cat) for cat in cats]

        ent_cats = dict(Counter(ent_cats))
        linked += [{"entity": entity["word"], "categories": ent_cats}]

    return pd.DataFrame(linked)


articles = read_df_from_file("data/dataframes/articles_tt_new_df.jsonl")
merged_entities = read_df_from_file("data/dataframes/merged_entities_tt_new_df.jsonl")

print("Analyzing categories…")
categories = link_entities_to_categories(articles, merged_entities)
lookup = link_categories_to_entities(articles, merged_entities)
print("Done analyzing!")

write_df_to_file(categories, "data/dataframes/categories_tt_new_df.jsonl")
write_df_to_file(lookup, "data/dataframes/tt_entity_lookup_df.jsonl")
