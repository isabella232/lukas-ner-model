from collections import Counter

import pandas as pd

from ..utils.file_handling import write_df_to_file, read_df_from_file
from .analysis import link_entities_to_categories


def create_category_df(articles):
    """Similar to its counterpart in analysis.py but adapted to the NER output from TT articles.
       
    Creates a dataframe of all categories in the data, paired with the articles using them.
    """
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

    return categories


def link_categories_to_entities(articles, entities):
    """Creates a dataframe of all found entities, paired with the categories in which they are used.

    The result is used in tt/specific/keyword_service.py.
    """
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

categories = create_category_df(articles)
categories = link_entities_to_categories(merged_entities, categories)
lookup = link_categories_to_entities(articles, merged_entities)

# write_df_to_file(categories, "data/dataframes/categories_tt_new_df.jsonl")
# write_df_to_file(lookup, "data/dataframes/tt_entity_lookup_df.jsonl")
