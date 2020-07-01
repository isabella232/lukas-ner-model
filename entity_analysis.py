import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from help_functions import write_df_to_file, read_df_from_file


def linear_regression(x_df, y_df):
    x = x_df.values.reshape(-1, 1)
    y = y_df.values.reshape(-1, 1)
    linear_regressor = LinearRegression().fit(x, y)
    y_pred = linear_regressor.predict(x)
    print("Linear regression slope:", linear_regressor.coef_)
    plt.scatter(x, y)
    plt.plot(x, y_pred, color="red")


def link_entities_to_categories(articles, entities):

    categories_list = []
    for ind in articles.index:
        is_mittmedia = articles["brand"][ind] == "MITTMEDIA"
        for tag in articles["tags"][ind]:
            mittmedia_categories = is_mittmedia and tag["category"].startswith("RYF-")
            if not is_mittmedia or mittmedia_categories:
                categories_list += [
                    {"category": tag["name"], "article_id": articles["id"][ind]}
                ]

    categories = pd.DataFrame(categories_list)
    categories = (
        categories.groupby("category")["article_id"]
        .apply(list)
        .reset_index(name="article_ids")
    )
    categories["no_uses"] = categories["article_ids"].str.len()

    new_column = []
    for i in categories.index:
        ents = []
        cnts = []
        for article_id in categories["article_ids"][i]:
            filtered = entities[
                entities["article_ids"].apply(lambda x: article_id in x)
            ]
            for j in filtered.index:
                ent = filtered["word"][j]
                cnt = filtered["article_ids"][j].count(article_id)
                if ent in ents:
                    cnts[ents.index(ent)] += cnt
                else:
                    ents += [ent]
                    cnts += [cnt]

        new_column += [sorted(zip(cnts, ents), reverse=True)]

    categories["entities"] = new_column
    categories["no_unique_entities"] = categories["entities"].str.len()

    # most_frequent = categories_df.head(5)
    # print(most_frequent)
    # for ind in most_frequent.index:
    #     print('\n', most_frequent['category'][ind])
    #     for entity in most_frequent['entities'][ind]:
    #         if entity[0] > 10: print(entity)

    tot_no = []
    for ind in categories.index:
        tot_no += [0]
        for entity in categories["entities"][ind]:
            if entity:
                tot_no[-1] += entity[0]

    categories["tot_no_entities"] = tot_no
    return categories


articles = read_df_from_file("data/articles_df.jsonl")
unam_entities = read_df_from_file("data/unambiguous_entities_df.jsonl")
merged_entities = read_df_from_file("data/merged_entities_df.jsonl")

# Visualize how many entities occur x number of times
count = (
    pd.DataFrame(merged_entities.groupby("no_occurrences").size())
    .reset_index()
    .rename(columns={0: "no_entities"})
)
count = count[count["no_entities"] > 1]
# count.plot(x='no_occurrences', y='no_entities')
# plt.show()

# Visualize the relationship between text length and the number of entities found (per article)
per_article_list = []
grouped_entities = unam_entities.groupby(["article_id"]).count().reset_index()
for ind in articles.index:
    no_entities = grouped_entities[
        grouped_entities["article_id"] == articles["id"][ind]
    ]["index"]
    no_entities = no_entities.item() if len(no_entities) > 0 else 0
    article_len = len(articles["content_text"][ind])
    per_article_list += [{"no_entities": no_entities, "article_len": article_len}]

per_article = pd.DataFrame(per_article_list).sort_values(
    by=["no_entities"], ascending=False
)
# linear_regression(per_article['no_entities'], per_article['article_len'])
# plt.show()

# Category-wise analysis
categories = link_entities_to_categories(articles, merged_entities)
print(categories.sort_values(by="no_uses", ascending=False))
linear_regression(categories["no_unique_entities"], categories["tot_no_entities"])
hist = categories.hist(bins=70)
plt.show()

write_df_to_file(categories, "data/categories_df_new.jsonl")

