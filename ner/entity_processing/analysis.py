import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ..utils.file_handling import write_df_to_file, read_df_from_file


def linear_regression(x_df, y_df):
    """Simple linear regression."""

    x = x_df.values.reshape(-1, 1)
    y = y_df.values.reshape(-1, 1)

    linear_regressor = LinearRegression().fit(x, y)
    y_pred = linear_regressor.predict(x)
    print("Linear regression slope:", linear_regressor.coef_)

    s = plt.scatter(x, y)
    p = plt.plot(x, y_pred, color="red")

    return s, p


def create_category_df(articles):
    """Creates a dataframe of all categories in the data, paired with the articles using them."""
    categories_list = []

    for i in articles.index:
        is_mittmedia = articles["brand"][i] == "MITTMEDIA"

        for tag in articles["tags"][i]:
            mittmedia_categories = is_mittmedia and tag["category"].startswith("RYF-")
            is_placemark = tag["category"] == "PLACEMARK"

            if (not is_mittmedia and not is_placemark) or mittmedia_categories:
                categories_list += [
                    {"category": tag["name"], "article_id": articles["id"][i]}
                ]

    categories = pd.DataFrame(categories_list)
    categories = (
        categories.groupby("category")["article_id"]
        .apply(list)
        .reset_index(name="article_ids")
    )

    return categories


def link_entities_to_categories(entities, categories):
    categories["no_uses"] = categories["article_ids"].str.len()

    entity_column = []
    for i in categories.index:
        ents, cnts = [], []

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

    tot_no_column = []

    for i in categories.index:
        tot_no_column += [0]

        for entity in categories["entities"][i]:
            if entity:
                tot_no_column[-1] += entity[0]

    categories["tot_no_entities"] = tot_no_column

    return categories


if __name__ == "__main__":
    articles = read_df_from_file("data/dataframes/articles_10k_df.jsonl")
    all_entities = read_df_from_file("data/dataframes/all_entities_10k_df.jsonl")
    merged_entities = read_df_from_file("data/dataframes/merged_entities_10k_df.jsonl")

    # Visualize how many entities occur x number of times
    count = pd.DataFrame(merged_entities.groupby("no_occurrences").size()).reset_index()
    count = count.rename(columns={0: "no_entities"})
    count = count[count["no_entities"] > 1]

    # Visualize the relationship between text length and the number of entities found (per article)
    per_article_list = []
    grouped_entities = all_entities.groupby(["article_id"]).count().reset_index()

    for i in articles.index:
        no_entities = grouped_entities[
            grouped_entities["article_id"] == articles["id"][i]
        ]
        no_entities = no_entities["index"].item() if len(no_entities) > 0 else 0
        article_len = len(articles["content_text"][i])
        per_article_list += [{"no_entities": no_entities, "article_len": article_len}]

    per_article = pd.DataFrame(per_article_list)
    per_article = per_article.sort_values(by=["no_entities"], ascending=False)

    f1 = plt.figure(1)
    f1.suptitle("Article-Wise")
    plt.xlabel("Number of Found Entities")
    plt.ylabel("Article Length")
    linear_regression(per_article["no_entities"], per_article["article_len"])
    f1.show()

    # Category-wise analysis
    print("Analyzing categories…")
    categories = create_category_df(articles)
    categories = link_entities_to_categories(merged_entities, categories)
    print("Done analyzing!")
    f2 = plt.figure(2)
    f2.suptitle("Category-Wise")
    plt.xlabel("Number of Distinct Entities")
    plt.ylabel("Total number of entities")
    linear_regression(categories["no_unique_entities"], categories["tot_no_entities"])
    f2.show()

    count.plot(x="no_occurrences", y="no_entities")
    categories.hist(bins=70)
    plt.show()

    write_df_to_file(categories, "data/dataframes/categories_10k_df.jsonl")
