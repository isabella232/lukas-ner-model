import jsonlines
import pandas as pd


def write_output_to_file(output, path):
    with jsonlines.open(path, mode="w") as writer:
        for obj in output:
            writer.write(obj)


def create_dfs_from_file(path, include_articles):
    with jsonlines.open(path) as reader:

        articles_list = []
        entities_list = []

        for obj in reader:
            if include_articles:
                articles_list += [obj["article"]]
            article_id = obj["article"]["id"]

            for entity in obj["entities"]:
                entities_list += [entity]
                entities_list[-1]["article_id"] = article_id

    return pd.DataFrame(articles_list), pd.DataFrame(entities_list)


def write_df_to_file(df, path):
    json_form = df.to_json(orient="records", lines=True, force_ascii=False)

    with open(path, "w") as f:
        f.write(json_form)


def read_df_from_file(path):
    with jsonlines.open(path) as reader:
        obj_list = [obj for obj in reader]

    return pd.DataFrame(obj_list)
