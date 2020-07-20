import jsonlines
import lemmy
import pandas as pd

from ..utils.file_handling import create_dfs_from_file, write_df_to_file


def calculate_average_score(df):
    tot_score = 0
    no_scores = 0

    for i in df.index:
        scores = df["score"][i]
        tot_score += sum(scores)
        no_scores += len(scores)

    return tot_score / no_scores


def merge_entities(df):
    lemmatizer = lemmy.load("sv")
    lemmatize = lambda x: lemmatizer.lemmatize("PROPN", x)[0].lower()

    remove = []
    for i in df.index:
        i_w = df["word"][i]
        if len(i_w) < 3:
            continue
        i_l = lemmatize(i_w)

        for j in df.index[i + 1 :]:
            j_w = df["word"][j]
            if not i_w.lower()[0] == j_w.lower()[0]:
                break
            j_l = lemmatize(j_w)

            if i_l == j_l or i_w == j_l[0]:
                df.at[i, "article_ids"] += df.at[j, "article_ids"]
                remove += [j_w]

    deduplicated = df[df["word"].apply(lambda x: x not in remove)]

    return deduplicated


articles, entities = create_dfs_from_file("data/output/results_tt.jsonl", True)

df = entities.groupby("word")["article_id"].apply(list).reset_index(name="article_ids")
unique_entities = pd.DataFrame(df)

print("Merging entities…")
merged_entities = merge_entities(unique_entities.copy())
merged_entities["no_occurrences"] = merged_entities["article_ids"].str.len()
merged_entities = merged_entities.sort_values(by=["no_occurrences"], ascending=False)
print("Merged!")

print(merged_entities)

write_df_to_file(articles, "data/dataframes/articles_tt.jsonl")
write_df_to_file(entities, "data/dataframes/unambiguous_entities_tt_df.jsonl")
write_df_to_file(merged_entities, "data/dataframes/merged_entities_tt_df.jsonl")
