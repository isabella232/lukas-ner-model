import numpy as np
import pandas as pd
import lemmy

from ..utils.file_handling import create_dfs_from_file, write_df_to_file


def create_data_frames(file_name):
    """Load outputted entities in Pandas DataFrame."""
    articles, entities = create_dfs_from_file(file_name, True)

    ignore = ["TME", "MSR"]
    desired = entities[entities["entity"].apply(lambda x: x not in ignore)]

    return articles, entities, desired


def calculate_average_score(df):
    tot_score, no_scores = 0, 0

    # In older dataframes, the average score for an entity – based on the scores
    # of its individual tokens – was not yet calculated at this point.
    if df["score"].dtype != np.float64:
        for i in df.index:
            scores = df["score"][i]
            tot_score += sum(scores)
            no_scores += len(scores)
        avg_score = tot_score(no_scores)
    else:
        avg_score = df["score"].mean()

    return avg_score


def initial_analysis(articles, entities, desired):
    """Basic analysis of outputted entities."""
    divider = "-" * 100

    print(divider, "\nARTICLES\n", articles)
    print(divider, "\nENTITIES\n", entities)
    print(divider, "\nDESIRED ENTITIES\n", desired)

    art_len = len(articles)
    ent_len = len(entities)
    des_len = len(desired)

    avg_score = calculate_average_score(entities)
    avg_no_entities = ent_len / art_len

    des_score = calculate_average_score(desired)
    des_no = des_len / art_len
    des_share = des_len / ent_len

    print(divider)
    print(
        f"Total:\t\taverage score = {avg_score} | average number of entities per article = {avg_no_entities}"
    )
    print(
        f"Desired:\taverage score = {des_score} | average number of entities per article = {des_no} | share = {des_share}"
    )
    print(divider)

    ent_types = {
        "O",
        "OBJ",
        "TME",
        "ORG/PRS",
        "OBJ/ORG",
        "PRS/WRK",
        "WRK",
        "LOC",
        "ORG",
        "PER",
        "LOC/PRS",
        "LOC/ORG",
        "MSR",
        "EVN",
    }

    for ent_type in ent_types:
        filtered = entities[entities["entity"] == ent_type]
        if not filtered.empty:
            score = calculate_average_score(filtered)
            share = len(filtered) / ent_len

            print(f"{ent_type}: share = {share}\t| average score = {score}")

    print(divider)


def merge_entities(df):
    """Merges similar entities based on their lemmatized form with the purpose
    to reduce spelling/formatting/suffixes variations.
    """
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

            # Continue outer loop if the first letter has changed
            if not i_w.lower()[0] == j_w.lower()[0]:
                break
            j_l = lemmatize(j_w)

            if i_l == j_l or i_w == j_l[0]:
                df.at[i, "article_ids"] += df.at[j, "article_ids"]
                remove += [j_w]

    deduplicated = df[df["word"].apply(lambda x: x not in remove)]

    return deduplicated


articles, entities, desired = create_data_frames("data/output/results_tt_new.jsonl")
initial_analysis(articles, entities, desired)

df = desired.groupby("word")["article_id"].apply(list).reset_index(name="article_ids")
unique_entities = pd.DataFrame(df)

print("Merging entities…")
merged_entities = merge_entities(unique_entities.copy())
merged_entities["no_occurrences"] = merged_entities["article_ids"].str.len()
merged_entities = merged_entities.sort_values(by=["no_occurrences"], ascending=False)
print("Merged!")

write_df_to_file(articles, "data/dataframes/articles_10k.jsonl")
write_df_to_file(entities, "data/dataframes/all_entities_10k_df.jsonl")
write_df_to_file(merged_entities, "data/dataframes/NEW_merged_entities_10k_df.jsonl")
