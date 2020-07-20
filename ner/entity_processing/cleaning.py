import jsonlines
import lemmy
import pandas as pd

from ..utils.file_handling import create_dfs_from_file, write_df_to_file


def create_data_frames():
    articles, entities = create_dfs_from_file("data/output/results_10k.jsonl", True)

    entities = entities[entities["word"] != "s"]
    ambiguous = entities[entities["entity"].apply(lambda x: type(x) == list)]
    entities = entities[entities["entity"].apply(lambda x: type(x) != list)]

    essentials = ["PER", "ORG", "LOC", "EVN"]
    essentials = entities[entities["entity"].apply(lambda x: x in essentials)]

    return articles, entities, ambiguous, essentials


def calculate_average_score(df):
    tot_score = 0
    no_scores = 0

    for i in df.index:
        scores = df["score"][i]
        tot_score += sum(scores)
        no_scores += len(scores)

    return tot_score / no_scores


def initial_analysis(articles, entities, ambiguous, essentials):

    divider = "-" * 100

    print(divider, "\nARTICLES\n", articles)
    print(divider, "\nENTITIES\n", entities)
    print(divider, "\nAMBIGUOUS ENTITIES\n", ambiguous)
    print(divider, "\nESSENTIAL ENTITIES\n", essentials)

    art_len = len(articles)
    ent_len = len(entities)
    amb_len = len(ambiguous)
    ess_len = len(essentials)

    avg_score = calculate_average_score(entities)
    avg_no_entities = ent_len / art_len

    amb_score = calculate_average_score(ambiguous)
    amb_no = amb_len / art_len
    amb_share = amb_len / (amb_len + ent_len)

    ess_score = calculate_average_score(essentials)
    ess_no = ess_len / art_len
    ess_share = ess_len / ent_len

    print(divider)
    print(
        f"Total:\t\taverage score = {avg_score} | average number of entities per article = {avg_no_entities}"
    )
    print(
        f"Ambiguous:\taverage score = {amb_score} | average number of entities per article = {amb_no} | share = {amb_share}"
    )
    print(
        f"Essential:\taverage score = {ess_score} | average number of entities per article = {ess_no} | share = {ess_share}"
    )
    print(divider)

    ent_types = ["PER", "ORG", "LOC", "TME", "MSR", "WRK", "EVN", "OBJ"]

    for ent_type in ent_types:
        filtered = entities[entities["entity"] == ent_type]
        score = calculate_average_score(filtered)
        share = len(filtered) / ent_len

        print(f"{ent_type}: share = {share}\t| average score = {score}")

    print(divider)


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


articles, entities, ambiguous, essentials = create_data_frames()
initial_analysis(articles, entities, ambiguous, essentials)

df = (
    essentials.groupby("word")["article_id"].apply(list).reset_index(name="article_ids")
)
unique_entities = pd.DataFrame(df)

print("Merging entities…")
merged_entities = merge_entities(unique_entities.copy())
merged_entities["no_occurrences"] = merged_entities["article_ids"].str.len()
merged_entities = merged_entities.sort_values(by=["no_occurrences"], ascending=False)
print("Merged!")

write_df_to_file(articles, "data/dataframes/articles_10k.jsonl")
write_df_to_file(entities, "data/dataframes/unambiguous_entities_10k_df.jsonl")
write_df_to_file(merged_entities, "data/dataframes/merged_entities_10k_df.jsonl")
