import jsonlines
import pandas as pd
from string import punctuation
from file_handling import read_df_from_file


def create_df_from_results(path):
    with jsonlines.open(path) as reader:
        entities_list = []
        for obj in reader:
            article_id = obj["article"]["id"]
            for entity in obj["entities"]:
                entities_list += [entity]
                entities_list[-1]["article_id"] = article_id
    return pd.DataFrame(entities_list)


def clean_entities(df):

    chars = set(punctuation)
    chars.update(["*", "–", "‒", "—", "”", "“", "’", " ", "￭", "✓", "►", "•", "■", "…"])

    is_reg_char = lambda x: x not in chars and not x.isdigit()

    to_be_removed = []
    to_be_duplicated = []
    for ind in df.index:
        word = df["word"][ind]
        word = word.strip()
        reg_chars = [char for char in word if is_reg_char(char)]
        if reg_chars:
            clean = False
            while not clean:
                if word[:1] in chars:
                    word = word[1:].strip()
                elif word[-1:] in chars:
                    word = word[:-1].strip()
                else:
                    clean = True
                    word = word.strip()

            df["word"][ind] = word.strip()

            count = df["count"][ind]
            if count > 1:
                for i in range(count - 1):
                    to_be_duplicated += [df.iloc[ind]]
        else:
            to_be_removed += [word]
    cleaned = df[df["word"].apply(lambda x: x not in to_be_removed)]
    cleaned = cleaned.append(to_be_duplicated)
    return cleaned


def create_entities_df():
    bert_df = create_df_from_results("data/output/results.jsonl")
    bert_df = bert_df[
        bert_df["entity"].apply(lambda x: x in ["PER", "ORG", "LOC"])
    ].reset_index(drop=True)
    nerd_df = create_df_from_results("data/output/results_nerd.jsonl")

    nerd_df = clean_entities(nerd_df)
    nerd_df = nerd_df.drop(["count"], axis=1)

    return nerd_df, bert_df


def compare_unique_entities(nerd_df, bert_df):
    nerd_unique = (
        nerd_df.groupby("word")["entity"].count().index.get_level_values(0).tolist()
    )
    bert_unique = (
        bert_df.groupby("word")["entity"].count().index.get_level_values(0).tolist()
    )

    nerd_diff = [x for x in nerd_unique if x not in bert_unique]
    bert_diff = [x for x in bert_unique if x not in nerd_unique]

    print(nerd_diff, bert_diff)


def extract_mentioned_tags():
    articles_df = read_df_from_file("data/dataframes/articles_df.jsonl")

    tags_dict = []
    for ind in articles_df.index:
        text = articles_df["content_text"][ind]
        tags = [tag["name"] for tag in articles_df["tags"][ind]]
        mentioned_tags = [tag for tag in tags if tag in text]
        tags_dict += [{"id": articles_df["id"][ind], "tags": mentioned_tags}]

    tags_dict = [tags for tags in tags_dict if tags["tags"]]
    return pd.DataFrame(tags_dict)


def evaluate_against_tags(entities, tags):

    id_list = tags["id"].values
    filtered_entities = entities[entities["article_id"].apply(lambda x: x in id_list)]
    filtered_entities = (
        filtered_entities.groupby("article_id")["word"]
        .apply(list)
        .reset_index(name="entities")
    )
    all_tags = []
    found_tags = []
    for ind in tags.index:
        article_id = tags["id"][ind]

        found = filtered_entities[filtered_entities["article_id"] == article_id][
            "entities"
        ].tolist()
        if found:
            [found] = found
        for tag in tags["tags"][ind]:
            all_tags += [tag]
            if tag in found:
                found_tags += [tag]
            # For when BERT entities were merged too much
            # else:
            #     for t in found:
            #         words = t.split()
            #         if len(words) > 2:
            #             combinations = []
            #             i = 0
            #             while i < len(words) - 1:
            #                 combinations += [words[i] + " " + words[i + 1]]
            #                 i += 1
            #             if tag in combinations:
            #                 found_tags += [tag]

    return all_tags, found_tags


dfs = create_entities_df()
nerd_df = dfs[0]
bert_df = dfs[1]

# compare_unique_entities(nerd_df, bert_df)

tags_df = extract_mentioned_tags()

print("Evaluating NERD…")
nerd_results = evaluate_against_tags(nerd_df, tags_df)
all_tags = nerd_results[0]
nerd_found = nerd_results[1]
nerd_score = len(nerd_found) / len(all_tags)

print("Evaluating BERT…")
bert_results = evaluate_against_tags(bert_df, tags_df)
bert_found = bert_results[1]
bert_score = len(bert_found) / len(all_tags)


print("NERD score:", nerd_score, "| BERT score:", bert_score)

nerd_bert_diff = [x for x in nerd_found if x not in bert_found]
print(len(nerd_bert_diff))
# = 108 with filtered BERT, = 41 with unfiltered BERT (minus at least 9)
for entity in nerd_bert_diff:
    print(entity)
print("-" * 50)
diff = [x for x in all_tags if x not in bert_found]
for entity in diff:
    print(entity)
