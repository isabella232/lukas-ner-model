import pickle
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
import torch
from torch import nn


from ..utils.file_handling import read_df_from_file


def create_embedding(word):
    """Creates and returns a word embedding using BERT."""
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    # The last hidden-state is the first element of the output tuple
    last_hidden_states = outputs[0]

    return last_hidden_states


def create_sub_lookup(sub_lookup, first_char):
    sub_lookup.sort(key=lambda x: x[0])
    int_reps, indexes = zip(*sub_lookup)

    return [{"first": first_char, "int_reps": int_reps, "indexes": indexes}]


def partition_into_lookup(embeddings):
    """Partitions the entity embeddings based on their first letters."""
    lookup, sub_lookup = [], []
    first_char = embeddings[0]["entity"][0]

    for i, emb in enumerate(embeddings):

        first = emb["entity"][0]
        last_iteration = i == len(embeddings) - 1

        if not first_char == first:
            lookup += create_sub_lookup(sub_lookup, first_char)
            sub_lookup = []
            first_char = first

        sub_lookup += [(emb["int_rep"], i)]

        if last_iteration:
            lookup += create_sub_lookup(sub_lookup, first_char)

    return lookup


def int_representation(entity):
    """Returns a unique integer representation of a string (entity)."""
    return int.from_bytes(entity.encode(), "little")


def create_entity_embeddings(path_1, path_2=None, selected_aids=None, mittmedia=False):
    """Creates embeddings for entities and saves them in a lookup table."""
    if path_2 is not None and selected_aids is not None:
        entities_1 = read_df_from_file(path_1)

        if mittmedia:
            any_common = lambda x: True if set(x) & selected_aids else False
            entities_1 = entities_1[entities_1["article_ids"].apply(any_common)]

        entities_2 = read_df_from_file(path_2)
        all_entities = pd.concat([entities_1, entities_2]).drop_duplicates(
            subset=["word"], keep="first"
        )

    else:
        all_entities = read_df_from_file(path_1)

    all_entities = all_entities["word"].tolist()

    print("Creating embeddings…")
    tot_len = len(all_entities)
    embeddings = []

    for i, entity in enumerate(all_entities):
        print(f"{i+1}/{tot_len}")
        int_rep = int_representation(entity)
        embeddings += [
            {
                "entity": entity,
                "int_rep": int_rep,
                "embedding": create_embedding(entity),
            }
        ]

    print("Sorting…")
    embeddings.sort(key=lambda x: x["entity"])

    print("Partitioning…")
    lookup = partition_into_lookup(embeddings)

    print("Pickling…")
    with open("data/pickles/tt_embeddings.pickle", "wb") as f:
        torch.save(embeddings, f)

    with open("data/pickles/tt_lookup.pickle", "wb") as f:
        pickle.dump(lookup, f)


def binary_search(lookup, val):
    int_reps = lookup["int_reps"]

    first = 0
    last = len(int_reps) - 1
    index = -1

    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        if int_reps[mid] == val:
            index = mid
        else:
            if val < int_reps[mid]:
                last = mid - 1
            else:
                first = mid + 1

    return lookup["indexes"][index]


def retrieve_embedding(entity, lookup, embeddings):
    """Retrieves entity embedding from the lookup table using binary search."""
    sub_lookup = [sub for sub in lookup if sub["first"] == entity[0]]

    index = binary_search(sub_lookup[0], int_representation(entity))
    embedding = embeddings[index]["embedding"]

    return embedding


def rescale(vs):
    """Rescales similairy vectors."""
    # mn = min(vs)
    # mx = max(vs)
    # denominator = mx - mn
    # return [(v - mn) / denominator for v in vs]
    scaler = 1 / sum(vs)
    return [scaler * v for v in vs]


def calculate_entity_weight(df, i1, i2):
    frequency = df["entities"][i1][i2][0] / df["tot_no_entities"][i1]

    return frequency


def compare_categories(categories, top_categories=None, selected=None):
    """Compares each category with all other categories by calculating similarity using entities."""
    start_time = time.time()

    print("Unpickling…")
    with open("data/pickles/tt_embeddings.pickle", "rb") as f:
        embeddings = torch.load(f)
    with open("data/pickles/tt_lookup.pickle", "rb") as f:
        lookup = pickle.load(f)
    print("Unpickled!")

    if top_categories is None or selected is None:
        top_categories = categories.copy()
        selected = categories["categories"].tolist()

    no_categories = categories.shape[0]
    no_top_categories = top_categories.shape[0]
    sim_matrix = np.zeros([no_categories, no_top_categories])

    for i1 in categories.index:
        iter_time = time.time()
        if not categories["category"][i1] in selected:
            continue

        cat_sim = [0] * no_top_categories

        for j1 in top_categories.index:
            print("Comparing category", i1, "with category", j1, "…")

            len_i = len(categories["entities"][i1])
            len_j = len(top_categories["entities"][j1])
            ent_sim = [None] * len_i

            for i2 in range(0, len_i):
                w_i = calculate_entity_weight(categories, i1, i2)
                ent_i = categories["entities"][i1][i2][1]
                emb_i = retrieve_embedding(ent_i, lookup, embeddings)

                single_ent = [None] * len_j

                for j2 in range(0, len_j):
                    w_j = calculate_entity_weight(top_categories, j1, j2)
                    ent_j = top_categories["entities"][j1][j2][1]
                    emb_j = retrieve_embedding(ent_j, lookup, embeddings)

                    shortest = range(min(emb_i.shape[1], emb_j.shape[1]))
                    emb_i_reshape = torch.reshape(emb_i[:, shortest, :], (-1,))
                    emb_j_reshape = torch.reshape(emb_j[:, shortest, :], (-1,))

                    sim = cos(emb_i_reshape, emb_j_reshape)
                    single_ent[j2] = sim.item() * w_i / math.exp(abs(w_i - w_j))

                ent_sim[i2] = max(single_ent) if single_ent else 0

            cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0

        sim_matrix[i1] = rescale(cat_sim)
        print(f"--- Iteration {i1}: {(time.time() - iter_time)/60} min ---")

    with open("data/pickles/tt_similarity_matrix.pickle", "wb") as f:
        pickle.dump(sim_matrix, f)

    print(f"--- Total: {(time.time() - start_time)/60} min ---")


def load_and_print_top_similarities(categories, top_categories, selected):
    with open("data/pickles/tt_similarity_matrix.pickle", "rb") as f:
        sim_matrix = pickle.load(f)

    max_val = np.fliplr(np.sort(sim_matrix, axis=1)[:, -17:])
    max_ind = np.fliplr(np.argsort(sim_matrix, axis=1)[:, -17:])

    top_cats_list = top_categories["category"].values.tolist()
    top_cats_list = [x.split()[0] for x in top_cats_list]
    tot_scores = [0] * len(top_cats_list)

    for ind in categories.index:
        category = categories["category"][ind]
        tot_no = categories["tot_no_entities"][ind]

        if category not in selected:
            continue

        maxs = [
            (top_categories["category"][n], max_val[ind, :][i])
            for i, n in enumerate(max_ind[ind])
        ]

        print("_" * 100)
        print(f"{category} (with {tot_no} entities) has largest similarity with:")

        for m in maxs:
            i = top_cats_list.index(m[0].split()[0])
            tot_scores[i] += m[1]
            print(m)

    avg_scores = [score / len(selected) for score in tot_scores]

    return dict(zip(top_cats_list, avg_scores))


def top_categories_plots(scores, entities):
    # scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    b = plt.figure(1)
    plt.bar(scores.keys(), scores.values())
    plt.title("Score Distribution")
    plt.xlabel("MittMedia Category")
    plt.ylabel("Average Score")
    b.show()

    s = plt.figure(2)
    plt.scatter(entities, scores.values())
    plt.title("Covariance Between Number of Entities & Score")
    plt.xlabel("Total Number of Entities")
    plt.ylabel("Average Score")
    s.show()
    plt.show()


if __name__ == "__main__":
    model_name = "KB/bert-base-swedish-cased-ner"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    cos = nn.CosineSimilarity(dim=0)

    categories = read_df_from_file("data/dataframes/categories_tt_df.jsonl")
    top_categories = read_df_from_file("data/dataframes/top_categories_df.jsonl")
    selected = [
        "Musik",
        "Brottslighet",
        "Olyckor",
        "Näringsliv",
        "Skolsystemet",
        "Ekologi",
        "Sjukdomar & tillstånd",
        "Familjefrågor",
        "Anställningsförhållanden",
        "Mat & dryck",
        "Politiska frågor",
        "Religiösa byggnader",
        "Samhällsvetenskaper",
        "Infrastruktur",
        "Fotboll",
        "Oroligheter",
        "Väderfenomen",
    ]

    selected_tt = [
        "Politik",
        "Brott, lag och rätt",
        "Ekonomi, affärer och finans",
        "Konst, kultur och nöje",
    ]

    # selected_aids = categories[categories["category"].apply(lambda x: x in selected)][
    #     "article_ids"
    # ].tolist()
    selected_aids = categories["article_ids"].tolist()
    selected_aids = set([aid for sublist in selected_aids for aid in sublist])

    # create_entity_embeddings(path_1="data/dataframes/merged_entities_tt_df.jsonl")
    # compare_categories(categories=categories)

    create_entity_embeddings(
        path_1="data/dataframes/merged_entities_tt_df.jsonl",
        path_2="data/dataframes/merged_entities_mittmedia_df.jsonl",
        selected_aids=selected_aids,
    )
    compare_categories(categories, top_categories, selected_tt)

    top_scores = load_and_print_top_similarities(
        categories, top_categories, selected_tt
    )
    top_ents = top_categories["tot_no_entities"].values.tolist()
    top_categories_plots(top_scores, top_ents)
