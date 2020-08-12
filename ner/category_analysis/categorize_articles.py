import pickle
import random

import numpy as np
import torch
from torch import nn

from .utils.parse_articles import get_articles
from .utils.file_handling import write_df_to_file, read_df_from_file
from .category_similarity import (
    binary_search,
    retrieve_embedding,
    rescale,
    calculate_entity_weight,
)


def max_similarity(emb_i, emb_j):
    cos = nn.CosineSimilarity(dim=0)
    sim = 0

    len_i = emb_i.shape[1]
    len_j = emb_j.shape[1]
    len_diff = len_i - len_j

    if len_diff == 0:
        emb_i_reshape = torch.reshape(emb_i, (-1,))
        emb_j_reshape = torch.reshape(emb_j, (-1,))
        sim = cos(emb_i_reshape, emb_j_reshape)
    else:
        longer, shorter = (emb_i, emb_j) if len_diff > 0 else (emb_j, emb_i)
        len_short = min(len_i, len_j)
        short_reshape = torch.reshape(shorter, (-1,))

        for i in range(abs(len_diff)):
            sub_range = range(i, i + len_short)
            long_reshape = torch.reshape(longer[:, sub_range, :], (-1,))
            sim = max(sim, cos(short_reshape, long_reshape))

    return sim


def calculate_similarities(categories, articles):
    print("Unpickling…")
    with open("data/pickles/tt_embeddings.pickle", "rb") as f:
        embeddings = torch.load(f)
    with open("data/pickles/tt_lookup.pickle", "rb") as f:
        lookup = pickle.load(f)
    print("Unpickled!")

    entities = read_df_from_file("data/dataframes/merged_entities_tt_df.jsonl")

    cos = nn.CosineSimilarity(dim=0)

    no_articles = articles.shape[0]
    no_categories = categories.shape[0]
    sim_matrix = np.zeros([no_articles, no_categories])

    i = 0

    for i1 in articles.index:
        aid = articles["id"][i1]
        article_entities = []
        tot_cnt = 0
        for e in entities.index:
            aids = entities["article_ids"][e]
            cnt = aids.count(aid)
            if cnt > 0:
                article_entities += [(entities["word"][e], cnt)]
                tot_cnt += cnt
        [print(ent) for ent in article_entities]
        cat_sim = [0] * no_categories

        for j1 in categories.index:
            print(f"Comparing article {i1} with category {j1}…")

            len_i = len(article_entities)
            len_j = len(categories["entities"][j1])
            ent_sim = [None] * len_i

            for i2 in range(0, len_i):
                w_i = article_entities[i2][1] / tot_cnt
                ent_i = article_entities[i2][0]
                emb_i = retrieve_embedding(ent_i, lookup, embeddings)

                ent_pair = [None] * len_j
                single_ent = [None] * len_j

                for j2 in range(0, len_j):
                    w_j = calculate_entity_weight(categories, j1, j2)
                    ent_j = categories["entities"][j1][j2][1]
                    emb_j = retrieve_embedding(ent_j, lookup, embeddings)

                    sim = max_similarity(emb_i, emb_j)
                    # shortest = range(min(emb_i.shape[1], emb_j.shape[1]))
                    # emb_i_reshape = torch.reshape(emb_i[:, shortest, :], (-1,))
                    # emb_j_reshape = torch.reshape(emb_j[:, shortest, :], (-1,))

                    # sim = cos(emb_i_reshape, emb_j_reshape)
                    single_ent[j2] = sim.item() * w_i  # / math.exp(abs(w_i - w_j))
                    ent_pair[j2] = (ent_i, ent_j)

                ent_sim[i2] = max(single_ent) if single_ent else 0
                ent_pair_ind = single_ent.index(max(single_ent))
                print(ent_pair[ent_pair_ind], ent_sim[i2])

            cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0

        sim_matrix[i] = rescale(cat_sim)
        i += 1

    return sim_matrix


def print_similarities(categories, articles, sim_matrix):
    max_val = np.fliplr(np.sort(sim_matrix, axis=1)[:, -17:])
    max_ind = np.fliplr(np.argsort(sim_matrix, axis=1)[:, -17:])

    cats_list = categories["category"].values.tolist()
    cats_list = [x.split()[0] for x in cats_list]

    i = 0
    for ind in articles.index:
        title = articles["categories"][ind]

        maxs = [
            (categories["category"][n], max_val[i, :][j])
            for j, n in enumerate(max_ind[i])
        ]

        print("_" * 100)
        print(f'Article with title "{title}" has largest similarity with:')
        for m in maxs:
            print(m)
        # print("\n\n", articles["content_text"][ind])
        i += 1


categories = read_df_from_file("data/dataframes/top_categories_df.jsonl")
articles = read_df_from_file("data/dataframes/articles_tt_df.jsonl")

indexes = random.sample(range(0, len(articles.index) - 1), 1)
articles = articles.loc[indexes]

sim_matrix = calculate_similarities(categories, articles)
print_similarities(categories, articles, sim_matrix)
