from collections import Counter
import pickle
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from scipy.spatial.distance import cosine
from wmd import WMD
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils.parse_articles import get_articles
from .utils.file_handling import write_df_to_file, read_df_from_file


def create_embedding(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    # The last hidden-state is the first element of the output tuple
    last_hidden_states = outputs[0]

    return last_hidden_states


def initial_similarity_comparison():
    categories = read_df_from_file("categories_df")
    categories["embedding"] = categories["category"].apply(
        lambda x: create_embedding(x)
    )

    copy = categories.copy()

    cnt = 0
    for i in categories.index:
        emb_i = categories["embedding"][i]

        for j in copy.index[i + 1 :]:
            emb_j = categories["embedding"][j]
            sim = cos(emb_i, emb_j)

            if sim.item() > 0.995:
                cnt += 1
                print(
                    "Merged categories",
                    cnt,
                    sim.item(),
                    categories["category"][i],
                    categories["category"][j],
                )


def TFIDF_article_similarity():
    stopwords = stopwords.words("swedish")
    articles = get_articles("data/articles_10k.json")

    corpus = [article["content_text"] for article in articles]

    vect = TfidfVectorizer(min_df=1, stop_words=stopwords)
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T

    return pairwise_similarity.toarray()


def BERT_article_similarity():

    articles = get_articles("data/articles_small.json")
    stopwords = stopwords.words("swedish")
    embeddings = []
    for article in articles:
        temp = []
        sentences = article["content_text"].replace("\n\n", ".").split(".")
        for sentence in sentences:
            if not sentence.strip():
                continue
            words = sentence.split()
            print(tokenizer.encode(sentence))
            # print([word for word in words if word not in stopwords ])
            ok_ind_w = [i for i in range(len(words)) if words[i] not in stopwords]
            token_lens = [len(tokenizer.encode(word)) - 2 for word in words]
            ok_ind_t = []
            for i in ok_ind_w:
                prev = sum(token_lens[0:i]) if i > 0 else 0
                ok_ind_t += [prev + i for i in range(0, token_lens[i])]
            ok_ind_t = [i + 1 for i in ok_ind_t]
            try:
                embedding = create_embedding(sentence.strip() + ".")
                embedding = embedding[:, ok_ind_t, :]
                temp += [embedding]
            except IndexError:  # 1541 max length sentence
                print(sentence)
                continue
        embeddings += [temp]
        print("-" * 100)
    print(len(embeddings), "article embeddings created!")

    all_sims = []
    for i, art_i in enumerate(embeddings):
        art_sims = []
        for j, art_j in enumerate(embeddings):
            print("Now comparing article", i, "with article", j, "…")
            if i == j:
                continue
            sen_sims = []
            for sen_i in art_i:
                tok_sims = []
                for sen_j in art_j:
                    for tok_i in range(0, sen_i.size()[1]):
                        for tok_j in range(0, sen_j.size()[1]):
                            sim = cos(sen_i[:, tok_i, :], sen_j[:, tok_j, :]).item()
                            tok_sims += [sim]
                sen_sims += [max(tok_sims)]
            art_sims += [(sum(sen_sims) / len(sen_sims)) ** 2]
        all_sims += [art_sims]

    print("[hockey, hockey, börs, börs]")
    for sim in all_sims:
        print(sim)


def create_sub_lookup(sub_lookup, first_char):
    sub_lookup.sort(key=lambda x: x[0])
    hashed, indexes = zip(*sub_lookup)

    return [{"first": first_char, "hashed": hashed, "indexes": indexes}]


def partition_into_lookup(embeddings):
    lookup = []
    sub_lookup = []
    first_char = embeddings[0]["entity"][0]

    for i, emb in enumerate(embeddings):

        first = emb["entity"][0]
        last_iteration = i == len(embeddings) - 1

        if not first_char == first:
            lookup += create_sub_lookup(sub_lookup, first_char)
            sub_lookup = []
            first_char = first

        sub_lookup += [(hash(emb["entity"]), i)]

        if last_iteration:
            lookup += create_sub_lookup(sub_lookup, first_char)

    return lookup


def create_entity_embeddings(**kwargs):
    if len(kwargs) > 1:
        any_common = lambda x: True if set(x) & kwargs["selected_aids"] else False

        entities_1 = read_df_from_file(kwargs["path_1"])
        entities_1 = entities_1[entities_1["article_ids"].apply(any_common)]
        entities_2 = read_df_from_file(kwargs["path_2"])

        all_entities = pd.concat([entities_1, entities_2]).drop_duplicates(
            subset=["word"], keep="first"
        )
    else:
        all_entities = read_df_from_file(kwargs["path"])

    all_entities = all_entities["word"].tolist()

    print("Creating embeddings…")
    tot_len = len(all_entities)
    embeddings = []
    for i, entity in enumerate(all_entities):
        print(f"{i+1}/{tot_len}")
        embeddings += [
            {
                "entity": entity,
                "hash": hash(entity),
                "embedding": create_embedding(entity),
            }
        ]

    print("Sorting…")
    embeddings.sort(key=lambda x: x["entity"])

    print("Partitioning…")
    lookup = partition_into_lookup(embeddings)

    print("Pickling…")
    with open("data/pickles/new_entity_embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f)

    with open("data/pickles/new_entities_lookup.pickle", "wb") as f:
        pickle.dump(lookup, f)


def binary_search(lookup, val):
    hash_list = lookup["hashed"]

    first = 0
    last = len(hash_list) - 1
    index = -1

    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        if hash_list[mid] == val:
            index = mid
        else:
            if val < hash_list[mid]:
                last = mid - 1
            else:
                first = mid + 1

    return lookup["indexes"][index]


def retrieve_embedding(entity, lookup, embeddings):
    sub_lookup = [sub for sub in lookup if sub["first"] == entity[0]]
    try:
        index = binary_search(sub_lookup[0], hash(entity))
        embedding = embeddings[index]["embedding"]
    except IndexError:
        print(entity)
        print(sub_lookup)

    return embedding


def rescale(vs):
    mn = min(vs)
    mx = max(vs)
    denominator = mx - mn
    return [(v - mn) / denominator for v in vs]
    # scaler = 1 / sum(vs)
    # return [scaler * v for v in vs]


def filter_out_infrequent(df):
    df = df[df["no_uses"] > 2]

    return df  # .reset_index(drop=True)


def calculate_entity_weight(df, i1, i2):
    frequency = df["entities"][i1][i2][0] / df["tot_no_entities"][i1]

    return frequency


def compare_categories(**kwargs):
    print("Unpickling…")
    with open("data/pickles/new_entity_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)
    with open("data/pickles/new_entities_lookup.pickle", "rb") as f:
        lookup = pickle.load(f)
    print("Unpickled!")

    categories = kwargs["categories"]

    if "top_categories" in kwargs:
        top_categories = kwargs["top_categories"]
    else:
        top_categories = categories.copy()

    if "selected" in kwargs:
        selected = kwargs["selected"]
    else:
        selected = categories["category"].tolist()

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

                    shortest = range(0, min(emb_i.shape[1], emb_j.shape[1]))
                    emb_i_reshape = torch.reshape(emb_i[:, shortest, :], (-1,))
                    emb_j_reshape = torch.reshape(emb_j[:, shortest, :], (-1,))

                    sim = cos(emb_i_reshape, emb_j_reshape)
                    single_ent[j2] = sim.item() * w_i / math.exp(abs(w_i - w_j))

                ent_sim[i2] = max(single_ent) if single_ent else 0

            cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0

        sim_matrix[i1] = rescale(cat_sim)
        print(f"--- Iteration {i1}: {(time.time() - iter_time)/60} min ---")

    with open("data/pickles/new3_similarity_matrix.pickle", "wb") as f:
        pickle.dump(sim_matrix, f)


def load_and_print_top_similarities(categories, top_categories, selected):
    with open("data/pickles/new2_similarity_matrix.pickle", "rb") as f:
        sim_matrix = pickle.load(f)

    max_val = np.fliplr(np.sort(sim_matrix, axis=1)[:, -17:])
    max_ind = np.fliplr(np.argsort(sim_matrix, axis=1)[:, -17:])

    # max_val = np.fliplr(np.sort(sim_matrix, axis=1))
    # max_ind = np.fliplr(np.argsort(sim_matrix, axis=1))

    print(sim_matrix[sim_matrix > 0.5].shape)
    threshold = 0.9

    top_cats_list = top_categories["category"].values.tolist()
    top_cats_list = [x.split()[0] for x in top_cats_list]
    tot_scores = [0] * len(top_cats_list)

    for ind in categories.index:
        category = categories["category"][ind]
        no_unique = categories["no_unique_entities"][ind]
        if not category in selected:
            continue
        a = np.where(max_val[ind, :] >= threshold)
        m_v = max_val[ind, :][a]
        m_i = max_ind[ind, :][a]
        maxs = [
            (top_categories["category"][n], max_val[ind, :][i])
            for i, n in enumerate(max_ind[ind])
        ]
        # maxs = [(top_categories["category"][n], m_v[i]) for i, n in enumerate(m_i)]
        print("-" * 60)
        print(
            f"{category} (with {no_unique} unique entities) has largest similarity with:"
        )

        for m in maxs:
            i = top_cats_list.index(m[0].split()[0])
            tot_scores[i] += m[1]
            print(m)

    return dict(zip(top_cats_list, tot_scores))


def top_categories_plots(scores, entities):
    # scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    b = plt.figure(1)
    plt.bar(scores.keys(), scores.values())
    plt.title("Score Distribution")
    plt.xlabel("Top Category")
    plt.ylabel("Aggregated Score")
    b.show()

    s = plt.figure(2)
    plt.scatter(entities, scores.values())
    plt.title("Covariance Between Number of Entities & Score")
    plt.xlabel("Number of Unique Entities")
    plt.ylabel("Aggregated Score")
    s.show()
    plt.show()


tokenizer = BertTokenizer.from_pretrained("KB/bert-base-swedish-cased-ner")
model = BertModel.from_pretrained("KB/bert-base-swedish-cased-ner")
cos = nn.CosineSimilarity(dim=0)


categories = read_df_from_file("data/dataframes/categories_10k_df.jsonl")
top_categories = read_df_from_file("data/dataframes/top_categories_df.jsonl")
selected = [
    # "Musik",
    # "Brottslighet",
    # "Olyckor",
    # "Näringsliv",
    # "Skolsystemet",
    "Ekologi",
    "Sjukdomar & tillstånd",
    "Familjefrågor",
    "Anställningsförhållanden",
    "Mat & dryck",
    # "Politiska frågor",
    # "Religiösa byggnader",
    # "Samhällsvetenskaper",
    # "Infrastruktur",
    # "Fotboll",
    # "Oroligheter",
    # "Väderfenomen",
]

selected_aids = categories[categories["category"].apply(lambda x: x in selected)][
    "article_ids"
].tolist()
selected_aids = set([aid for sublist in selected_aids for aid in sublist])


# create_entity_embeddings(path="data/dataframes/merged_entities_10k_df.jsonl")
# compare_categories(categories=categories)

# create_entity_embeddings(
#     path_1="data/dataframes/merged_entities_10k_df.jsonl",
#     path_2="data/dataframes/merged_entities_mittmedia_df.jsonl",
#     selected_aids=selected_aids,
# )
# start_time = time.time()
# compare_categories(
#     categories=categories, top_categories=top_categories, selected=selected
# )
# print(f"--- Total: {(time.time() - start_time)/60} min ---")

top_scores = load_and_print_top_similarities(categories, top_categories, selected)
# top_ents = top_categories["no_uses"].values.tolist()
# top_categories_plots(top_scores, top_ents)

# TODO: Jämföra enskilda artiklar mot toppkategori

