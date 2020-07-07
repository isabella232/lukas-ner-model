from collections import Counter
import pickle
import math
import time

from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from wmd import WMD
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from utils.parse_articles import get_articles
from utils.file_handling import write_df_to_file, read_df_from_file


def create_embedding(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    # The last hidden-state is the first element of the output tuple
    last_hidden_states = outputs[0]

    return last_hidden_states


def initial_similarity(categories_df):
    categories_df["embedding"] = categories_df["category"].apply(
        lambda x: create_embedding(x)
    )
    print("Embeddings created")
    copy_df = categories_df.copy()

    cnt = 0
    for i in categories_df.index:
        emb_i = categories_df["embedding"][i]
        for j in copy_df.index[i + 1 :]:
            emb_j = categories_df["embedding"][j]
            sim = cos(emb_i, emb_j)
            if sim.item() > 0.995:
                cnt += 1
                print(i, j)
                print(
                    "Merged categories",
                    cnt,
                    sim.item(),
                    categories_df["category"][i],
                    categories_df["category"][j],
                )


def TFIDF_article_similarity():
    articles = get_articles("data/articles_small.json")
    stopwords = stopwords.words("swedish")
    corpus = [article["content_text"] for article in articles]
    vect = TfidfVectorizer(min_df=1, stop_words=stopwords)
    tfidf = vect.fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    print(pairwise_similarity.toarray())


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


def create_entity_embeddings(categories):
    entities = read_df_from_file("data/dataframes/merged_entities_df.jsonl")
    entities["embedding"] = entities["word"].apply(lambda x: create_embedding(x))
    print(entities)

    embeddings_list = []
    for ind in categories.index:
        category_embeddings = []
        for entity in categories["entities"][ind]:
            category_embeddings += [
                entities[entities["word"].apply(lambda x: x == entity[1])]["embedding"]
            ]
        embeddings_list += [category_embeddings]

    with open("data/pickles/entity_embeddings.pickle", "wb") as f:
        pickle.dump(embeddings_list, f)


def create_entity_embeddings_all():
    entities = read_df_from_file("data/dataframes/merged_entities_df.jsonl")
    mm_entities = read_df_from_file(
        "data/dataframes/merged_entities_mittmedia_df.jsonl"
    )
    all_entities = pd.concat([entities, mm_entities])
    all_entities = all_entities.groupby("word")["no_occurrences"].sum()
    all_entities = all_entities.index.get_level_values(0).tolist()

    print("Creating embeddings…")
    embeddings = []
    for entity in all_entities:
        embeddings += [{"entity": entity, "embedding": create_embedding(entity)}]

    with open("data/pickles/all_entity_embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f)


def rescale(vs):
    scaler = 1 / sum(vs)
    mn = min(vs)
    mx = max(vs)
    denominator = mx - mn
    return [(v - mn) / denominator for v in vs]


def filter_out_infrequent(df):
    df = df[df["no_uses"] > 2]

    return df  # .reset_index(drop=True)


def calculate_entity_weight(df, i1, i2):
    frequency = df["entities"][i1][i2][0] / df["tot_no_entities"][i1]
    return frequency


def compare_with_all_categories(categories):
    print("Unpickling…")
    with open("data/pickles/entity_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)
    print("Unpickled!")

    categories = filter_out_infrequent(categories)
    # categories["embedding"] = categories["category"].apply(
    #     lambda x: create_embedding(x)
    # )
    embeddings = list(map(embeddings.__getitem__, categories.index.tolist()))
    categories = categories.reset_index(drop=True)
    write_df_to_file(categories, "data/dataframes/filtered_cats_df.jsonl")
    no_categories = categories.shape[0]
    sim_matrix = np.zeros([no_categories, no_categories])

    for i1 in categories.index:
        cat_sim = [0] * len(categories.index)
        for j1 in categories.index:
            len_i = len(categories["entities"][i1])
            len_j = len(categories["entities"][j1])
            print("Comparing category", i1, "with category", j1, "…")
            ent_sim = [None] * len_i
            for i2 in range(0, len_i):
                w1 = calculate_entity_weight(categories, i1, i2)
                emb_i = embeddings[i1][i2].item()
                single_ent = [None] * len_j
                for j2 in range(0, len_j):
                    w2 = calculate_entity_weight(categories, j1, j2)
                    emb_j = embeddings[j1][j2].item()
                    shortest = range(0, min(emb_i.shape[1], emb_j.shape[1]))
                    emb_i_reshape = torch.reshape(emb_i[:, shortest, :], (-1,))
                    emb_j_reshape = torch.reshape(emb_j[:, shortest, :], (-1,))
                    sim = cos(emb_i_reshape, emb_j_reshape)
                    # sim = cos(emb_i[:, shortest, :], emb_j[:, shortest, :])
                    single_ent[j2] = sim.item() * w1 / math.exp(abs(w1 - w2))
                ent_sim[i2] = max(single_ent) if single_ent else 0
            cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0
        sim_matrix[i1] = rescale(cat_sim)

    with open("data/pickles/category_similarity_matrix.pickle", "wb") as f:
        pickle.dump(sim_matrix, f)


def compare_with_top_categories(categories, top_categories, selected):
    start_time = time.time()
    print("Unpickling…")
    with open("data/pickles/all_entity_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)
    print("Unpickled!")

    no_categories = categories.shape[0]
    no_top_categories = top_categories.shape[0]
    sim_matrix = np.zeros([no_categories, no_top_categories])

    for i1 in categories.index:
        if not categories["category"][i1] in selected:
            continue
        print("-" * 100)
        print(categories["category"][i1])
        cat_sim = [0] * no_top_categories
        for j1 in top_categories.index:
            print("Comparing category", i1, "with category", j1, "…")
            len_i = len(categories["entities"][i1])
            len_j = len(top_categories["entities"][j1])
            ent_sim = [None] * len_i
            for i2 in range(0, len_i):
                w1 = calculate_entity_weight(categories, i1, i2)
                ent_i = categories["entities"][i1][i2][1]
                emb_i = [x["embedding"] for x in embeddings if x["entity"] == ent_i][0]
                single_ent = [None] * len_j
                for j2 in range(0, len_j):
                    t1 = time.time()
                    w2 = calculate_entity_weight(top_categories, j1, j2)
                    print("Entity weights", (time.time() - t1) * 10000)
                    t1 = time.time()
                    ent_j = top_categories["entities"][j1][j2][1]
                    print("Ent:", (time.time() - t1) * 10000)
                    t2 = time.time()
                    # TODO: Extremt långsam, potentiellt ändra data struktur till hashmap/tree
                    emb_j = [
                        x["embedding"] for x in embeddings if x["entity"] == ent_j
                    ][0]
                    print("Emb:", (time.time() - t2) * 10000)

                    shortest = range(0, min(emb_i.shape[1], emb_j.shape[1]))
                    emb_i_reshape = torch.reshape(emb_i[:, shortest, :], (-1,))
                    emb_j_reshape = torch.reshape(emb_j[:, shortest, :], (-1,))

                    sim = cos(emb_i_reshape, emb_j_reshape)
                    single_ent[j2] = sim.item() * w1 / math.exp(abs(w1 - w2))

                ent_sim[i2] = max(single_ent) if single_ent else 0
            cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0
        sim_matrix[i1] = rescale(cat_sim)

    with open("data/pickles/category_similarity_matrix.pickle", "wb") as f:
        pickle.dump(sim_matrix, f)
    print("--- %s seconds ---" % (time.time() - start_time))


def load_and_print_top_similarities(categories, top_categories, selected):
    with open("data/pickles/category_similarity_matrix.pickle", "rb") as f:
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

selected = [
    "Apple",
    "Facebook",
    "Greentech",
    "Forskning",
    "Teknik",
    "Fintech",
    "E-handel",
    "Pandemi",
    "Coronaviruset",
    "Reklam",
    "TV4",
    "SVT",
    "Fastighetsaffärer",
    "Victoria Park Fastigheter",
    "Klarna",
    "Nordea",
    "SEB",
    "Miljöaktuellt",
    "Spotify",
    "EU",
    "Eu",
]

categories = read_df_from_file("data/dataframes/categories_df.jsonl")
top_categories = read_df_from_file("data/dataframes/top_categories_df.jsonl")
# create_entity_embeddings(categories)
# compare_with_all_categories(categories)

# create_entity_embeddings_all()
compare_with_top_categories(categories, top_categories, selected)


top_scores = load_and_print_top_similarities(categories, top_categories, selected)

top_ents = top_categories["no_uses"].values.tolist()
top_categories_plots(top_scores, top_ents)

"""
TODO
Möjliga fortsättnignar:
    1. Bestämma ett tröskelvärde för när kategorier är tillräckligt lika för att sammanslås
    2. Jämföra alla kategorier med enbart MittMedias toppkategorier, d.v.s. RYF-XXX
"""
