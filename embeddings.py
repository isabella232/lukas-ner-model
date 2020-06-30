from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from scipy.spatial.distance import cosine
import numpy as np
from wmd import WMD
import nltk
from nltk.corpus import stopwords
from collections import Counter
from parse_articles import get_articles
from sklearn.feature_extraction.text import TfidfVectorizer
from help_functions import write_df_to_file, read_df_from_file
import pickle


def create_embedding(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[
        0
    ]  # The last hidden-state is the first element of the output tuple
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
    entities = read_df_from_file("data/merged_entities_df.jsonl")
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

    with open("data/entity_embeddings.pickle", "wb") as f:
        pickle.dump(embeddings_list, f)


def rescale(vs):
    mn = min(vs)
    mx = max(vs)
    denominator = mx - mn
    return [(v - mn) / denominator for v in vs]


def filter_out_infrequent(df):
    df = df[df["no_uses"] > 2]
    return df  # .reset_index(drop=True)


def calculate_entity_weight(df, i1, i2):
    return df["entities"][i1][i2][0] / df["tot_no_entities"][i1]


tokenizer = BertTokenizer.from_pretrained("KB/bert-base-swedish-cased-ner")
model = BertModel.from_pretrained("KB/bert-base-swedish-cased-ner")
cos = nn.CosineSimilarity()

categories = read_df_from_file("data/categories_df.jsonl")
# create_entity_embeddings(categories)

print("Unpickling…")
with open("data/entity_embeddings.pickle", "rb") as f:
    embeddings = pickle.load(f)
print("Unpickled!")

categories = filter_out_infrequent(categories)
embeddings = list(map(embeddings.__getitem__, categories.index.tolist()))
categories = categories.reset_index(drop=True)
write_df_to_file(categories, "data/filtered_cats_df.jsonl")
no_categories = categories.shape[0]
sim_matrix = np.zeros([no_categories, no_categories])

for i1 in categories.index:
    cat_sim = [None] * len(categories.index)
    for j1 in categories.index:
        len_i = len(categories["entities"][i1])
        len_j = len(categories["entities"][j1])
        print("Comparing category", i1, "with category", j1, "…")
        ent_sim = [None] * len_i * len_j
        for i2 in range(0, len(categories["entities"][i1])):
            w1 = calculate_entity_weight(categories, i1, i2)
            emb_i = embeddings[i1][i2].item()
            for j2 in range(0, len(categories["entities"][j1])):
                # w2 = calculate_entity_weight(categories, j1, j2)
                emb_j = embeddings[j1][j2].item()
                smallest = range(0, min(emb_i.shape[1], emb_j.shape[1]))
                sim = cos(emb_i[:, smallest, :], emb_j[:, smallest, :])
                # print(sim.mean())
                curr_ind = i2 * len_j + j2
                ent_sim[curr_ind] = sim.mean().item() * w1
        cat_sim[j1] = sum(ent_sim) / len(ent_sim) if ent_sim else 0
    sim_matrix[i1] = rescale(cat_sim)

with open("data/category_similarity_matrix.pickle", "wb") as f:
    pickle.dump(sim_matrix, f)

# with open("data/category_similarity_matrix.pickle", "rb") as f:
#     sim_matrix = pickle.load(f)

# max_val = np.fliplr(np.partition(sim_matrix, -3)[:, -3:])
# max_ind = np.fliplr(np.argpartition(sim_matrix, -5)[:, -5:])

# for ind in categories.index[0:1]:
#     maxs = [
#         (categories["category"][n], categories["no_unique_entities"][n])
#         for n in max_ind[ind]
#     ]
#     print(categories["category"][ind], "has largest similarity with", maxs)

# TODO: Frekeventa entiteter har FÖR LITE impact på den slutigiltiga likheten
"""Hypoteser:   1. viktningen är icke-representativ
                2. mean av tensor är ett dåligt sätt att erhålla en skalär för entiteslikhet
                3. mean av entititlikheter är ett dåligt sätt att erhålla den slutgiltiga artikellikheten"""

# TODO: en viss andel av entitetslikheter bör överstiga något tröskelvärde
# TODO: dela upp utifrån entitetstyper

# Using two weights:
# AB Volvo has largest similarity with [('Soundcloud', 3), ('Drograttfylleri', 6), ('Omsorg', 8), ('Fordonsbrand', 10), ('Kry', 8)]
