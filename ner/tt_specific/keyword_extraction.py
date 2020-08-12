import json

import torch
from torch import nn
import pandas as pd
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

from ..utils.file_handling import read_df_from_file


def get_iptc_all(path):
    with open(path, "r") as f:
        categories = json.load(f)

    return [category for category in categories["conceptSet"]]


def get_iptc_tt(path):
    with open(path, "r") as f:
        categories = json.load(f)

    codes = [category["code"] for category in categories]
    return set(codes)


def create_embedding(word):
    try:
        input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    except ValueError:
        print(word)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]

    return last_hidden_states


def create_lookup():
    iptc_all = get_iptc_all("data/iptc_all.json")
    iptc_tt = get_iptc_tt("data/iptc_tt.json")

    top_level = lambda x: x.endswith("000000")
    lookup = []

    for category in iptc_all:
        name = category["prefLabel"]

        if not name:
            continue

        try:
            name = name["se"]
        except KeyError:
            pass

        code = category["qcode"].replace("medtop:", "")
        if code in iptc_tt:
            print(name)
        tt = True if code in iptc_tt else False

        embedding = create_embedding(name)
        temp = {"code": code, "name": name, "embedding": embedding, "tt": tt}

        if not top_level(code):
            broader = category["broader"][0].split("/")[-1]
            temp["broader"] = broader

        lookup += [temp]

    with open("data/pickles/iptc_embeddings.pickle", "wb") as f:
        torch.save(lookup, f)


def find_nearest_tt(cat):
    if not cat["tt"]:
        code = cat["broader"]
        cat = [cat for cat in lookup if cat["code"] == code][0]
        return find_nearest_tt(cat)
    else:
        return cat


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

    return sim.item()


def category_match(keyword):
    kw = create_embedding(keyword)
    similarities = []

    for category in lookup:
        cat = category["embedding"]
        similarities += [max_similarity(kw, cat)]

        # shortest = range(min(kw.shape[1], cat.shape[1]))
        # kw_reshape = torch.reshape(kw[:, shortest, :], (-1,))
        # cat_reshape = torch.reshape(cat[:, shortest, :], (-1,))
        # similarities += [cos(kw_reshape, cat_reshape).item()]

    max_val = max(similarities)
    max_ind = similarities.index(max_val)

    return lookup[max_ind], max_val


def find_entities(keyword, freq_thresh, uniq_thresh):
    categories = read_df_from_file("data/dataframes/categories_tt_new_df.jsonl")
    lookup = read_df_from_file("data/dataframes/tt_entity_lookup_df.jsonl")
    relevant_entities = []

    filter_series = pd.Series(list(zip(*categories["category"]))[0])
    match = categories[filter_series == keyword["code"]]

    if not match.empty:
        entities = match["entities"].tolist()[0]
        quant = pd.DataFrame([e[0] for e in entities]).quantile(freq_thresh).item()

        for entity in entities:
            frequency = entity[0]
            if frequency > quant:
                occurrences = lookup[lookup["entity"] == entity[1]]["categories"].item()
                tot_freq = sum(occurrences.values())
                uniqueness = frequency / tot_freq
                if uniqueness > uniq_thresh:
                    relevant_entities += [entity[1]]

    return relevant_entities


if __name__ == "__main__":
    model_name = "KB/bert-base-swedish-cased-ner"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    cos = nn.CosineSimilarity(dim=0)

    # create_lookup()

    with open("data/pickles/iptc_embeddings.pickle", "rb") as f:
        lookup = torch.load(f)

    # Must be the name of an IPTC category
    keyword = "Bilar"

    match, sim = category_match(keyword)
    tt_match = find_nearest_tt(match)
    entities = find_entities(tt_match, 0.8, 0.4)
    [print(e) for e in entities]
    print("_" * 50)
    print(tt_match["name"])
