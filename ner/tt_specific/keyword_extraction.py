import json

import pandas as pd

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


def create_lookup():
    """Create a lookup table for all IPTC categories."""
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
        tt = True if code in iptc_tt else False

        temp = {"code": code, "name": name, "tt": tt}

        if not top_level(code):
            broader = category["broader"][0].split("/")[-1]
            temp["broader"] = broader

        lookup += [temp]

    return lookup


def find_nearest_tt(cat):
    """Returns the nearest IPTC category in the tree used by TT.
    
    Example: input "Jazz" => output "Musik"
    """
    if not cat["tt"]:
        code = cat["broader"]
        cat = [cat for cat in lookup if cat["code"] == code][0]
        return find_nearest_tt(cat)
    else:
        return cat


def find_entities(category, freq_thresh, uniq_thresh):
    """Returns the entities/keywords used in a category given the thresholds for frequency and uniqueness."""
    categories = read_df_from_file("data/dataframes/categories_tt_new_df.jsonl")
    lookup = read_df_from_file("data/dataframes/tt_entity_lookup_df.jsonl")
    relevant_entities = []

    filter_series = pd.Series(list(zip(*categories["category"]))[0])
    match = categories[filter_series == category["code"]]

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
    lookup = create_lookup()

    # Must be the name of an IPTC category
    category = "Jazz"
    category = [cat for cat in lookup if cat["name"] == category][0]

    tt_match = find_nearest_tt(category)
    entities = find_entities(tt_match, 0.8, 0.4)
    [print(e) for e in entities]
    print("_" * 50)
    print(tt_match["name"])
