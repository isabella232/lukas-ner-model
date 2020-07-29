import re

import jsonlines
import pandas as pd

from ner.utils.file_handling import write_df_to_file


def get_articles(path):
    with jsonlines.open(path) as reader:
        obj_list = [obj for obj in reader]

    return obj_list


def transform(articles, is_mm, mapping):
    transformed = []

    for article in articles:
        aid = article["id"]

        if is_mm:
            brand = "MM"
            text = article["content_text"]
            if re.search("<.*>", text):
                continue

            categories = [
                mapping[tag["category"]]
                for tag in article["tags"]
                if tag["category"].startswith("RYF-") and len(tag["category"]) == 7
            ]

        else:
            brand = "TT"
            text = article["text"]
            categories = [
                category[0]
                for category in article["categories"]
                if category[0].endswith("000000")
            ]

        transformed += [
            {"aid": aid, "brand": brand, "text": text, "categories": categories}
        ]

    return pd.DataFrame(transformed)


def check_distribution(df):
    df = df.explode("categories")

    count = df.groupby("categories")["aid"].count()
    print(count)


if __name__ == "__main__":
    seed = 1234567890

    iptc_codes = [f"{i:02d}000000" for i in range(1, 18)]
    mm_codes = [
        "RYF-XKI",
        "RYF-BIZ",
        "RYF-AKM",
        "RYF-IXA",
        "RYF-YDR",
        "RYF-VHD",
        "RYF-WUU",
        "RYF-CUW",
        "RYF-ZEI",
        "RYF-TKT",
        "RYF-KNI",
        "RYF-ITV",
        "RYF-EOI",
        "RYF-HPT",
        "RYF-QPR",
        "RYF-WXT",
        "RYF-WHZ",
    ]
    mapping = dict(zip(mm_codes, iptc_codes))

    tt_articles = get_articles("data/input/articles_tt_new.jsonl")
    mm_articles = get_articles("data/input/articles_mittmedia_10k.json")

    tt_df = transform(tt_articles, False, mapping)
    mm_df = transform(mm_articles, True, mapping)

    df = pd.DataFrame(columns=["aid", "brand", "text", "categories"])
    for code in iptc_codes:
        tt_filt = tt_df[tt_df["categories"].apply(lambda x: code in x)]
        mm_filt = mm_df[mm_df["categories"].apply(lambda x: code in x)]
        print(code, len(tt_filt.index), len(mm_filt.index))
        if len(tt_filt.index) < 150:
            mm_size = 300 - len(tt_filt.index)
            mm_filt = mm_filt.sample(n=mm_size, random_state=seed)
        else:
            tt_filt = tt_filt.sample(n=150, random_state=seed)
            mm_filt = mm_filt.sample(n=150, random_state=seed)
        df = df.append([tt_filt, mm_filt], ignore_index=True)

    df = df.drop_duplicates(subset=["aid"])
    print(df)
    check_distribution(df)

    train = df.sample(frac=0.8, random_state=seed)
    test = df.drop(train.index)

    train.to_csv("mltc/data/train.csv")
    test.to_csv("mltc/data/test.csv")

