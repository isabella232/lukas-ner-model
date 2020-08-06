import re

import jsonlines
import pandas as pd


def get_articles(path):
    with jsonlines.open(path) as reader:
        obj_list = [obj for obj in reader]

    return obj_list


def get_category_codes(is_top):
    if is_top:
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
        cat_dict = dict(zip(list(mapping.values()), [0] * len(mapping)))
    else:
        iptc_codes = ["20000005", "20000013", "20000018", "20000029", "20000051"]
        mm_codes = [
            "RYF-XKI-JIJ",
            "RYF-XKI-YFJ",
            "RYF-XKI-FEY",
            "RYF-XKI-GJH",
            "RYF-XKI-BUS",
        ]
        mapping = dict(zip(iptc_codes, mm_codes))

        mm_codes += [
            "RYF-XKI-DEG",
            "RYF-XKI-YBJ",
            "RYF-XKI-LNE",
            "RYF-XKI-HLO",
            "RYF-XKI-FXL",
            "RYF-XKI-WEG",
            "RYF-XKI-ISL",
            "RYF-XKI-KKX",
            "RYF-XKI-IUV",
            "RYF-XKI-SFU",
            "RYF-XKI-IHA",
            "RYF-XKI-TMS",
            "RYF-XKI-JUJ",
            "RYF-XKI-PGP",
            "RYF-XKI-VME",
            "RYF-XKI-CDA",
        ]
        cat_dict = dict(zip(list(mm_codes), [0] * len(mm_codes)))

    return set(iptc_codes), set(mm_codes), mapping, cat_dict


def transform(
    articles, is_mm, use_iptc, mapping, cat_dict, iptc_codes=set(), mm_codes=set()
):
    transformed = []
    for article in articles:
        aid = article["id"]
        categories = cat_dict.copy()
        # print(categories)

        if is_mm:
            brand = "MM"
            text = article["content_text"]
            if re.search("<.*>", text):
                continue

            for tag in article["tags"]:
                category = tag["category"]
                if category in mm_codes:
                    cat = mapping[category] if use_iptc else category
                    categories[cat] = 1

        else:
            brand = "TT"
            text = article["text"]
            for category in article["categories"]:
                if category[0] in iptc_codes:
                    cat = category[0] if use_iptc else mapping[category[0]]
                    categories[cat] = 1

        if not text or sum(categories.values()) == 0:
            continue

        transformed += [{"aid": aid, "brand": brand, "text": text, **categories}]

    return pd.DataFrame(transformed)


def smooth_category_distribution(tt_df, mm_df, iptc_codes):
    df = pd.DataFrame()
    for code in iptc_codes:
        tt_filt = tt_df[tt_df[code] == 1]
        mm_filt = mm_df[mm_df[code] == 1]
        # print(code, len(tt_filt.index), len(mm_filt.index))
        if len(tt_filt.index) < 150:
            mm_size = 300 - len(tt_filt.index)
            mm_filt = mm_filt.sample(n=mm_size, random_state=seed)
        else:
            tt_filt = tt_filt.sample(n=150, random_state=seed)
            mm_filt = mm_filt.sample(n=150, random_state=seed)
        df = df.append([tt_filt, mm_filt], ignore_index=True)
    return df


def check_category_distribution(df):
    for i in df.columns[3:]:
        filt = df[df[i] == 1]
        occurrence = filt[i].count()
        co_occurrence = filt.iloc[:, 3:].sum(axis=1) - 1
        single_occurrence = co_occurrence[co_occurrence == 0].count() / occurrence
        print(
            f"Category {i}: total occurrence = {occurrence}, average co occurrence = {co_occurrence.mean()} and share of single occurrences {single_occurrence}"
        )


if __name__ == "__main__":
    seed = 1234567890

    names = [
        "Konst, kultur och nöje",
        "Brott, lag och rätt",
        "Katastrofer och olyckor",
        "Ekonomi, affärer och finans",
        "Utbildning",
        "Miljö och natur",
        "Medicin och hälsa",
        "Mänskligt",
        "Arbete",
        "Fritid och livsstil",
        "Politik",
        "Etik och religion",
        "Teknik och vetenskap",
        "Samhälle",
        "Sport",
        "Krig, konflikter och oroligheter",
        "Väder",
    ]

    tt_articles = get_articles("data/input/articles_tt_culture.jsonl")
    mm_articles = get_articles("data/input/articles_mittmedia_culture.json")

    is_top = False
    iptc_codes, mm_codes, mapping, cat_dict = get_category_codes(is_top)

    tt_df = transform(
        tt_articles, False, is_top, mapping, cat_dict, iptc_codes, mm_codes
    )
    mm_df = transform(
        mm_articles, True, is_top, mapping, cat_dict, iptc_codes, mm_codes
    )

    # df = smooth_category_distribution(tt_df, mm_df)

    df = tt_df.append(mm_df, ignore_index=True)
    df = df.drop_duplicates(subset=["aid"])
    check_category_distribution(df)

    print(f"Number of articles {df.shape[0]}")

    train = df.sample(frac=0.85, random_state=seed)
    test = df.drop(train.index)

    # print(train)

    train.to_csv("mltc/data/datasets/train_culture.csv", index=False)
    test.to_csv("mltc/data/datasets/test_culture.csv", index=False)
