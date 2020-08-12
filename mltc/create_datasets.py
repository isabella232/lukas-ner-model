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


def process_mm_subcategories():
    articles = get_articles("data/input/articles_mittmedia_subcategories.json")

    codes = [
        "RYF-TKT-QWJ",
        "RYF-QPR-HGB",
        "RYF-KNI-RLW",
        "RYF-TKT-ZWD",
        "RYF-KNI-ZIS",
        "RYF-EOI-DLM",
        "RYF-VHD-QTT",
        "RYF-XKI-DEG",
        "RYF-TKT-MRA",
        "RYF-TKT-RRC",
        "RYF-VHD-IFE",
        "RYF-CUW-XVL",
        "RYF-BIZ-XSV",
        "RYF-TKT-SPE",
        "RYF-WUU-QNB",
        "RYF-IXA-KEV",
        "RYF-IXA-LHV",
        "RYF-KNI-XNS",
        "RYF-TKT-GSR",
        "RYF-TKT-PEJ",
        "RYF-XKI-IHA",
        "RYF-TKT-HFA",
        "RYF-AKM-TYE",
        "RYF-XKI-JIJ",
        "RYF-XKI-FEY",
        "RYF-CUW-GIV",
        "RYF-AKM-AUE",
        "RYF-TKT-MNX",
        "RYF-TKT-VVD",
        "RYF-XKI-HLO",
        "RYF-BIZ-GCG",
        "RYF-XKI-SFU",
        "RYF-HPT-ZSS",
        "RYF-KNI-OHT",
        "RYF-BIZ-WZJ",
        "RYF-AKM-QGJ",
        "RYF-HPT-YGE",
        "RYF-HPT-XRQ",
        "RYF-CUW-LGT",
        "RYF-WXT-BND",
        "RYF-HPT-PFF",
        "RYF-XKI-GJH",
        "RYF-CUW-JCL",
        "RYF-TKT-AGB",
        "RYF-VHD-ZKU",
        "RYF-ZEI-QWT",
        "RYF-YDR-MHH",
        "RYF-VHD-OAY",
        "RYF-TKT-FGO",
        "RYF-EOI-DOE",
        "RYF-XKI-ISL",
        "RYF-KNI-MRR",
        "RYF-TKT-ESA",
        "RYF-TKT-YIG",
        "RYF-XKI-YFJ",
        "RYF-IXA-CVI",
        "RYF-XKI-LNE",
        "RYF-XKI-WEG",
        "RYF-YDR-UER",
        "RYF-IXA-QED",
        "RYF-EOI-GLR",
        "RYF-XKI-FXL",
        "RYF-ITV-LPR",
        "RYF-TKT-ESN",
        "RYF-VHD-GUZ",
        "RYF-YDR-WXH",
        "RYF-CUW-IAA",
        "RYF-HPT-XAO",
        "RYF-XKI-BUS",
        "RYF-ITV-IQC",
        "RYF-TKT-INK",
        "RYF-ITV-QAK",
        "RYF-TKT-PYI",
        "RYF-WUU-SUX",
        "RYF-HPT-THK",
        "RYF-CUW-AEJ",
        "RYF-XKI-YBJ",
        "RYF-TKT-EPN",
        "RYF-ZEI-RYD",
        "RYF-WUU-JZK",
        "RYF-KNI-SQH",
        "RYF-EOI-GJM",
        "RYF-ZEI-CLV",
        "RYF-WHZ-LAW",
    ]
    cat_dict = dict(zip(list(codes), [0] * len(codes)))

    _, _, mapping, cat_dict_top = get_category_codes(True)

    cat_dict = {**cat_dict, **cat_dict_top}

    transformed = []
    for article in articles:
        aid = article["identifier"]["id"]
        categories = cat_dict.copy()
        brand = "MM"
        text = article["content_text"]
        if re.search("<.*>", text):
            continue

        for tag in article["tags"]:
            category = tag["category"]
            categories[category] = 1
            categories[mapping[category[:-4]]] = 1
        if not text or sum(categories.values()) == 0:
            continue

        transformed += [{"aid": aid, "brand": brand, "text": text, **categories}]

    df = pd.DataFrame(transformed)
    df = df.drop_duplicates(subset=["aid"])

    pruned_df = None
    seed = 1234567890
    for i in df.columns[3:-17]:
        filt = df[df[i] == 1].copy()
        if filt[i].count() > 500:
            filt = filt.sample(500, random_state=seed)
        if pruned_df is None:
            pruned_df = filt.reset_index(drop=True)
            print(pruned_df)
        else:
            pruned_df = pruned_df.append(filt, ignore_index=True)

    pruned_df = pruned_df.drop_duplicates(subset=["aid"])
    check_category_distribution(pruned_df)
    print(pruned_df.shape[0])

    train = df.sample(frac=0.85, random_state=seed)
    test = df.drop(train.index)

    split_and_save(train, "train_sub.csv", "train_sub_parent_labels.csv")
    split_and_save(test, "test_sub.csv", "test_sub_parent_labels.csv")


def split_and_save(df, file_1, file_2):
    top = df.iloc[:, -17:].copy()
    top["aid"] = df["aid"].values
    cols = top.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    top = top[cols]

    df.drop(df.columns[-17:], axis=1, inplace=True)

    df.to_csv("mltc/data/datasets/" + file_1, index=False)
    top.to_csv("mltc/data/datasets/" + file_2, index=False)


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

    process_mm_subcategories()

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
