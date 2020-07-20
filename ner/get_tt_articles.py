import requests
import re

from urllib.request import urlopen
from urllib.error import HTTPError

from .utils.file_handling import write_output_to_file


def get_subject_articles(subject_id, products):
    params = {
        "ak": "<key>",
        "q": f"{subject_id:02d}000000",
        "p": ",".join(products),
        "trs": "2018-01-01",
        "tre": "2020-07-20",
        "s": 1,
    }

    resp = requests.get("https://tt.se/api/search", params=params)
    print("Response size:", len(resp.json()))

    articles = []
    for article in resp.json():
        aid = article["originaltransmissionreference"]
        products = [product["code"] for product in article["product"]]
        uri = article["uri"] + "-cutpaste.txt"

        categories = []
        text = ""

        try:
            categories = [subj["name"] for subj in article["subject"]]

            resource = urlopen(uri)
            raw_text = resource.read().decode("utf-8")
            sentences = raw_text.split("\r\n")[2:]

            for sentence in sentences:
                if not sentence.endswith("TT") and not sentence.startswith("http"):
                    text += sentence.strip() + " "
                else:
                    break
        except HTTPError:
            continue

        articles += [
            {"id": aid, "products": products, "categories": categories, "text": text}
        ]

    return articles


products = [
    "TTKUL",
    "TTINR",
    "TTUTR",
    "TTSPT",
    "TTEKO",
    "TTREC",
    "TTNOJ",
    "TTVDG",
    "FTBOS",
    "FTDFD",
    "FTMOT",
    "FTRES",
    "FTRESPLS",
    "FTHOL",
    "FTTGF",
    "FTKOT",
]

all_articles = []
for subject_id in range(1, 18):
    print(subject_id)

    articles = get_subject_articles(subject_id, products)
    all_articles += articles
    [print(article["categories"]) for article in articles]

# write_output_to_file(all_articles, "data/input/articles_tt.jsonl")
