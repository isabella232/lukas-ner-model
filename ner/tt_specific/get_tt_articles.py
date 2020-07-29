import requests
import re
import time

from urllib.request import urlopen
from urllib.error import HTTPError

from ..utils.file_handling import write_output_to_file


def get_subject_articles(subject_id, products, processed_aids):
    payload = {
        "ak": "<key>",
        "q": f"subject.code:{subject_id:02d}000000+language:sv+type:text",
        # "p": ",".join(products),
        # "trs": "2010-01-01",
        # "tre": "2020-07-20",
        "s": 500,
    }
    payload_str = "&".join("%s=%s" % (k, v) for k, v in payload.items())

    resp = requests.get("https://tt.se/api/search", params=payload_str)
    print("Response size:", len(resp.json()))

    articles = []
    for article in resp.json():
        print(article["datetime"])
        try:
            aid = article["originaltransmissionreference"]

            if aid in processed_aids:
                continue

            processed_aids.add(aid)

            products = [product["code"] for product in article["product"]]
            uri = article["uri"] + "-cutpaste.txt"

            categories = []
            text = ""

            categories = [(subj["code"], subj["name"]) for subj in article["subject"]]

            resource = urlopen(uri)
            raw_text = resource.read().decode("utf-8")
            sentences = raw_text.split("\r\n")[2:]

            for sentence in sentences:
                if not sentence.endswith("TT") and not sentence.startswith("http"):
                    text += sentence.strip() + " "
                else:
                    break

            articles += [
                {
                    "id": aid,
                    "products": products,
                    "categories": categories,
                    "text": text,
                }
            ]

        except (HTTPError, KeyError):
            print("Error!")
            pass

    return articles, processed_aids


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
processed_aids = set()
for subject_id in [12]:  # range(1, 18):
    print(subject_id)

    articles, processed_aids = get_subject_articles(
        subject_id, products, processed_aids
    )
    all_articles += articles
    # [print(article["categories"]) for article in articles]

# write_output_to_file(all_articles, "data/input/articles_tt_new.jsonl")
