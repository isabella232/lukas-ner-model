import requests
import re

from urllib.request import urlopen
import regex

params = {
    "ak": "<key>",
    "q": "riksbanken",
    "p": "TTEKO0",
}
resp = requests.get("https://tt.se/api/search", params=params)

uris = []

for article in resp.json():
    print(article["headline"])
    print(article["uri"])

    try:
        subjects = []
        subjects = [subj["name"] for subj in article["subject"]]
        print(*subjects, sep=" | ")
    except:
        print("No subjects")

    try:
        print(article["associations"]["a001"]["uri"])
        uris += [article["associations"]["a001"]["uri"]]
    except:
        print("No associations")

    print("-" * 100)

for uri in uris:
    try:
        resource = urlopen(uri)
        content = resource.read().decode(resource.headers.get_content_charset())

        text = re.findall("<p>((.|\n)*)<\/p>", content)[0][0]
        print(text.strip())
        print("-" * 100)
    except:
        print("Something went wrong…")
    # clean_text = re.findall("\r\n.*\r\n", text)[0].strip()
    # print(clean_text)

