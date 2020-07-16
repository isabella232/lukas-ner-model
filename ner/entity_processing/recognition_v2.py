import time
import re
import json
import random
from string import punctuation

import jsonlines
from transformers import pipeline

from ..utils.file_handling import write_output_to_file


def get_articles(path):
    # with open(path, "r") as f:
    #     articles = json.load(f)

    # return [article for article in articles]

    with jsonlines.open(path) as reader:
        obj_list = [obj for obj in reader]

    return obj_list


def avg_score(entity):
    return sum(entity["score"]) / len(entity["score"])


def handle_ambiguity(previous, current):
    if avg_score(previous) < current["score"]:
        previous["entity"] = current["entity"]


def format_entities(raw_entities, text):
    chars = set(punctuation)
    chars.update("’")
    is_char = lambda x: x in chars

    formatted_entities = []

    for current in raw_entities:
        # Remove unknown tokens
        if "[UNK]" in current["word"]:
            current["word"] = current["word"].replace("[UNK]", "").strip()

        # Ignore empty tokens
        if not current["word"]:
            current["entity"] = "NA"
            continue

        if formatted_entities:
            previous = formatted_entities[-1]
            adjacent = previous["index"] == current["index"] - 1
            same_entity = previous["entity"] == current["entity"]

        is_per_or_loc = current["entity"] == "PER" or current["entity"] == "LOC"

        # Handle subwords
        if formatted_entities and current["word"].startswith("##"):
            if adjacent:
                # Handle subwords that are of different entity types
                if not same_entity:
                    handle_ambiguity(previous, current)

                previous["index"] = current["index"]
                previous["word"] += current["word"][2:]
                previous["score"] += [current["score"]]

            # Ignore subwords that do not have a starting part
            else:
                current["entity"] = "NA"

        # Handle entities that consist of multiple words
        elif formatted_entities and adjacent and same_entity:
            # Persons and locations do not have "," or "och" in their names
            if is_per_or_loc and (current["word"] == "," or current["word"] == "och"):
                current["entity"] = "NA"
                continue

            previous["index"] = current["index"]
            either_is_char = is_char(previous["word"][-1]) or is_char(current["word"])

            # Determine if the word suffix should be preceded by a space
            suffix = (
                current["word"]
                if either_is_char and not current["word"] == "och"
                else " " + current["word"]
            )

            previous["word"] += suffix
            previous["score"] += [current["score"]]

        # Ignore single characters and "s"
        elif is_char(current["word"]) or current["word"] == "s":
            current["word"] = "NA"

        # Handle trivial entities
        else:
            current["score"] = [current["score"]]
            formatted_entities += [current.copy()]

    for entity in formatted_entities:
        entity["score"] = avg_score(entity)
        del entity["index"]

    return formatted_entities


def validate_scores(entities):
    for entity in entities:
        for score in entity["score"]:
            if score > 1:
                print("Score larger than 1.0 for:", entity)
                exit()


model_name = "KB/bert-base-swedish-cased-ner"
# Can now use grouped_entities=True to auto group tokens/words into entities
nlp = pipeline("ner", model=model_name, tokenizer=model_name)

articles = get_articles("data/input/articles_10k.json")


indexes = random.sample(range(0, len(articles) - 1), 10)
print(indexes)
# articles = [article for i, article in enumerate(articles) if i in indexes]

json_output = []
omitted_articles = []

for i, article in enumerate(articles[9707:9709]):
    print("Processing article", i, "…")

    text = article["content_text"]
    sentences = text.replace("\n\n", ".").split(".")
    entities = []

    for sentence in sentences:
        if re.search("<.*>", sentence):
            omitted_articles += [article]
            break
        elif not sentence.strip():
            continue

        try:
            input_sentence = sentence.strip() + "."
            sentence_entities = nlp(input_sentence)
            entities += sentence_entities
        except IndexError:  # 1541 max length for input sentence
            omitted_articles += [article]
            continue

    [print(ent) for ent in entities]
    print("-" * 100)
    formatted_entities = format_entities(entities, text)
    json_output += [{"article": article, "entities": formatted_entities}]

    [print(ent) for ent in formatted_entities]
    # validate_scores(formatted_entities)

# write_output_to_file(json_output, "data/output/results_10k.jsonl")
# write_output_to_file(omitted_articles, "data/output/omitted_10k.jsonl")
