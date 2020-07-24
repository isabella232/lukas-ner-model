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


def group_entities(raw_entities, punct):
    is_punct = lambda x: x in punct
    grouped_entities = []

    for current in raw_entities:
        # Remove unknown tokens
        if "[UNK]" in current["word"]:
            current["word"] = current["word"].replace("[UNK]", "").strip()

        # Ignore empty tokens
        if not current["word"]:
            current["entity"] = "NA"
            continue

        if grouped_entities:
            previous = grouped_entities[-1]
            adjacent = previous["index"] == current["index"] - 1
            same_entity = previous["entity"] == current["entity"]

        is_per_or_loc = current["entity"] == "PER" or current["entity"] == "LOC"

        # Handle subwords
        if grouped_entities and current["word"].startswith("##"):
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
        elif grouped_entities and adjacent and same_entity:
            # Ignore persons and locations that have "," or "och" in their names
            if is_per_or_loc and (current["word"] == "," or current["word"] == "och"):
                current["entity"] = "NA"
                continue

            previous["index"] = current["index"]
            either_punct = is_punct(previous["word"][-1]) or is_punct(current["word"])

            # Determine if the word suffix should be preceded by a space
            suffix = (
                current["word"]
                if either_punct and not current["word"] == "och"
                else " " + current["word"]
            )

            previous["word"] += suffix
            previous["score"] += [current["score"]]

        # Ignore single characters and "s"
        elif is_punct(current["word"]) or current["word"] == "s":
            current["word"] = "NA"

        # Handle trivial entities
        else:
            current["score"] = [current["score"]]
            grouped_entities += [current.copy()]

    for entity in grouped_entities:
        entity["score"] = avg_score(entity)
        del entity["index"]

    return grouped_entities


def validate_scores(entities):
    for entity in entities:
        if entity["score"] > 1.0:
            print("Score larger than 1.0 for:", entity)
            exit()


def recognize_entities(articles):
    """
    Possible to use parameter grouped_entities=True in pipeline to auto-group
    tokens/words into entities as of pr #3957 in the transformers repo. However,
    it does not work as well (yet, 2020-07) as the group_entities function above.
    """
    model_name = "KB/bert-base-swedish-cased-ner"
    nlp = pipeline("ner", model=model_name, tokenizer=model_name)

    # indexes = random.sample(range(0, len(articles) - 1), 10)
    # articles = [article for i, article in enumerate(articles) if i in indexes]

    punct = set(punctuation)
    punct.update("’")
    all_entities = []
    omitted_articles = []

    for i, article in enumerate(articles):
        print("Processing article", i, "…")

        text = article["text"].replace("\n\n", ".")
        sentences = re.findall(".*?[.?!]", text)
        entities = []

        for sentence in sentences:
            # Omit any article containing HTML tags
            if re.search("<.*>", sentence):
                omitted_articles += [article]
                break

            input_sentence = sentence.strip()
            if not input_sentence or input_sentence in punct:
                continue

            try:
                sentence_entities = nlp(input_sentence)
                entities += sentence_entities
            except IndexError:  # 1541 max length for input sentence
                omitted_articles += [article]
                break

        grouped_entities = group_entities(entities, punct)
        all_entities += [{"article": article, "entities": grouped_entities}]

    # validate_scores(grouped_entities)

    return all_entities, omitted_articles


if __name__ == "__main__":
    articles = get_articles("data/input/articles_tt_new.jsonl")
    entities, omitted = recognize_entities(articles)

    write_output_to_file(entities, "data/output/results_tt_new.jsonl")
    write_output_to_file(omitted, "data/output/omitted_tt_new.jsonl")
