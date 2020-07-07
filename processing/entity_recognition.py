from transformers import pipeline
import re
import json
import jsonlines

from utils.file_handling import write_output_to_file
import time


def get_articles(path):
    with open(path, "r") as f:
        articles = json.load(f)

    return [article for article in articles]

    # with jsonlines.open(path) as reader:
    #     obj_list = [obj for obj in reader]

    # return obj_list


def handle_ambiguity(current, previous):
    if not isinstance(previous["entity"], list):
        previous["entity"] = [previous["entity"]]

    previous["entity"] += [{"type": current["entity"], "subtoken": current["word"]}]


def format_entities(raw_entities, text):

    is_char = lambda x: x in ["-", ",", '"', "'", "’", "#"]

    formatted_entities = []
    no_subtokens = 1

    for i, current in enumerate(raw_entities):
        if "[UNK]" in current["word"]:
            current["word"] = current["word"].replace("[UNK]", "").strip()

        if not current["word"]:
            current["entity"] = "NA"
            continue

        if formatted_entities:
            previous = formatted_entities[-1]
            adjacent = previous["index"] == current["index"] - 1
            same_entity = previous["entity"] == current["entity"]

        # End of article
        eoa = i + 1 >= len(raw_entities)

        if not eoa:
            following = raw_entities[i + 1]
            # End of sentence
            eos = following["index"] < current["index"]

            if current["word"] == ":" and following["word"] == "s":
                following["word"] = ""
                no_subtokens = 1
                continue

        if formatted_entities and current["word"].startswith("##"):
            if adjacent:
                previous["index"] = current["index"]
                previous["word"] += current["word"][2:]
                previous["score"][-1] += current["score"]
                no_subtokens += 1

                if not same_entity:
                    handle_ambiguity(current, previous)

            else:
                current["entity"] = "NA"

        elif formatted_entities and adjacent and same_entity:
            previous["index"] = current["index"]
            either_is_char = is_char(current["word"]) or is_char(previous["word"][-1])

            if either_is_char and not current["word"] == "och":
                previous["word"] += current["word"]
            else:
                previous["word"] += " " + current["word"]

            previous["score"] += [current["score"]]
            no_subtokens = 1

        else:
            current["score"] = [current["score"]]
            formatted_entities += [current.copy()]
            no_subtokens = 1

        if not eoa:
            # End of entity series
            eoes = no_subtokens > 1 and not following["word"].startswith("##")

        if (eoa or eos) or eoes:
            avg_score = formatted_entities[-1]["score"][-1] / no_subtokens
            formatted_entities[-1]["score"][-1] = avg_score
            no_subtokens = 1

    return formatted_entities


def validate_scores(entities):
    for entity in entities:
        for score in entity["score"]:
            if score > 1:
                print("Score larger than 1.0 for:", entity)
                exit()


model = "KB/bert-base-swedish-cased-ner"
tokenizer = "KB/bert-base-swedish-cased-ner"
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

articles = get_articles("data/input/articles_small.json")

json_output = []
omitted_articles = []

for i, article in enumerate(articles):
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

    formatted_entities = format_entities(entities, text)
    json_output += [{"article": article, "entities": formatted_entities}]

    validate_scores(formatted_entities)

# write_output_to_file(json_output, "data/output/results_10k.jsonl")
# write_output_to_file(omitted_articles, "data/output/omitted_10k.jsonl")
