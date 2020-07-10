from string import punctuation

from transformers import pipeline

from .evaluator import Evaluator
from ..utils.file_handling import write_output_to_file

print("Preprocessing…")
model = "KB/bert-base-swedish-cased-ner"
tokenizer = "KB/bert-base-swedish-cased-ner"
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

evaluator = Evaluator(False)

all_sentences, all_tags = evaluator.load_corpus()
sentences, tags = evaluator.prepare_for_evaluation(all_sentences, all_tags, 1.0)

print("Extracting entities…")
entities = [nlp(sentence) for sentence in sentences]

punct = set(punctuation)
all_types = evaluator.desired + ["TME", "MSR", "WRK", "EVN", "OBJ"]
formatted = []

print("Formatting entities…")
for ents in entities:
    sen_form = []
    for token in ents:
        if token["entity"] not in all_types:
            print(token["entity"], token["word"])

        faulty_token = token["word"] == "[CLS]" or token["word"] == "[UNK]"
        if faulty_token or token["entity"] not in evaluator.desired:
            continue

        if token["word"].startswith("##"):
            if not sen_form:
                continue

            same_type = token["entity"] == sen_form[-1]["entity"]
            current_larger = token["score"] > sen_form[-1]["score"]

            if not same_type and current_larger:
                sen_form[-1]["entity"] = token["entity"]

            sen_form[-1]["word"] += token["word"][2:]
            sen_form[-1]["score"] += token["score"]
            sen_form[-1]["score"] = sen_form[-1]["score"] / 2

        elif len(sen_form) > 2 and sen_form[-1]["word"] in punct:
            sen_form[-2]["word"] += sen_form[-1]["word"] + token["word"]
            sen_form[-2]["score"] += sen_form[-1]["score"] + token["score"]
            sen_form[-2]["score"] = sen_form[-2]["score"] / 3
            del sen_form[-1]

        else:
            sen_form += [token]

    formatted += [sen_form]

write_output_to_file(formatted, "data/output/bert_evaluation_v2.jsonl")

entities = evaluator.get_results("data/output/bert_evaluation_v2.jsonl")
per, org, loc = evaluator.evaluate(entities, tags)

per_metrics = evaluator.calculate_metrics(per, "PER")
org_metrics = evaluator.calculate_metrics(org, "ORG")
loc_metrics = evaluator.calculate_metrics(loc, "LOC")


# test_sentence = 'Den är tre timmar lång och släpps ut samtidigt med boken , " In the Arena " , i april .'
# test_sentence = "( TT )"
# results = nlp(test_sentence)
# [print(entity) for entity in results]

