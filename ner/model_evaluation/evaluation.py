from string import punctuation
import random

import jsonlines
import regex


def load_corpus():
    with open("data/input/ner_corpus.txt") as f:
        lines = f.readlines()
        f.seek(0)
        corpus = f.read()

    entities = [line for line in lines if not line.endswith("\t0\n")]

    return corpus, entities


def format_sentences(corpus):
    corpus = corpus.split("\n\n")
    sentences = []

    for sentence in corpus:
        # sentence = regex.sub("\t.*\n(?=\p{P})", "", sentence)
        sentence = regex.sub("\t.*?\n", " ", sentence)
        sentence = regex.split("\t.*", sentence)[0]
        # sentence = regex.sub('" (?=.*")', ' "', sentence)
        sentences += [sentence]

    return sentences[:-1]


def format_tags(entities):
    new_l = [-1]
    new_l += [i for i, val in enumerate(entities) if val == "\n"]
    grouped = []

    for i in range(1, len(new_l)):
        sen_ent = [entities[j].split("\t") for j in range(new_l[i - 1] + 1, new_l[i])]
        ent_dict = [{"word": e[0], "entity": e[1].strip()} for e in sen_ent]
        grouped += [ent_dict]

    return grouped


def filter_tags(tags):
    # desired = ["B-PRS", "I-PRS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    desired = ["PER", "ORG", "LOC"]
    filtered = []

    for ts in tags:
        filt = [t for t in ts if t["entity"] in desired]
        filtered += [filt] if filt else [[]]

    return filtered


def prepare_for_evaluation(sentences, tags, sample):
    if sample:
        no_sentences = len(sentences)
        eval_size = int(no_sentences * 0.1)
        random.seed(1234567890)
        eval_inds = random.sample(range(0, eval_size), eval_size)

        sentences = [sentences[i] for i in eval_inds]
        tags = [tags[i] for i in eval_inds]

    filtered = filter_tags(tags)

    return sentences, filtered


def get_results(path):
    with jsonlines.open(path, "r") as reader:
        results = [obj for obj in reader]

    return results


# f = found, g = golden standard, w = words, e = entities, i = index, d = dictionary, t = tag
def evaluate_typewise(f_w, f_e, g_w, g_e, d, t):
    n_i = [i for i, x in enumerate(f_e) if x == t]
    g_i = [i for i, x in enumerate(g_e) if x == t]

    f_w = [f_w[i].split() for i in n_i]
    f_w = [w for ws in f_w for w in ws if w not in set(punctuation)]
    g_w = [g_w[i] for i in g_i]

    true_positives = []
    all_positives = f_w
    relevant = g_w.copy()

    for w in f_w:
        if w in g_w:
            true_positives += [w]
            del g_w[g_w.index(w)]

    d["tp"] += true_positives
    d["ap"] += all_positives
    d["rel"] += relevant

    return d


def evaluate(entities, tags):
    # True positives, all positives, relevant
    per = {"tp": [], "ap": [], "rel": []}
    org = {"tp": [], "ap": [], "rel": []}
    loc = {"tp": [], "ap": [], "rel": []}

    for i, ents in enumerate(entities):
        found_w = [e["word"] for e in ents]
        found_e = [e["entity"] for e in ents]

        gold_w = [e["word"] for e in tags[i]]
        gold_e = [e["entity"] for e in tags[i]]
        # gold_e = ["PER" if e == "PRS" else e for e in gold_e]

        per = evaluate_typewise(found_w, found_e, gold_w, gold_e, per, "PER")
        org = evaluate_typewise(found_w, found_e, gold_w, gold_e, org, "ORG")
        loc = evaluate_typewise(found_w, found_e, gold_w, gold_e, loc, "LOC")

    return per, org, loc


def calculate_metrics(res, s):
    precision = len(res["tp"]) / len(res["ap"])
    recall = len(res["tp"]) / len(res["rel"])
    f1 = 2 * precision * recall / (precision + recall)

    print(f"{s}: precision = {precision}, recall = {recall}, f1 = {f1}")

    return precision, recall, f1


corpus = load_corpus()
all_sentences = format_sentences(corpus[0])
all_tags = format_tags(corpus[1])

sentences, tags = prepare_for_evaluation(all_sentences, all_tags, False)
"""
At this point, NER would be performed by a model.
In this case however, the NER results are loaded from file.
"""

entities = get_results("data/output/bert_evaluation_v2.jsonl")
per, org, loc = evaluate(entities, tags)

per_metrics = calculate_metrics(per, "PER")
org_metrics = calculate_metrics(org, "ORG")
loc_metrics = calculate_metrics(loc, "LOC")

