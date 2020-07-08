from string import punctuation
import regex
import random
import jsonlines


def load_corpus():
    with open("data/input/suc_3.0_iob.txt") as f:
        lines = f.readlines()
        f.seek(0)
        corpus = f.read()

    entities = [line for line in lines if not line.endswith("\tO\n")]

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
    desired = ["B-PRS", "I-PRS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    filtered = []

    for ts in tags:
        filt = [t for t in ts if t["entity"] in desired]
        filtered += [filt] if filt else [[]]

    return filtered


def prepare_for_evaluation(sentences, tags):
    no_sentences = len(all_sentences)
    eval_size = int(no_sentences * 0.1)
    random.seed(1234567890)
    eval_inds = [random.randint(0, no_sentences - 1) for i in range(0, eval_size)]

    sentences = [all_sentences[i] for i in eval_inds]
    tags = [all_tags[i] for i in eval_inds]
    filtered = filter_tags(tags)

    return sentences, filtered


def get_nerd_results():
    with jsonlines.open("data/output/nerd_evaluation_v2.jsonl", "r") as reader:
        results = [obj for obj in reader]

    return results


def evaluate_typewise(n_w, n_e, g_w, g_e, d, t):
    n_i = [i for i, x in enumerate(n_e) if x == t]
    g_i = [i for i, x in enumerate(g_e) if x == t]

    n_w = [n_w[i].split() for i in n_i]
    n_w = [w for ws in n_w for w in ws if w not in set(punctuation)]
    g_w = [g_w[i] for i in g_i]

    true_positives = []
    all_positives = n_w
    relevant = g_w.copy()

    for w in n_w:
        if w in g_w:
            true_positives += [w]
            del g_w[g_w.index(w)]

    d["tp"] += true_positives
    d["ap"] += all_positives
    d["rel"] += relevant

    return d


def calculate_metrics(res, s):
    precision = len(res["tp"]) / len(res["ap"])
    recall = len(res["tp"]) / len(res["rel"])
    f1 = 2 * precision * recall / (precision + recall)

    print(f"{s}: precision = {precision}, recall = {recall}, f1 = {f1}")

    return precision, recall, f1


corpus = load_corpus()
all_sentences = format_sentences(corpus[0])
all_tags = format_tags(corpus[1])

sentences, tags = prepare_for_evaluation(all_sentences, all_tags)
nerd_ents = get_nerd_results()

# True positives, all positives, relevant
per = {"tp": [], "ap": [], "rel": []}
org = {"tp": [], "ap": [], "rel": []}
loc = {"tp": [], "ap": [], "rel": []}

for i, entities in enumerate(nerd_ents):
    nerd_w = [e["word"] for e in entities]
    nerd_e = [e["entity"] for e in entities]

    gold_w = [e["word"] for e in tags[i]]
    gold_e = [e["entity"][2:] for e in tags[i]]
    gold_e = ["PER" if e == "PRS" else e for e in gold_e]

    per = evaluate_typewise(nerd_w, nerd_e, gold_w, gold_e, per, "PER")
    org = evaluate_typewise(nerd_w, nerd_e, gold_w, gold_e, org, "ORG")
    loc = evaluate_typewise(nerd_w, nerd_e, gold_w, gold_e, loc, "LOC")

per_metrics = calculate_metrics(per, "PER")
org_metrics = calculate_metrics(org, "ORG")
loc_metrics = calculate_metrics(loc, "LOC")

