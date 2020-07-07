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
        sentence = regex.sub("\t.*\n(?=\p{P})", "", sentence)
        sentence = regex.sub("\t.*?\n", " ", sentence)
        sentence = regex.split("\t.*", sentence)[0]
        sentence = regex.sub('" (?=.*")', ' "', sentence)
        sentences += [sentence]

    return sentences[:-1]


def format_labels(entities):
    new_l = [-1]
    new_l += [i for i, val in enumerate(entities) if val == "\n"]
    grouped = []

    for i in range(1, len(new_l)):
        sen_ent = [entities[j].split("\t") for j in range(new_l[i - 1] + 1, new_l[i])]
        ent_dict = [{"word": e[0], "entity": e[1].strip()} for e in sen_ent]
        grouped += [ent_dict]

    return grouped


def filter_labels(labels):
    desired = ["B-PRS", "I-PRS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    filtered = []

    for ls in labels:
        filt = [l for l in ls if l["entity"] in desired]
        filtered += [filt] if filt else [[]]

    return filtered


def evaluate_type_wise(n_w, n_l, g_w, g_l, t):
    n_i = [i for i, x in enumerate(n_l) if x == t]
    g_i = [i for i, x in enumerate(g_l) if x == t]

    n_w = [n_w[i].split() for i in n_i]
    n_w = [w for ws in n_w for w in ws if w not in punct]
    g_w = [g_w[i] for i in g_i]

    true_positives = []
    all_positives = n_w
    relevant = g_w.copy()

    for w in n_w:
        if w in g_w:
            true_positives += [w]
            del g_w[g_w.index(w)]

    if len(relevant) < len(true_positives):
        print("det är JIDDER")
        print(relevant)
        print(true_positives)
    return true_positives, all_positives, relevant


def print_metrics(tp, ap, rel, s):
    precision = len(tp) / len(ap)
    recall = len(tp) / len(rel)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"{s}: precision = {precision}, recall = {recall}, f1 = {f1}")

    return precision, recall, f1


corpus = load_corpus()
all_sentences = format_sentences(corpus[0])
all_labels = format_labels(corpus[1])

no_sentences = len(all_sentences)
eval_size = int(no_sentences * 0.1)
random.seed(1234567890)
eval_inds = [random.randint(0, no_sentences - 1) for i in range(0, eval_size)]

sentences = [all_sentences[i] for i in eval_inds]
[print(s) for s in sentences[187:190]]
labels = [all_labels[i] for i in eval_inds]
filtered = filter_labels(labels)

with jsonlines.open("data/output/nerd_evaluation_v2.jsonl", "r") as reader:
    nerd_ents = [obj for obj in reader]


punct = set(punctuation)

per_tp = []
per_ap = []
per_rel = []

org_tp = []
org_ap = []
org_rel = []

loc_tp = []
loc_ap = []
loc_rel = []

tot = 0
for i, entities in enumerate(nerd_ents):
    # pre_w = [e["word"].split() for e in entities]
    # nerd_w = [e.strip() for es in pre_w for e in es]
    nerd_w = [e["word"] for e in entities]
    nerd_e = [e["entity"] for e in entities]

    gold_w = [e["word"] for e in filtered[i]]
    gold_e = [e["entity"][2:] for e in filtered[i]]
    gold_e = ["PER" if e == "PRS" else e for e in gold_e]

    per_res = evaluate_type_wise(nerd_w, nerd_e, gold_w, gold_e, "PER")
    per_tp += per_res[0]
    per_ap += per_res[1]
    per_rel += per_res[2]

    org_res = evaluate_type_wise(nerd_w, nerd_e, gold_w, gold_e, "ORG")
    org_tp += org_res[0]
    org_ap += org_res[1]
    org_rel += org_res[2]

    loc_res = evaluate_type_wise(nerd_w, nerd_e, gold_w, gold_e, "LOC")
    loc_tp += loc_res[0]
    loc_ap += loc_res[1]
    loc_rel += loc_res[2]

# print(org_ap)
per_metrics = print_metrics(per_tp, per_ap, per_rel, "PER")
org_metrics = print_metrics(org_tp, org_ap, org_rel, "ORG")
loc_metrics = print_metrics(loc_tp, loc_ap, loc_rel, "LOC")

