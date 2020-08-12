from string import punctuation
import random

import jsonlines
import regex


class Evaluator:
    """Class for evaluating NER models on two IOB tagged datasets: SUC 3.0 and Web News 2012, both from Språkbanken."""

    def __init__(self, suc):
        self.suc = suc

        if suc:
            self.path = "data/input/suc_3.0_iob.txt"
            self.no_chunk = "O"
            self.desired = ["B-PRS", "I-PRS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        else:
            self.path = "data/input/ner_corpus.txt"
            self.no_chunk = "0"
            self.desired = ["PER", "ORG", "LOC"]

    def _format_sentences(self, corpus):
        """Returns a list of only sentences."""
        corpus = corpus.split("\n\n")
        sentences = []

        for sentence in corpus:
            # sentence = regex.sub("\t.*\n(?=\p{P})", "", sentence)
            sentence = regex.sub("\t.*?\n", " ", sentence)
            sentence = regex.split("\t.*", sentence)[0]
            # sentence = regex.sub('" (?=.*")', ' "', sentence)
            sentences += [sentence]

        return sentences[:-1]

    def _format_tags(self, entities):
        """Returns a list of only tags"""
        new_l = [-1]
        new_l += [i for i, val in enumerate(entities) if val == "\n"]
        grouped = []

        for i in range(1, len(new_l)):
            sen_ent = [
                entities[j].split("\t") for j in range(new_l[i - 1] + 1, new_l[i])
            ]
            ent_dict = [{"word": e[0], "entity": e[1].strip()} for e in sen_ent]
            grouped += [ent_dict]

        return grouped

    def load_corpus(self):
        """Loads the NER-tagged corpus and returns the sentences and tags."""
        with open(self.path) as f:
            lines = f.readlines()
            f.seek(0)
            corpus = f.read()

        entities = [line for line in lines if not line.endswith(f"\t{self.no_chunk}\n")]

        sentences = self._format_sentences(corpus)
        tags = self._format_tags(entities)

        return sentences, tags

    def _filter_tags(self, tags):
        """Filters out unwanted IOB tags."""
        filtered = []

        for ts in tags:
            filt = [t for t in ts if t["entity"] in self.desired]
            filtered += [filt] if filt else [[]]

        return filtered

    def prepare_for_evaluation(self, sentences, tags, sample_size):
        """Prepare dataset for evaluation."""
        if sample_size < 1.0:
            no_sentences = len(sentences)
            eval_size = int(no_sentences * sample_size)
            random.seed(1234567890)
            eval_inds = random.sample(range(0, no_sentences - 1), eval_size)

            sentences = [sentences[i] for i in eval_inds]
            tags = [tags[i] for i in eval_inds]

        filtered = self._filter_tags(tags)

        return sentences, filtered

    @staticmethod
    def get_results(path):
        """Returns previouslt saved evaluation results."""
        with jsonlines.open(path, "r") as reader:
            results = [obj for obj in reader]

        return results

    # f = found, g = golden standard, w = words, e = entities, i = index, d = dictionary, t = tag
    def _evaluate_typewise(self, f_w, f_e, g_w, g_e, d, t):
        """Evaluates model performance for one entity type."""
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

    def evaluate(self, entities, tags, min_thresh):
        """Performs evaluation."""
        # True positives, all positives, relevant
        per = {"tp": [], "ap": [], "rel": []}
        org = {"tp": [], "ap": [], "rel": []}
        loc = {"tp": [], "ap": [], "rel": []}

        no_selected = 0
        for i, ents in enumerate(entities):
            found_w = [e["word"] for e in ents if e["score"] >= min_thresh]
            found_e = [e["entity"] for e in ents if e["score"] >= min_thresh]
            gold_w = [e["word"] for e in tags[i]]

            no_selected += len(found_w)

            if self.suc:
                gold_e = [e["entity"][2:] for e in tags[i]]
                gold_e = ["PER" if e == "PRS" else e for e in gold_e]
            else:
                gold_e = [e["entity"] for e in tags[i]]

            per = self._evaluate_typewise(found_w, found_e, gold_w, gold_e, per, "PER")
            org = self._evaluate_typewise(found_w, found_e, gold_w, gold_e, org, "ORG")
            loc = self._evaluate_typewise(found_w, found_e, gold_w, gold_e, loc, "LOC")

        return per, org, loc, no_selected

    @staticmethod
    def calculate_metrics(res, tag):
        """Calculates and returns precision, recall and F1 score."""
        precision = len(res["tp"]) / len(res["ap"])
        recall = len(res["tp"]) / len(res["rel"])
        f1 = 2 * precision * recall / (precision + recall)

        print(f"{tag}: precision = {precision}, recall = {recall}, f1 = {f1}")

        return precision, recall, f1
