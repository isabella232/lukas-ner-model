import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from .evaluator import Evaluator


def calculate_scores(entities):
    precisions = []
    recalls = []
    f1s = []
    no_sels = []

    for min_thresh in range(0, 1000, 1):

        thresh = min_thresh / 1000
        print(thresh)
        per, org, loc, no_sel = evaluator.evaluate(entities, tags, thresh)
        per_metrics = evaluator.calculate_metrics(per, "PER")
        org_metrics = evaluator.calculate_metrics(org, "ORG")
        loc_metrics = evaluator.calculate_metrics(loc, "LOC")

        combined_precisions = (per_metrics[0] + org_metrics[0] + loc_metrics[0]) / 3
        combined_recalls = (per_metrics[1] + org_metrics[1] + loc_metrics[1]) / 3
        combined_f1s = (per_metrics[2] + org_metrics[2] + loc_metrics[2]) / 3

        precisions += [(thresh, combined_precisions)]
        recalls += [(thresh, combined_recalls)]
        f1s += [(thresh, combined_f1s)]
        no_sels += [(thresh, no_sel)]

    return precisions, recalls, f1s, no_sels


def threshold_evaluation(scores):
    precision = scores[0]
    recall = scores[1]
    f1 = scores[2]

    max_f1 = max(f1, key=lambda x: x[1])[0]

    x, y = zip(*f1)
    y_spl = UnivariateSpline(x, y, s=0, k=4)
    x_range = np.linspace(x[0], x[-1], 1000)
    y_spl_2d = y_spl.derivative(n=2)
    # plt.plot(x_range, y_spl_2d(x_range))
    max_f1_2d = [x for x in x_range if y_spl_2d(x) == max(y_spl_2d(x_range))]

    p_x, p_y = zip(*precision)
    r_x, r_y = zip(*recall)

    p_x = np.array(p_x)
    p_y = np.array(p_y)
    r_y = np.array(r_y)

    idx = np.argwhere(np.diff(np.sign(p_y - r_y))).flatten()
    inter = p_x[idx]  # A bit off, interpolation should make it better

    return max_f1, max_f1_2d[0], inter


def plot_thresholds(scores):
    max_f1, max_f1_2d, inter = threshold_evaluation(scores)

    f1 = plt.figure(1)

    plt.plot(*zip(*scores[3]), label="Number of Entities")
    plt.title(
        "Effect of Threshold for Minimum Entity\nConfidence on the Number of Entities"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Number of Entities")

    f1.show()

    f2 = plt.figure(2)

    plt.plot(*zip(*scores[0]), label="Precision")
    plt.plot(*zip(*scores[1]), label="Recall")
    plt.plot(*zip(*scores[2]), label="F1")

    plt.axvline(max_f1, c="grey", ls="--")
    plt.text(max_f1 - 0.03, 0.5, f"thresh = {'{0:.4g}'.format(max_f1)}", rotation=90)
    plt.axvline(max_f1_2d, c="grey", ls="--")
    plt.text(
        max_f1_2d - 0.03, 0.5, f"thresh = {'{0:.4g}'.format(max_f1_2d)}", rotation=90
    )

    if inter:
        plt.axvline(inter[0], c="grey", ls="--")
        plt.text(
            inter[0] - 0.03, 0.5, f"thresh = {'{0:.4g}'.format(inter[0])}", rotation=90
        )

    plt.title(
        "Effect of Threshold for Minimum Entity\nConfidence on Precision, Recall & F1 Score"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(0.1, 0.3), loc="upper left")

    f2.show()

    plt.show()


evaluator = Evaluator(False)

all_sentences, all_tags = evaluator.load_corpus()
sentences, tags = evaluator.prepare_for_evaluation(all_sentences, all_tags, 1.0)

entities = evaluator.get_results("data/output/bert_evaluation_v2.jsonl")
scores = calculate_scores(entities)
plot_thresholds(scores)

