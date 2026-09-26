"""
Your own local F_0.5 scorer, matching the challenge's evaluation exactly:
computed PER Source-1 entity, then macro-averaged across entities.
Singletons (no true matches) score 1.0 if you correctly predict an empty
list, 0.0 if you predict anything for them.

Use this against a held-out validation split of the TRAINING data (never
the test data, which has no ground truth) to tune your threshold before
spending a leaderboard submission.
"""

import csv
from collections import defaultdict


def load_ground_truth(path):
    """Returns dict: source1_entity_id -> set(matched_entity_ids)."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            matched = row[1] if len(row) > 1 and row[1].strip() else ""
            gt[s1_id] = set(matched.split(",")) if matched else set()
    return gt


def f_beta_per_entity(predicted: set, actual: set, beta: float = 0.5) -> float:
    """F_beta for one Source-1 entity's predicted vs actual match sets."""
    if not actual and not predicted:
        return 1.0  # correctly predicted a singleton
    if not predicted:
        return 0.0  # missed everything (or falsely predicted empty)
    tp = len(predicted & actual)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(actual)
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(predictions: dict, ground_truth: dict, beta: float = 0.5) -> float:
    """predictions, ground_truth: dict s1_entity_id -> set(matched_ids).
    Every id in ground_truth must have an entry in predictions (missing
    entities are treated as an empty prediction, same as the real scorer
    would likely do with a malformed submission - but validate_submission.py
    should catch that before you ever get here).
    """
    scores = []
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        scores.append(f_beta_per_entity(predicted, actual, beta))
    return sum(scores) / len(scores) if scores else 0.0


def precision_recall_summary(predictions: dict, ground_truth: dict) -> dict:
    """Micro precision/recall across all pairs, plus singleton accuracy -
    useful diagnostics alongside the macro F_0.5 score."""
    tp = fp = fn = 0
    singleton_correct = 0
    singleton_total = 0
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        if not actual:
            singleton_total += 1
            if not predicted:
                singleton_correct += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total else 0.0
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "singleton_accuracy": singleton_acc,
        "singleton_total": singleton_total,
    }