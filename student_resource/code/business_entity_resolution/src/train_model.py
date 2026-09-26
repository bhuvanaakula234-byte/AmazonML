"""
Trains the match / no-match classifier.

Pipeline:
  1. Load train S1/S2/S3 + ground truth.
  2. Run the SAME blocking code used at inference time to get candidate
     pairs for every train S1 entity - this also tells you your blocking
     recall ceiling, the single most important diagnostic in the pipeline
     (the model can never recover a true match blocking never proposed).
  3. Label each candidate pair: 1 if it's a true match, 0 otherwise.
  4. Split by S1 ENTITY (not by pair) into train/validation, so no
     entity's pairs leak across the split.
  5. Train a small, license-clean classifier (sklearn's
     HistGradientBoostingClassifier - MIT-licensed, a few hundred KB,
     trivially satisfies the <=8B parameter / open-license constraint).
  6. Score the validation split with the challenge's own F_0.5 formula and
     report it, plus a precision/recall breakdown.
  7. Save the trained model.

Run from the src/ directory:
    python train_model.py
"""

import os
import random
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config
import evaluate
from features import pair_features, FEATURE_NAMES
from generate_candidates import (
    build_lookup,
    build_indices,
    generate_candidates,
    write_candidate_pairs,
)

READ_KWARGS = dict(sep="\t", dtype=str, keep_default_na=False)


def load_data():
    s1 = pd.read_csv(config.TRAIN_S1, **READ_KWARGS)
    s2 = pd.read_csv(config.TRAIN_S2, **READ_KWARGS)
    s3 = pd.read_csv(config.TRAIN_S3, **READ_KWARGS)
    gt = evaluate.load_ground_truth(config.TRAIN_GT)
    return s1, s2, s3, gt


def build_labeled_pairs(candidates, ground_truth):
    """Returns list of (s1_id, cand_id, source, label)."""
    rows = []
    for s1_id, pairs in candidates.items():
        actual = ground_truth.get(s1_id, set())
        for cand_id, source in pairs:
            rows.append((s1_id, cand_id, source, 1 if cand_id in actual else 0))
    return rows


def blocking_recall(candidates, ground_truth):
    """Fraction of TRUE matches that survived blocking - the pipeline's
    recall ceiling."""
    total_true = recovered = 0
    for s1_id, actual in ground_truth.items():
        if not actual:
            continue
        cand_ids = {cid for cid, _source in candidates.get(s1_id, [])}
        total_true += len(actual)
        recovered += len(actual & cand_ids)
    return recovered / total_true if total_true else 1.0


def avg_candidates_per_entity(candidates):
    """Mean candidate-set size per S1 entity. The challenge update says a
    SMALLER candidate_pairs.tsv (at equal recall) ranks higher beyond the
    leaderboard score - track this number every run and try to shrink it."""
    if not candidates:
        return 0.0
    return sum(len(v) for v in candidates.values()) / len(candidates)


def featurize(rows, s1_lookup, s2_lookup, s3_lookup):
    X, y, s1_ids = [], [], []
    for s1_id, cand_id, source, label in rows:
        name1, addr1, _c1 = s1_lookup[s1_id]
        cand_lookup = s2_lookup if source == "S2" else s3_lookup
        name2, addr2, _c2 = cand_lookup.get(cand_id, ("", "", ""))
        X.append(pair_features(name1, addr1, name2, addr2))
        y.append(label)
        s1_ids.append(s1_id)
    return X, y, s1_ids


def predict_matches_for_entities(model, entity_ids, candidates, s1_lookup, s2_lookup, s3_lookup, threshold):
    """Scores every candidate pair for the given S1 entities and returns
    dict: s1_id -> set(predicted matched ids), using the same feature
    pipeline as training."""
    predictions = {s1_id: set() for s1_id in entity_ids}
    rows = []
    for s1_id in entity_ids:
        for cand_id, source in candidates.get(s1_id, []):
            rows.append((s1_id, cand_id, source, 0))  # label unused here
    if not rows:
        return predictions

    X, _y, _ = featurize(rows, s1_lookup, s2_lookup, s3_lookup)
    probs = model.predict_proba(X)[:, 1]
    for (s1_id, cand_id, _source, _label), prob in zip(rows, probs):
        if prob >= threshold:
            predictions[s1_id].add(cand_id)
    return predictions


def main():
    t0 = time.time()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading training data...")
    s1, s2, s3, gt = load_data()
    print(f"  S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  ground_truth={len(gt)}  ({time.time()-t0:.1f}s)")

    print("Building lookups + blocking indices...")
    s1_lookup = build_lookup(s1)
    s2_lookup = build_lookup(s2)
    s3_lookup = build_lookup(s3)
    s2_index, s3_index, s2_dropped, s3_dropped = build_indices(s2_lookup, s3_lookup)
    print(f"  dropped {s2_dropped} overly-generic S2 buckets, {s3_dropped} S3 buckets ({time.time()-t0:.1f}s)")

    print("Generating candidates (blocking stage) for all train S1 entities...")
    candidates = generate_candidates(s1, s2_lookup, s3_lookup, s2_index, s3_index)
    print(f"  done ({time.time()-t0:.1f}s)")

    recall_ceiling = blocking_recall(candidates, gt)
    avg_cands = avg_candidates_per_entity(candidates)
    print(f"Blocking recall ceiling: {recall_ceiling:.4f} "
          "(fraction of true matches that survived blocking - raise MAX_CANDIDATES_PER_SOURCE "
          "or add blocking keys in blocking.py if this is too low)")
    print(f"Average candidates per S1 entity: {avg_cands:.2f} "
          "(candidate_pairs.tsv size counts toward final ranking - try to shrink this "
          "without dropping the recall ceiling above)")

    write_candidate_pairs(candidates, config.TRAIN_CANDIDATE_PAIRS_PATH)

    print("Splitting S1 entities into train/validation...")
    s1_ids = list(candidates.keys())
    random.Random(config.RANDOM_SEED).shuffle(s1_ids)
    n_val = int(len(s1_ids) * config.VALIDATION_FRACTION)
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])
    print(f"  train entities={len(train_ids)}  val entities={len(val_ids)}")

    labeled = build_labeled_pairs(candidates, gt)
    train_rows = [r for r in labeled if r[0] in train_ids]
    print(f"  train pairs={len(train_rows)} (positives={sum(r[3] for r in train_rows)})")

    print("Computing features for training pairs...")
    X_train, y_train, _ = featurize(train_rows, s1_lookup, s2_lookup, s3_lookup)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("Training classifier...")
    model = HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("Scoring validation split with the challenge's F_0.5 formula...")
    val_predictions = predict_matches_for_entities(
        model, val_ids, candidates, s1_lookup, s2_lookup, s3_lookup, config.MATCH_THRESHOLD
    )
    val_ground_truth = {s1_id: gt.get(s1_id, set()) for s1_id in val_ids}
    macro_f05 = evaluate.macro_f_beta(val_predictions, val_ground_truth, beta=0.5)
    diagnostics = evaluate.precision_recall_summary(val_predictions, val_ground_truth)
    print(f"  Validation macro F_0.5: {macro_f05:.4f}")
    print(f"  Validation diagnostics: {diagnostics}")

    print(f"Saving model to {config.MODEL_PATH}")
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, config.MODEL_PATH)
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()