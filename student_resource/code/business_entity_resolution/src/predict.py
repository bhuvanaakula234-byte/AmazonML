"""
Runs blocking + the trained model over the TEST data and writes both
required output files:
  - output/candidate_pairs.tsv   (the blocking stage's candidate set)
  - output/matching_results.tsv  (final matches, scored on the leaderboard)

Every Source-1 test entity gets exactly one row in matching_results.tsv,
empty string for singletons, and every matched id is guaranteed to also
appear in candidate_pairs.tsv (predictions are only ever chosen from the
candidate set, never invented separately).

Run from the src/ directory, after train_model.py has produced a model:
    python predict.py
"""

import csv
import os
import time

import joblib
import pandas as pd

import config
from generate_candidates import build_lookup, build_indices, generate_candidates, write_candidate_pairs
from train_model import predict_matches_for_entities

READ_KWARGS = dict(sep="\t", dtype=str, keep_default_na=False)


def load_test_data():
    s1 = pd.read_csv(config.TEST_S1, **READ_KWARGS)
    s2 = pd.read_csv(config.TEST_S2, **READ_KWARGS)
    s3 = pd.read_csv(config.TEST_S3, **READ_KWARGS)
    return s1, s2, s3


def write_matching_results(predictions: dict, s1_ids_in_order, path: str):
    """One row per S1 test entity, in the same order the entity appeared in
    test_source1.tsv, satisfying 'every Source 1 entity must appear
    exactly once'."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids_in_order:
            matched = sorted(predictions.get(s1_id, set()))
            writer.writerow([s1_id, ",".join(matched)])


def main():
    t0 = time.time()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from {config.MODEL_PATH}")
    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]

    print("Loading test data...")
    s1, s2, s3 = load_test_data()
    print(f"  S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  ({time.time()-t0:.1f}s)")

    print("Building lookups + blocking indices (test data, including any new countries)...")
    s1_lookup = build_lookup(s1)
    s2_lookup = build_lookup(s2)
    s3_lookup = build_lookup(s3)
    s2_index, s3_index, s2_dropped, s3_dropped = build_indices(s2_lookup, s3_lookup)
    print(f"  dropped {s2_dropped} S2 buckets, {s3_dropped} S3 buckets ({time.time()-t0:.1f}s)")

    print("Generating candidates for all test S1 entities...")
    candidates = generate_candidates(s1, s2_lookup, s3_lookup, s2_index, s3_index)
    print(f"  done ({time.time()-t0:.1f}s)")
    write_candidate_pairs(candidates, config.CANDIDATE_PAIRS_PATH)
    print(f"  wrote {config.CANDIDATE_PAIRS_PATH}")

    print("Scoring candidates with the trained model...")
    all_s1_ids = list(s1["entity_id"])
    predictions = predict_matches_for_entities(
        model, all_s1_ids, candidates, s1_lookup, s2_lookup, s3_lookup, config.MATCH_THRESHOLD
    )
    print(f"  done ({time.time()-t0:.1f}s)")

    write_matching_results(predictions, all_s1_ids, config.MATCHING_RESULTS_PATH)
    print(f"  wrote {config.MATCHING_RESULTS_PATH}")

    n_matched = sum(1 for v in predictions.values() if v)
    print(f"Predicted matches for {n_matched}/{len(all_s1_ids)} S1 entities "
          f"({len(all_s1_ids) - n_matched} singletons predicted)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()