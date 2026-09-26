"""
Central configuration for the entity-resolution pipeline.

Everything that could change between your machine and a teammate's machine
(file locations, thresholds, sample sizes) lives here so the rest of the
code never hardcodes a path.

Override paths with environment variables if you don't want to edit this
file, e.g.:
    ER_DATA_DIR=/path/to/dataset ER_OUTPUT_DIR=/path/to/output python run_pipeline.py train
"""

import os

# ---------------------------------------------------------------------------
# Data locations - follows the dataset/train/... and dataset/test/... layout
# from the problem statement.
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get("ER_DATA_DIR", "dataset")

TRAIN_S1 = os.path.join(BASE_DIR, "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "train", "train_ground_truth.tsv")

TEST_S1 = os.path.join(BASE_DIR, "test", "test_source1.tsv")
TEST_S2 = os.path.join(BASE_DIR, "test", "test_source2.tsv")
TEST_S3 = os.path.join(BASE_DIR, "test", "test_source3.tsv")

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("ER_OUTPUT_DIR", "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "match_model.joblib")

TRAIN_CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "train_candidate_pairs.tsv")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ---------------------------------------------------------------------------
# Validation split (carved out of the TRAINING data, never touches test)
# ---------------------------------------------------------------------------
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Blocking / candidate-generation knobs
# ---------------------------------------------------------------------------
# IMPORTANT (per the challenge update): candidate_pairs.tsv size counts
# toward your FINAL ranking, separately from the leaderboard F_0.5 score.
# A smaller candidate set per S1 entity ranks higher, as long as recall
# doesn't suffer. Don't just maximize recall here - watch the reduction
# ratio (candidates generated / total possible pairs) and blocking recall
# that train_model.py prints, and push MAX_CANDIDATES_PER_SOURCE and
# MAX_BUCKET_SIZE down as far as possible before recall starts dropping.

# A blocking key shared by more than this many records is "too generic"
# (e.g. a common first name-token) and is skipped - it would blow up the
# candidate set without adding precision.
MAX_BUCKET_SIZE = 300

# After union-of-blocking-keys produces a raw candidate set for an S1
# entity, we keep only the top-N per source, ranked by quick fuzzy name
# similarity, before handing candidates to the ML model. Lower = smaller,
# leaner candidate_pairs.tsv (better for the ranking criterion above) but
# risks losing true matches if too aggressive - tune against the blocking
# recall number train_model.py prints.
MAX_CANDIDATES_PER_SOURCE = 12

# ---------------------------------------------------------------------------
# Matching-model decision threshold
# ---------------------------------------------------------------------------
# Start at 0.5, then RE-TUNE this against your own validation F_0.5 score
# (see evaluate.py) - F_0.5 rewards precision 2x over recall, so the best
# threshold is usually higher than 0.5, not lower.
MATCH_THRESHOLD = 0.5