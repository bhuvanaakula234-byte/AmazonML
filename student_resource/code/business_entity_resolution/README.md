# Business Entity Resolution — Pipeline

Matches Source 1 business records against Source 2 / Source 3 records using:
blocking (candidate generation) → similarity feature engineering →
a gradient-boosted classifier (match / no-match) → formatted TSV output.

## 1. Expected folder layout

Place this `code/` folder inside your `student_resource/` directory,
alongside `dataset/`, `utils/`, and (once created) `output/`:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py        # provided by the organizers
├── output/                            # created automatically by the pipeline
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── code/
    └── business_entity_resolution/
        ├── README.md                  # this file
        ├── requirements.txt
        └── src/
            ├── config.py
            ├── normalize.py
            ├── blocking.py
            ├── features.py
            ├── generate_candidates.py
            ├── train_model.py
            ├── predict.py
            ├── evaluate.py
            └── run_pipeline.py
```

## 2. Setup

From `student_resource/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## 3. Train the model

Run this from `student_resource/` (the root, not from inside `src/`):

```bash
python code/business_entity_resolution/src/run_pipeline.py train
```

This will:
1. Load `dataset/train/*.tsv`
2. Run blocking to generate candidate pairs for every train Source-1 entity
3. Print the **blocking recall ceiling** (fraction of true matches that
   survived blocking) and the **average candidates per entity** — watch
   both; blocking size now counts toward the final ranking, so the goal is
   the smallest candidate set that doesn't sacrifice recall
4. Train the classifier on a held-out split of the training entities
5. Print your **validation F_0.5** score (the exact metric used to grade
   the challenge) plus precision/recall/singleton-accuracy diagnostics
6. Save the trained model to `output/match_model.joblib`

Re-run this after changing any knob in `config.py` (see section 5).

## 4. Generate test predictions

```bash
python code/business_entity_resolution/src/run_pipeline.py test
```

This loads `dataset/test/*.tsv`, re-runs blocking (this also correctly
handles France, since blocking never hardcodes which countries exist),
scores every candidate with the trained model, and writes:

- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

## 5. Validate before submitting

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Fix anything it flags before uploading `matching_results.tsv` to the
leaderboard.

## 6. Tuning knobs (all in `src/config.py`)

| Knob | Effect |
|---|---|
| `MAX_BUCKET_SIZE` | Lower = drops more "too generic" blocking keys → fewer, more precise candidates |
| `MAX_CANDIDATES_PER_SOURCE` | Hard cap on candidates kept per S1 entity, per source. Lower = smaller `candidate_pairs.tsv` (helps the size-based ranking criterion) but risks losing recall — watch the "blocking recall ceiling" printed by `train_model.py` |
| `MATCH_THRESHOLD` | Probability cutoff for calling a candidate a match. F_0.5 rewards precision 2x over recall, so this is usually tuned *above* 0.5, not below |
| `VALIDATION_FRACTION` | Size of the held-out training slice used to self-score F_0.5 |

Workflow for tuning: change a knob → re-run `train_model.py` (via
`run_pipeline.py train`) → check the printed validation F_0.5 and average
candidate count → adjust → repeat. Only run `predict.py` (the `test`
stage) once you're happy with validation numbers, since each submission is
rate-limited.

## 7. Design notes (also in the methodology doc)

- **Blocking**: inverted-index blocking keyed on (country, blocking-key),
  with keys = first token, sorted token set, and short prefixes of the
  first two tokens of the normalized business name. Country is read as an
  open string label, never hardcoded, so an unseen country (France) is
  handled identically to US/India.
- **Normalization**: lowercasing, accent stripping, punctuation removal,
  legal-suffix and stopword removal, and address-abbreviation expansion —
  all pure offline string processing, no external lookups or APIs.
- **Features**: RapidFuzz token-sort/token-set/partial-ratio and Jaccard
  overlap on both name and address, plus length-difference and
  missing-value indicators.
- **Model**: `sklearn.ensemble.HistGradientBoostingClassifier` — MIT
  licensed, a few hundred KB, well under the 8B-parameter limit.
- **No external data**: nothing in this pipeline calls the network, a
  geocoding service, or any business-registry lookup.