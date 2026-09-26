"""
Feature engineering for a single (Source-1 record, candidate record) pair.

These features are computed only for records that already survived
blocking - never for the full cross product - which is what keeps this
stage fast even though the raw datasets have millions of rows.
"""

from rapidfuzz import fuzz

from normalize import token_set

FEATURE_NAMES = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "name_len_diff",
    "addr_len_diff",
    "name_empty",
    "addr_empty",
]


def pair_features(name1: str, addr1: str, name2: str, addr2: str) -> list:
    """Returns a fixed-order list of floats - order matches FEATURE_NAMES,
    which train_model.py and predict.py both import so the model always
    sees features in the same order it was trained on."""
    t1, t2 = token_set(name1), token_set(name2)
    a1, a2 = token_set(addr1), token_set(addr2)

    name_jaccard = _jaccard(t1, t2)
    addr_jaccard = _jaccard(a1, a2)

    return [
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.token_set_ratio(name1, name2) / 100.0,
        fuzz.partial_ratio(name1, name2) / 100.0,
        name_jaccard,
        fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        fuzz.token_set_ratio(addr1, addr2) / 100.0,
        addr_jaccard,
        abs(len(name1) - len(name2)),
        abs(len(addr1) - len(addr2)),
        1.0 if not name1 or not name2 else 0.0,
        1.0 if not addr1 or not addr2 else 0.0,
    ]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)