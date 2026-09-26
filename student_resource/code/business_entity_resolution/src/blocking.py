"""
Blocking stage: cuts the S1 x S2 x S3 cross product (which is far too big
to brute-force - millions of records per source) down to a small,
high-recall candidate set per Source-1 entity.

Strategy: build several cheap "blocking keys" per record (first name token,
sorted token set, short prefixes) and index records by (country, key).
For a given S1 entity, its candidates are the union of everything sharing
any of its keys, within the same country. This is the standard
inverted-index blocking approach for entity resolution at scale.

We always block on country first - never compare across countries - and
never hardcode which countries exist, so this works unchanged on France
in the test set even though it never appears in training.
"""

from collections import defaultdict

from rapidfuzz import fuzz

import config


def build_blocking_keys(name_norm: str) -> set:
    """A handful of cheap, complementary keys for one normalized name.
    Union of several weak keys gets much better recall than any single key.
    """
    toks = name_norm.split()
    keys = set()
    if not toks:
        return keys

    keys.add(("tok0", toks[0]))                      # first token
    keys.add(("sorted", " ".join(sorted(toks))))      # word-order-proof
    keys.add(("prefix0", toks[0][:4]))                # typo-tolerant prefix
    if len(toks) >= 2:
        keys.add(("tok1", toks[1]))
        keys.add(("prefix1", toks[1][:4]))
    return keys


def build_index(records) -> dict:
    """records: iterable of (entity_id, name_norm, country).
    Returns: dict[country][key] -> list[entity_id], with overly generic
    buckets (shared by more than MAX_BUCKET_SIZE records) dropped - a bucket
    that big is too common a token to discriminate between businesses and
    just slows everything down without adding recall.
    """
    raw = defaultdict(lambda: defaultdict(list))
    for entity_id, name_norm, country in records:
        for key in build_blocking_keys(name_norm):
            raw[country][key].append(entity_id)

    index = defaultdict(dict)
    dropped_buckets = 0
    for country, buckets in raw.items():
        for key, ids in buckets.items():
            if len(ids) <= config.MAX_BUCKET_SIZE:
                index[country][key] = ids
            else:
                dropped_buckets += 1
    return index, dropped_buckets


def raw_candidates(name_norm: str, country: str, index: dict) -> set:
    """Union of every id sharing any blocking key with this record, in the
    same country. This is the pre-ranking, pre-cap candidate set."""
    cand = set()
    bucket = index.get(country, {})
    for key in build_blocking_keys(name_norm):
        cand.update(bucket.get(key, []))
    return cand


def rank_and_cap(name_norm: str, addr_norm: str, candidate_ids, id_to_record: dict, max_candidates: int):
    """Cheap re-ranking pass: score each raw candidate by a blend of fast
    fuzzy name AND address similarity, then keep only the top
    max_candidates. Using address too (not just name) matters a lot here -
    a true match with a slightly weaker name match but a strong address
    match would otherwise get pushed out of the top-N by name-similar
    look-alikes. id_to_record: cand_id -> (name_norm, addr_norm)."""
    if not candidate_ids:
        return []
    scored = []
    for cid in candidate_ids:
        cand_name, cand_addr = id_to_record.get(cid, ("", ""))
        name_score = fuzz.token_sort_ratio(name_norm, cand_name)
        addr_score = fuzz.token_sort_ratio(addr_norm, cand_addr) if addr_norm and cand_addr else 0.0
        combined = 0.65 * name_score + 0.35 * addr_score
        scored.append((cid, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:max_candidates]]