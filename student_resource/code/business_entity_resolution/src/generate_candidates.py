"""
Runs the full blocking stage over a set of (S1, S2, S3) dataframes and
produces the candidate set: for every S1 entity, which S2/S3 records are
worth scoring with the ML model.

This is the code that produces candidate_pairs.tsv - the exact set fed to
the matching model at inference time, per the problem statement's
requirement.
"""

import csv

import config
from blocking import build_index, raw_candidates, rank_and_cap
from normalize import normalize_name, normalize_address


def build_lookup(df):
    """id -> (name_norm, addr_norm, country) for every record in df.
    Kept as a plain dict (not the dataframe) because downstream code does
    millions of point lookups by id - a dict is much faster for that than
    repeated dataframe indexing."""
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = (
            normalize_name(row.business_name),
            normalize_address(row.business_address),
            row.country,
        )
    return lookup


def generate_candidates(s1_df, s2_lookup, s3_lookup, s2_index, s3_index):
    """Returns dict: s1_entity_id -> list of (candidate_id, source_label)
    source_label is 'S2' or 'S3'. Every S1 entity gets an entry, possibly
    an empty list.
    """
    candidates = {}
    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        name_norm = normalize_name(row.business_name)
        addr_norm = normalize_address(row.business_address)
        country = row.country

        raw2 = raw_candidates(name_norm, country, s2_index)
        raw3 = raw_candidates(name_norm, country, s3_index)

        rec2 = {cid: s2_lookup[cid][:2] for cid in raw2 if cid in s2_lookup}
        rec3 = {cid: s3_lookup[cid][:2] for cid in raw3 if cid in s3_lookup}

        top2 = rank_and_cap(name_norm, addr_norm, raw2, rec2, config.MAX_CANDIDATES_PER_SOURCE)
        top3 = rank_and_cap(name_norm, addr_norm, raw3, rec3, config.MAX_CANDIDATES_PER_SOURCE)

        candidates[s1_id] = [(cid, "S2") for cid in top2] + [(cid, "S3") for cid in top3]

    return candidates


def write_candidate_pairs(candidates: dict, path: str):
    """Writes candidate_pairs.tsv in the exact required format: one row per
    S1 entity, comma-joined candidate ids, empty string (not 'nan') when
    there are no candidates."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, pairs in candidates.items():
            ids = [cid for cid, _source in pairs]
            writer.writerow([s1_id, ",".join(ids)])


def build_indices(s2_lookup, s3_lookup):
    s2_records = [(cid, name, country) for cid, (name, _addr, country) in s2_lookup.items()]
    s3_records = [(cid, name, country) for cid, (name, _addr, country) in s3_lookup.items()]
    s2_index, s2_dropped = build_index(s2_records)
    s3_index, s3_dropped = build_index(s3_records)
    return s2_index, s3_index, s2_dropped, s3_dropped