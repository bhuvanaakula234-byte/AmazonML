"""
Local validator for matching_results.tsv and candidate_pairs.tsv.

IMPORTANT: this is NOT the organizers' official utils/validate_submission.py.
That script is provided in the challenge's dataset download package - use
this one only as a local sanity check while iterating, and always run the
real one before you submit, since it may check things this reconstruction
doesn't know about.

Checks implemented, per the problem statement's "Important" / "Constraints"
sections:
  - Both files are tab-separated with the exact expected header
  - Every Source-1 test entity appears exactly once in matching_results.tsv
    (and in candidate_pairs.tsv)
  - No duplicate source1_entity_id rows in either file
  - matched_entity_ids / candidate_entity_ids only reference S2-/S3- ids
    that actually exist in the test set (no self-matches to S1, no
    made-up ids)
  - No duplicate ids within a single row's id list
  - Every id in matching_results.tsv also appears in that row's
    candidate_pairs.tsv list (a matched id that was never a candidate
    signals a pipeline bug)

Usage (run from the repo root):
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import argparse
import csv
import os
import sys


def read_ids(path, expected_prefix):
    """Read the entity_id column of a source file into a set."""
    ids = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if not eid.startswith(expected_prefix):
                continue
            ids.add(eid)
    return ids


def read_results_file(path, id_col, expected_header):
    """Reads matching_results.tsv or candidate_pairs.tsv.
    Returns (rows, issues) where rows is a list of (source1_id, [ids...])
    in file order, and issues is a list of strings for structural problems
    caught while parsing (e.g. wrong header, wrong column count)."""
    issues = []
    rows = []
    if not os.path.exists(path):
        issues.append(f"File not found: {path}")
        return rows, issues

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != expected_header:
            issues.append(
                f"{path}: header is {header!r}, expected {expected_header!r}"
            )
        for i, row in enumerate(reader, start=2):
            if len(row) == 1 and row[0] == "":
                continue  # trailing blank line
            if len(row) != 2:
                issues.append(f"{path} line {i}: expected 2 columns, got {len(row)}: {row!r}")
                continue
            s1_id, id_list_str = row
            id_list = id_list_str.split(",") if id_list_str.strip() else []
            rows.append((s1_id, id_list))
    return rows, issues


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in (s1_path, s2_path, s3_path):
        if not os.path.exists(p):
            issues.append(f"Test file not found: {p}")
    if issues:
        return issues

    valid_s1_ids = read_ids(s1_path, "S1-")
    valid_s2_ids = read_ids(s2_path, "S2-")
    valid_s3_ids = read_ids(s3_path, "S3-")
    valid_candidate_ids = valid_s2_ids | valid_s3_ids

    matching_rows, matching_issues = read_results_file(
        matching_path, "source1_entity_id", ["source1_entity_id", "matched_entity_ids"]
    )
    candidate_rows, candidate_issues = read_results_file(
        candidate_path, "source1_entity_id", ["source1_entity_id", "candidate_entity_ids"]
    )
    issues.extend(matching_issues)
    issues.extend(candidate_issues)

    issues.extend(
        _validate_rows(matching_rows, "matching_results.tsv", valid_s1_ids, valid_candidate_ids)
    )
    issues.extend(
        _validate_rows(candidate_rows, "candidate_pairs.tsv", valid_s1_ids, valid_candidate_ids)
    )

    # matched ids must be a subset of that entity's candidate ids
    candidate_by_s1 = {s1_id: set(ids) for s1_id, ids in candidate_rows}
    for s1_id, matched_ids in matching_rows:
        cand_ids = candidate_by_s1.get(s1_id, set())
        leaked = set(matched_ids) - cand_ids
        if leaked:
            issues.append(
                f"matching_results.tsv: {s1_id} has matched id(s) {sorted(leaked)} "
                "that never appeared in candidate_pairs.tsv for this entity"
            )

    return issues


def _validate_rows(rows, filename, valid_s1_ids, valid_candidate_ids):
    issues = []
    seen_s1_ids = set()
    row_s1_ids = set()

    for s1_id, id_list in rows:
        row_s1_ids.add(s1_id)

        if s1_id not in valid_s1_ids:
            issues.append(f"{filename}: {s1_id} is not a valid Source-1 test id")

        if s1_id in seen_s1_ids:
            issues.append(f"{filename}: duplicate source1_entity_id row for {s1_id}")
        seen_s1_ids.add(s1_id)

        if len(id_list) != len(set(id_list)):
            dupes = sorted({i for i in id_list if id_list.count(i) > 1})
            issues.append(f"{filename}: {s1_id} has duplicate id(s) within its list: {dupes}")

        for cid in id_list:
            if cid.startswith("S1-"):
                issues.append(f"{filename}: {s1_id} lists a Source-1 id ({cid}) - self-matches are not allowed")
            elif cid not in valid_candidate_ids:
                issues.append(f"{filename}: {s1_id} lists {cid}, which is not in the test set")

    missing = valid_s1_ids - row_s1_ids
    if missing:
        sample = sorted(missing)[:10]
        issues.append(
            f"{filename}: missing {len(missing)} Source-1 test entities, e.g. {sample}"
        )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate submission TSV files")
    parser.add_argument("--matching", required=True, help="path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="path to dataset/test/")
    args = parser.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL - {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()