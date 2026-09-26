"""
Basic exploratory data analysis over all four training files: shape,
columns, missing values, and a few sample rows for each.

Run from the repo root (amazon-challenge/):
    python code/business_entity_resolution/src/explore_data.py
"""

import pandas as pd

files = {
    "source1": "dataset/train/train_source1.tsv",
    "source2": "dataset/train/train_source2.tsv",
    "source3": "dataset/train/train_source3.tsv",
    "ground_truth": "dataset/train/train_ground_truth.tsv",
}

for name, path in files.items():
    df = pd.read_csv(path, sep="\t")

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))