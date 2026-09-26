"""
Quick sanity-check loader for the four training files.

Run from the repo root (amazon-challenge/), so the relative paths below
resolve correctly:
    python code/business_entity_resolution/src/data_loader.py
"""

import pandas as pd

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print("Source 1 shape:", s1.shape)
print("Source 2 shape:", s2.shape)
print("Source 3 shape:", s3.shape)
print("Ground truth shape:", ground_truth.shape)

print("\nSource 1 columns:")
print(s1.columns.tolist())

print("\nSource 1 sample:")
print(s1.head())

print("\nGround truth sample:")
print(ground_truth.head())