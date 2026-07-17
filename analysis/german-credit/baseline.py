"""
Train/test split, train-only scaling, the cost metric, and the two trivial
baselines from german-credit-notes.md. No classifier is trained here.

Positive class: bad credit (class == 2).
False negative: predicted good/approve, actually bad -> lender's costly error, weight 5.
False positive: predicted bad/deny, actually good -> applicant's harm, weight 1.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "german_credit_raw.csv"
TARGET_COL = "class"
BAD = 2  # positive class
GOOD = 1

# Integer-typed columns per UCI metadata; everything else (Categorical and
# Binary-typed, e.g. Attribute19/20) is still string-coded (A191, A192, ...)
# in the raw CSV, so it is not numeric and is not scaled here.
NUMERIC_COLS = [
    "Attribute2",   # Duration
    "Attribute5",   # Credit amount
    "Attribute8",   # Installment rate (%)
    "Attribute11",  # Present residence since
    "Attribute13",  # Age
    "Attribute16",  # Number of existing credits
    "Attribute18",  # Number of dependents
]

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c != TARGET_COL]
categorical_cols = [c for c in feature_cols if c not in NUMERIC_COLS]

# --- 1. Train/test split, stratified on the target ---
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)

print("=== SPLIT ===")
print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
print("\ntrain class balance:")
print(train_df[TARGET_COL].value_counts(normalize=False))
print(train_df[TARGET_COL].value_counts(normalize=True).round(3))
print("\ntest class balance:")
print(test_df[TARGET_COL].value_counts(normalize=False))
print(test_df[TARGET_COL].value_counts(normalize=True).round(3))

# --- 2. Scaling, fit on training data only ---
scaler = StandardScaler()
scaler.fit(train_df[NUMERIC_COLS])

print("\n=== SCALING ===")
print(f"StandardScaler fit on TRAINING DATA ONLY: n_samples_seen_ = {scaler.n_samples_seen_} "
      f"(train set size is {len(train_df)}, test set size is {len(test_df)} - test was not seen)")
print(f"Numeric columns scaled: {NUMERIC_COLS}")
print(f"Categorical columns (encoding deferred to a later step): {categorical_cols}")

# --- 3. Cost metric ---
def cost(fn, fp, n):
    total = 5 * fn + 1 * fp
    return total, total / n

# --- 4. Two trivial baselines, scored on the training set ---
n_train = len(train_df)
n_bad_train = (train_df[TARGET_COL] == BAD).sum()
n_good_train = (train_df[TARGET_COL] == GOOD).sum()

# Approve everyone: predicts "good" for all -> every actual bad is a false negative.
approve_fn, approve_fp = n_bad_train, 0
approve_total, approve_avg = cost(approve_fn, approve_fp, n_train)

# Reject everyone: predicts "bad" for all -> every actual good is a false positive.
reject_fn, reject_fp = 0, n_good_train
reject_total, reject_avg = cost(reject_fn, reject_fp, n_train)

print("\n=== BASELINES (scored on training set) ===")
baseline_table = pd.DataFrame([
    {"rule": "approve everyone", "FN": approve_fn, "FP": approve_fp,
     "total_cost": approve_total, "avg_cost": round(approve_avg, 4)},
    {"rule": "reject everyone", "FN": reject_fn, "FP": reject_fp,
     "total_cost": reject_total, "avg_cost": round(reject_avg, 4)},
])
print(baseline_table.to_string(index=False))

cheaper = "approve everyone" if approve_avg < reject_avg else "reject everyone"
print(f"\nCheaper rule: '{cheaper}' (avg cost {min(approve_avg, reject_avg):.4f} per applicant)")
print(f"This is the baseline the SVC must beat on the cost metric.")
