"""
Encode the 13 string-coded categorical features so an SVC can use them later.
Reuses the exact split from baseline.py (same random_state). No classifier is
trained here.

Positive class: bad credit (class == 2).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "german_credit_raw.csv"
TARGET_COL = "class"

NUMERIC_COLS = [
    "Attribute2", "Attribute5", "Attribute8", "Attribute11",
    "Attribute13", "Attribute16", "Attribute18",
]

# Unordered: one binary column per category, no assumed ordering.
ONEHOT_COLS = {
    "Attribute1": "checking account status",
    "Attribute3": "credit history",
    "Attribute4": "purpose",
    "Attribute6": "savings account/bonds",
    "Attribute9": "personal status and sex",   # protected attribute
    "Attribute10": "other debtors/guarantors",
    "Attribute12": "property",
    "Attribute14": "other installment plans",
    "Attribute15": "housing",
    "Attribute17": "job",
    "Attribute19": "telephone",
    "Attribute20": "foreign worker",
}

# Ordered: unemployed is the low end, longest tenure is the high end. Not
# learned from data - this ordering is a fixed real-world fact about the
# category codes, so it is applied identically to train and test.
EMPLOYMENT_COL = "Attribute7"
EMPLOYMENT_ORDER = ["A71", "A72", "A73", "A74", "A75"]
EMPLOYMENT_LABELS = ["unemployed", "<1 year", "1-4 years", "4-7 years", ">=7 years"]
EMPLOYMENT_MAP = {code: rank for rank, code in enumerate(EMPLOYMENT_ORDER)}

df = pd.read_csv(DATA_PATH)

# Same split as baseline.py: test_size=0.3, stratify on target, random_state=42.
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)

print("=== SPLIT (reused from baseline.py) ===")
print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")

# --- Numeric columns: same train-only scaler as baseline.py ---
scaler = StandardScaler()
scaler.fit(train_df[NUMERIC_COLS])
train_numeric = pd.DataFrame(
    scaler.transform(train_df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=train_df.index
)
test_numeric = pd.DataFrame(
    scaler.transform(test_df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=test_df.index
)

# --- Ordinal: present employment since ---
train_employment = train_df[[EMPLOYMENT_COL]].replace(EMPLOYMENT_MAP)
test_employment = test_df[[EMPLOYMENT_COL]].replace(EMPLOYMENT_MAP)

print(f"\n=== ORDINAL ENCODING: {EMPLOYMENT_COL} (present employment since) ===")
print("Order applied (low to high):")
for code, label, rank in zip(EMPLOYMENT_ORDER, EMPLOYMENT_LABELS, range(5)):
    print(f"  {code} ({label}) -> {rank}")

# --- One-hot: fit on training data only ---
onehot_source_cols = list(ONEHOT_COLS.keys())
encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
encoder.fit(train_df[onehot_source_cols])
onehot_feature_names = encoder.get_feature_names_out(onehot_source_cols)

train_onehot = pd.DataFrame(
    encoder.transform(train_df[onehot_source_cols]),
    columns=onehot_feature_names, index=train_df.index,
)
test_onehot = pd.DataFrame(
    encoder.transform(test_df[onehot_source_cols]),
    columns=onehot_feature_names, index=test_df.index,
)

print(f"\n=== ONE-HOT ENCODING ===")
print("OneHotEncoder fit on TRAINING DATA ONLY, then applied unchanged to test.")
print("\nOld column -> new columns:")
for col, desc in ONEHOT_COLS.items():
    new_cols = [c for c in onehot_feature_names if c.startswith(f"{col}_")]
    print(f"  {col} ({desc}) -> {new_cols}")

# --- Assemble final feature matrices ---
X_train = pd.concat([train_numeric, train_employment, train_onehot], axis=1)
X_test = pd.concat([test_numeric, test_employment, test_onehot], axis=1)

print("\n=== FINAL FEATURE MATRIX ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

print("\n=== PROTECTED ATTRIBUTE COLUMNS (for the later drop-and-refit test) ===")
age_cols = ["Attribute13"]
sex_cols = [c for c in onehot_feature_names if c.startswith("Attribute9_")]
print(f"Age: {age_cols} (numeric, scaled, in the NUMERIC_COLS block)")
print(f"Personal status and sex: {sex_cols}")
