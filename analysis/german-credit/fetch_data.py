"""
Fetch the Statlog German Credit Data (UCI id 144) via ucimlrepo and save a
local CSV copy. This is a data-only step: no cleaning, encoding, scaling,
splitting, or modeling happens here.
"""

import re
import sys
from pathlib import Path

import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    print(f"ucimlrepo import failed: {e}")
    sys.exit(1)

try:
    dataset = fetch_ucirepo(id=144)
except Exception as e:
    print(f"fetch_ucirepo(id=144) failed: {e}")
    sys.exit(1)

X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)

out_path = Path(__file__).parent / "german_credit_raw.csv"
df.to_csv(out_path, index=False)
print(f"Saved raw data to {out_path}")

print("\n=== SHAPE ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== VARIABLE DOCUMENTATION (from UCI metadata) ===")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 120)
print(dataset.variables[["name", "role", "type", "description"]])

target_col = y.columns[0]
print(f"\n=== TARGET COLUMN: {target_col} ===")
print("Value counts (raw encoding):")
print(df[target_col].value_counts(dropna=False))
target_desc = dataset.variables.loc[
    dataset.variables["name"] == target_col, "description"
]
print("\nDocumented meaning of this column:")
print(target_desc.to_string(index=False) if not target_desc.empty else "(no description found)")

print("\n=== MISSING VALUES PER COLUMN ===")
print(df.isnull().sum())

# Column headers are Attribute1..Attribute20 (not descriptive), so the
# readable meaning only exists in dataset.variables' description column.
# Look columns up by description text, not by header substring.
desc_by_name = dict(zip(dataset.variables["name"], dataset.variables["description"]))

def find_column(keyword):
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    return [name for name, desc in desc_by_name.items() if pattern.search(str(desc))]

print("\n=== AGE COLUMN ===")
age_cols = find_column("age")
print(f"Column(s) matching whole word 'age' in description: {[(c, desc_by_name[c]) for c in age_cols]}")
for col in age_cols:
    print(f"\n{col} ({desc_by_name[col]}).describe():")
    print(df[col].describe())

print("\n=== SEX / MARITAL STATUS (COMBINED) COLUMN ===")
personal_cols = find_column("personal status and sex")
print(f"Column(s) matching 'personal status and sex' in description: {[(c, desc_by_name[c]) for c in personal_cols]}")
for col in personal_cols:
    print(f"\n{col} ({desc_by_name[col]}) value_counts:")
    print(df[col].value_counts(dropna=False))
