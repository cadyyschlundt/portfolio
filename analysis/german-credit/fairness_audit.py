"""
Fairness audit: full model (age and sex as inputs) vs. dropped-attribute model,
compared on sex and age group error rates. Two different evaluation frames are
used deliberately, for two different purposes:

  - Group rates (Part A and Part B/4) use OUT-OF-FOLD predictions across ALL
    1000 rows (5-fold CV, refit per fold), so per-group cells are large enough
    to be stable. This is NOT the sealed train/test split.
  - Overall performance (Part B/5) reuses the ORIGINAL train/test split
    (random_state=42, train-only scaling/encoding), opened once, for a clean
    apples-to-apples comparison against the 0.4933 test-set cost from the
    threshold step.

Positive class: bad credit (class == 2). FN = approve an actual bad borrower
(cost 5). FP = deny an actual good borrower (cost 1). Model: calibrated linear
SVC, C=0.01, class_weight="balanced", method="sigmoid".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).parent / "german_credit_raw.csv"
TARGET_COL = "class"
BAD = 2
BEST_C = 0.01
FULL_MODEL_THRESHOLD = 0.20

NUMERIC_COLS = [
    "Attribute2", "Attribute5", "Attribute8", "Attribute11",
    "Attribute13", "Attribute16", "Attribute18",
]
ONEHOT_COLS = [
    "Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute9",
    "Attribute10", "Attribute12", "Attribute14", "Attribute15",
    "Attribute17", "Attribute19", "Attribute20",
]
EMPLOYMENT_COL = "Attribute7"
EMPLOYMENT_MAP = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}

REDUCED_NUMERIC_COLS = [c for c in NUMERIC_COLS if c != "Attribute13"]       # drop age
REDUCED_ONEHOT_COLS = [c for c in ONEHOT_COLS if c != "Attribute9"]          # drop personal status/sex

df = pd.read_csv(DATA_PATH)


def fit_transformers(source_df, numeric_cols, onehot_cols):
    scaler = StandardScaler()
    scaler.fit(source_df[numeric_cols])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
    encoder.fit(source_df[onehot_cols])
    return scaler, encoder


def transform_features(source_df, scaler, encoder, numeric_cols, onehot_cols):
    numeric = pd.DataFrame(
        scaler.transform(source_df[numeric_cols]), columns=numeric_cols, index=source_df.index
    )
    employment = source_df[[EMPLOYMENT_COL]].replace(EMPLOYMENT_MAP)
    onehot = pd.DataFrame(
        encoder.transform(source_df[onehot_cols]),
        columns=encoder.get_feature_names_out(onehot_cols), index=source_df.index,
    )
    return pd.concat([numeric, employment, onehot], axis=1)


def make_calibrated_model():
    base_svc = SVC(kernel="linear", C=BEST_C, class_weight="balanced", random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return CalibratedClassifierCV(base_svc, method="sigmoid", cv=inner_cv, ensemble=False)


def cost_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    return fn, fp, 5 * fn + fp


def sex_from_attribute9(attr9_series):
    return np.where(attr9_series == "A92", "female", "male")


def age_band(age_series):
    return pd.cut(
        age_series, bins=[-np.inf, 24, 40, np.inf], labels=["under_25", "25_to_40", "over_40"]
    )


def group_rate_table(y_true, preds, group_labels):
    rows = []
    for g in sorted(pd.unique(group_labels.astype(str))):
        idx = group_labels == g
        n = int(idx.sum())
        good_idx = idx & (y_true == 0)
        bad_idx = idx & (y_true == 1)
        n_good, n_bad = int(good_idx.sum()), int(bad_idx.sum())
        fp = int(((good_idx) & (preds == 1)).sum())
        fn = int(((bad_idx) & (preds == 0)).sum())
        fp_rate = fp / n_good if n_good else float("nan")
        fn_rate = fn / n_bad if n_bad else float("nan")
        rows.append({
            "group": g, "n": n,
            "n_good": n_good, "FP": fp, "FP_rate": round(fp_rate, 4),
            "n_bad": n_bad, "FN": fn, "FN_rate": round(fn_rate, 4),
        })
    return pd.DataFrame(rows)


def oof_predictions(numeric_cols, onehot_cols, threshold):
    """5-fold CV across ALL 1000 rows, refitting per fold. Returns (probs, preds)."""
    df_all = df.reset_index(drop=True)
    y_all = (df_all[TARGET_COL] == BAD).astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.empty(len(df_all))
    for tr_idx, val_idx in skf.split(df_all, y_all):
        train_fold, val_fold = df_all.iloc[tr_idx], df_all.iloc[val_idx]
        scaler_f, encoder_f = fit_transformers(train_fold, numeric_cols, onehot_cols)
        X_tr = transform_features(train_fold, scaler_f, encoder_f, numeric_cols, onehot_cols)
        X_val = transform_features(val_fold, scaler_f, encoder_f, numeric_cols, onehot_cols)
        model = make_calibrated_model()
        model.fit(X_tr, y_all[tr_idx])
        probs[val_idx] = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


def select_threshold(X_tr_full, y_tr_full):
    thresholds = np.round(np.arange(0.0, 1.001, 0.01), 2)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_costs = []
    for tr_idx, val_idx in outer_cv.split(X_tr_full, y_tr_full):
        X_tr, X_val = X_tr_full.iloc[tr_idx], X_tr_full.iloc[val_idx]
        y_tr, y_val = y_tr_full.iloc[tr_idx].values, y_tr_full.iloc[val_idx].values
        model = make_calibrated_model()
        model.fit(X_tr, y_tr)
        probs_val = model.predict_proba(X_val)[:, 1]
        n_val = len(y_val)
        fold_costs.append([cost_at_threshold(y_val, probs_val, t)[2] / n_val for t in thresholds])
    fold_costs = np.array(fold_costs)
    mean_costs, std_costs = fold_costs.mean(axis=0), fold_costs.std(axis=0)
    best_idx = int(np.argmin(mean_costs))
    return thresholds[best_idx], mean_costs[best_idx], std_costs[best_idx]


df_full = df.reset_index(drop=True)
y_all = (df_full[TARGET_COL] == BAD).astype(int).to_numpy()
sex_labels = sex_from_attribute9(df_full["Attribute9"])
age_labels = age_band(df_full["Attribute13"])

# =========================================================================
# PART A - audit the full model (age and sex included as inputs)
# =========================================================================
print("=== PART A: FULL MODEL (age and sex included), threshold = "
      f"{FULL_MODEL_THRESHOLD:.2f} ===")
print("Out-of-fold predictions across all 1000 rows, 5-fold CV, refit per fold.\n")

full_probs, full_preds = oof_predictions(NUMERIC_COLS, ONEHOT_COLS, FULL_MODEL_THRESHOLD)

print("--- 1. SEX (derived from Attribute9: male = A91+A93+A94, female = A92) ---")
print("REMINDER: A92 fuses divorced/separated/married women into one category, and there is "
      "no female-single category in this dataset. Sex is confounded with marital status here, "
      "so a male-female gap cannot be cleanly attributed to sex alone.\n")
sex_table_full = group_rate_table(y_all, full_preds, sex_labels)
print(sex_table_full.to_string(index=False))
male_row = sex_table_full[sex_table_full["group"] == "male"].iloc[0]
female_row = sex_table_full[sex_table_full["group"] == "female"].iloc[0]
print(f"\nMale-female FP rate gap: {abs(male_row.FP_rate - female_row.FP_rate):.4f} "
      f"(male {male_row.FP_rate:.4f}, female {female_row.FP_rate:.4f})")
print(f"Male-female FN rate gap: {abs(male_row.FN_rate - female_row.FN_rate):.4f} "
      f"(male {male_row.FN_rate:.4f}, female {female_row.FN_rate:.4f})")

print("\n--- 2. AGE (fixed bins: under_25, 25_to_40, over_40) ---")
age_table_full = group_rate_table(y_all, full_preds, age_labels.astype(str))
under25_n = age_table_full[age_table_full["group"] == "under_25"]["n"].iloc[0]
print(f"NOTE: under_25 group n = {under25_n} (smallest group - rates here are the least stable).")
print(age_table_full.to_string(index=False))
worst_fp = age_table_full.loc[age_table_full["FP_rate"].idxmax()]
worst_fn = age_table_full.loc[age_table_full["FN_rate"].idxmax()]
print(f"\nHighest FP rate: {worst_fp['group']} at {worst_fp['FP_rate']:.4f} (n_good={worst_fp['n_good']})")
print(f"Highest FN rate: {worst_fn['group']} at {worst_fn['FN_rate']:.4f} (n_bad={worst_fn['n_bad']})")

# =========================================================================
# PART B - drop protected attributes, refit
# =========================================================================
print("\n\n=== PART B: DROPPED-ATTRIBUTE MODEL (age and personal-status/sex removed) ===")

train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)
scaler_r, encoder_r = fit_transformers(train_df, REDUCED_NUMERIC_COLS, REDUCED_ONEHOT_COLS)
X_train_r = transform_features(train_df, scaler_r, encoder_r, REDUCED_NUMERIC_COLS, REDUCED_ONEHOT_COLS)
y_train_r = (train_df[TARGET_COL] == BAD).astype(int)

print("--- 3. Re-tune threshold from scratch (CV on training set only, reduced features) ---")
new_threshold, new_mean_cost, new_std_cost = select_threshold(X_train_r, y_train_r)
print(f"New chosen threshold: {new_threshold:.2f}")
print(f"CV cost at new threshold: {new_mean_cost:.4f} +/- {new_std_cost:.4f}")

print("\n--- 4. Group rates for the dropped-attribute model (out-of-fold, all 1000 rows) ---")
dropped_probs, dropped_preds = oof_predictions(REDUCED_NUMERIC_COLS, REDUCED_ONEHOT_COLS, new_threshold)

print("\nSEX:")
sex_table_dropped = group_rate_table(y_all, dropped_preds, sex_labels)
print(sex_table_dropped.to_string(index=False))

print("\nAGE:")
age_table_dropped = group_rate_table(y_all, dropped_preds, age_labels.astype(str))
print(age_table_dropped.to_string(index=False))

print("\n--- 5. Overall test-set cost, dropped-attribute model (test set opened once) ---")
X_test_r = transform_features(test_df, scaler_r, encoder_r, REDUCED_NUMERIC_COLS, REDUCED_ONEHOT_COLS)
y_test_r = (test_df[TARGET_COL] == BAD).astype(int).to_numpy()

final_dropped_model = make_calibrated_model()
final_dropped_model.fit(X_train_r, y_train_r)
test_probs_r = final_dropped_model.predict_proba(X_test_r)[:, 1]
fn_t, fp_t, total_t = cost_at_threshold(y_test_r, test_probs_r, new_threshold)
n_test = len(y_test_r)
avg_cost_dropped = total_t / n_test

print(f"Dropped-attribute model test-set average cost: {avg_cost_dropped:.4f} "
      f"(FN={fn_t}, FP={fp_t}, n={n_test})")
print(f"Full model test-set average cost (from threshold step): 0.4933")
delta = avg_cost_dropped - 0.4933
print(f"Difference: {delta:+.4f} ({'worse' if delta > 0 else 'better' if delta < 0 else 'identical'} "
      f"for the dropped-attribute model)")

# =========================================================================
# PART C - compare, read honestly
# =========================================================================
print("\n\n=== PART C: FULL vs DROPPED-ATTRIBUTE MODEL, SIDE BY SIDE ===")


def compare_tables(full_t, dropped_t, label):
    print(f"\n{label}:")
    merged = full_t.merge(dropped_t, on="group", suffixes=("_full", "_dropped"))
    for _, row in merged.iterrows():
        fp_shift = row["FP_rate_dropped"] - row["FP_rate_full"]
        fn_shift = row["FN_rate_dropped"] - row["FN_rate_full"]
        print(f"  {row['group']:>10} (n={row['n_full']}): "
              f"FP_rate {row['FP_rate_full']:.4f} -> {row['FP_rate_dropped']:.4f} ({fp_shift:+.4f}), "
              f"FN_rate {row['FN_rate_full']:.4f} -> {row['FN_rate_dropped']:.4f} ({fn_shift:+.4f})")


compare_tables(sex_table_full, sex_table_dropped, "SEX")
compare_tables(age_table_full, age_table_dropped, "AGE")

full_sex_gap = abs(male_row.FP_rate - female_row.FP_rate)
male_d = sex_table_dropped[sex_table_dropped["group"] == "male"].iloc[0]
female_d = sex_table_dropped[sex_table_dropped["group"] == "female"].iloc[0]
dropped_sex_gap = abs(male_d.FP_rate - female_d.FP_rate)

print(f"\nSex FP-rate gap: {full_sex_gap:.4f} (full model) -> {dropped_sex_gap:.4f} (dropped model)")
print(f"Overall cost: 0.4933 (full model) -> {avg_cost_dropped:.4f} (dropped model)")
