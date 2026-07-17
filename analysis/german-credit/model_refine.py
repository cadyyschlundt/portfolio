"""
Close the edge-of-grid C flag, replace the deprecated SVC(probability=True)
with CalibratedClassifierCV, and check calibration. Linear kernel only, chosen
in the modeling step. No threshold tuning, no cost, no test-set evaluation, no
fairness audit - cost is applied at the threshold step, later.

Positive class: bad credit (class == 2), relabeled to 1/0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).parent / "german_credit_raw.csv"
TARGET_COL = "class"
BAD = 2

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

df = pd.read_csv(DATA_PATH)

# Same split as every prior step. test_df is never touched again below.
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)

scaler = StandardScaler()
scaler.fit(train_df[NUMERIC_COLS])
train_numeric = pd.DataFrame(
    scaler.transform(train_df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=train_df.index
)

train_employment = train_df[[EMPLOYMENT_COL]].replace(EMPLOYMENT_MAP)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
encoder.fit(train_df[ONEHOT_COLS])
train_onehot = pd.DataFrame(
    encoder.transform(train_df[ONEHOT_COLS]),
    columns=encoder.get_feature_names_out(ONEHOT_COLS), index=train_df.index,
)

X_train = pd.concat([train_numeric, train_employment, train_onehot], axis=1)
y_train = (train_df[TARGET_COL] == BAD).astype(int)  # 1 = bad (positive), 0 = good

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 1. Close the edge-of-grid flag: lower C range ---
c_grid = [0.001, 0.003, 0.01, 0.03, 0.1]
search = GridSearchCV(
    SVC(kernel="linear", class_weight="balanced", random_state=42),  # no probability=True: AUC uses decision_function
    {"C": c_grid},
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)

print("=== 1. LOWER-C GRID SEARCH (linear SVC) ===")
results_table = pd.DataFrame({
    "C": c_grid,
    "mean_auc": [search.cv_results_["mean_test_score"][i] for i in range(len(c_grid))],
    "std_auc": [search.cv_results_["std_test_score"][i] for i in range(len(c_grid))],
})
print(results_table.to_string(index=False))

best_c = search.best_params_["C"]
best_mean = search.best_score_
best_std = search.cv_results_["std_test_score"][search.best_index_]
print(f"\nBest C: {best_c}, CV ROC AUC: {best_mean:.4f} +/- {best_std:.4f}")

if best_c == c_grid[0]:
    print(f"FLAG: best C ({best_c}) is again at the low edge of the grid {c_grid}.")
else:
    aucs = results_table["mean_auc"].tolist()
    spread = max(aucs) - min(aucs)
    print(f"Best C ({best_c}) is not at the edge. AUC across the grid ranges "
          f"{min(aucs):.4f} to {max(aucs):.4f} (spread {spread:.4f}) - "
          f"{'a plateau, as expected' if spread < best_std else 'more movement than a flat plateau'}.")

# --- 2. Switch to CalibratedClassifierCV ---
base_svc = SVC(kernel="linear", C=best_c, class_weight="balanced", random_state=42)
calibrated_model = CalibratedClassifierCV(base_svc, method="sigmoid", cv=cv, ensemble=False)
calibrated_model.fit(X_train, y_train)

print("\n=== 2. CALIBRATION SWITCH ===")
print(f"Replaced SVC(probability=True) with CalibratedClassifierCV("
      f"SVC(kernel='linear', C={best_c}, class_weight='balanced'), method='sigmoid', ensemble=False).")
print("Fit on X_train/y_train only (700 rows). Test set not touched.")

# --- 3. Calibration check on cross-validated training predictions ---
oof_probs = cross_val_predict(calibrated_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

bins = pd.qcut(oof_probs, q=10, duplicates="drop")
calib_df = pd.DataFrame({"predicted_prob": oof_probs, "actual_bad": y_train.values, "bin": bins})
summary = calib_df.groupby("bin", observed=True).agg(
    predicted_prob=("predicted_prob", "mean"),
    actual_rate=("actual_bad", "mean"),
    count=("actual_bad", "size"),
).reset_index(drop=True)
summary["predicted_prob"] = summary["predicted_prob"].round(3)
summary["actual_rate"] = summary["actual_rate"].round(3)
summary["gap"] = (summary["predicted_prob"] - summary["actual_rate"]).round(3)

print("\n=== 3. CALIBRATION TABLE (cross-validated, out-of-fold predictions) ===")
print(summary.to_string(index=False))

mean_abs_gap = summary["gap"].abs().mean()
print(f"\nMean absolute gap (predicted - actual) across bins: {mean_abs_gap:.3f}")
overall_pred = oof_probs.mean()
overall_actual = y_train.mean()
print(f"Overall mean predicted probability: {overall_pred:.3f} vs overall actual bad rate: {overall_actual:.3f}")
print("class_weight='balanced' reweights the loss as if classes were ~50/50, so probabilities are "
      "expected to skew away from the true ~30% base rate. Reading the table above, report honestly "
      "whether that skew is present and how large it is - this is not being treated as an error.")
