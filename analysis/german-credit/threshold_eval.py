"""
Threshold selection (on training data only, via nested CV) and the final,
one-time test-set evaluation. This is the only script that opens the test set,
and it does so once, after the threshold is already chosen.

Positive class: bad credit (class == 2), relabeled to 1/0.
False negative: predicted good, actually bad -> cost 5.
False positive: predicted bad, actually good -> cost 1.
Model: calibrated linear SVC, C=0.01, class_weight="balanced", method="sigmoid".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).parent / "german_credit_raw.csv"
TARGET_COL = "class"
BAD = 2
BEST_C = 0.01

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
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)

# Scaler and encoder are FIT on training data only. Fitting does not touch
# test rows' values, only train_df's - the test set is still sealed here.
scaler = StandardScaler()
scaler.fit(train_df[NUMERIC_COLS])
encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
encoder.fit(train_df[ONEHOT_COLS])


def build_features(source_df):
    numeric = pd.DataFrame(
        scaler.transform(source_df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=source_df.index
    )
    employment = source_df[[EMPLOYMENT_COL]].replace(EMPLOYMENT_MAP)
    onehot = pd.DataFrame(
        encoder.transform(source_df[ONEHOT_COLS]),
        columns=encoder.get_feature_names_out(ONEHOT_COLS), index=source_df.index,
    )
    return pd.concat([numeric, employment, onehot], axis=1)


X_train = build_features(train_df)
y_train = (train_df[TARGET_COL] == BAD).astype(int)

def make_calibrated_model():
    base_svc = SVC(kernel="linear", C=BEST_C, class_weight="balanced", random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return CalibratedClassifierCV(base_svc, method="sigmoid", cv=inner_cv, ensemble=False)


def cost_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    return fn, fp, 5 * fn + fp


# --- 1. Threshold selection via CV, training set only ---
thresholds = np.round(np.arange(0.0, 1.001, 0.01), 2)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_costs = []  # (n_folds, n_thresholds)
for tr_idx, val_idx in outer_cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx].values, y_train.iloc[val_idx].values

    model = make_calibrated_model()
    model.fit(X_tr, y_tr)
    probs_val = model.predict_proba(X_val)[:, 1]
    n_val = len(y_val)

    costs = []
    for t in thresholds:
        _, _, total = cost_at_threshold(y_val, probs_val, t)
        costs.append(total / n_val)
    fold_costs.append(costs)

fold_costs = np.array(fold_costs)
mean_costs = fold_costs.mean(axis=0)
std_costs = fold_costs.std(axis=0)

best_idx = int(np.argmin(mean_costs))
tied_idx = np.where(np.isclose(mean_costs, mean_costs[best_idx]))[0]
best_threshold = thresholds[best_idx]

print("=== 1. THRESHOLD SELECTION (cross-validated, training set only) ===")
print(f"Chosen threshold: {best_threshold:.2f}")
print(f"CV cost at chosen threshold: {mean_costs[best_idx]:.4f} +/- {std_costs[best_idx]:.4f}")
if len(tied_idx) > 1:
    tied_range = thresholds[tied_idx]
    print(f"Note: {len(tied_idx)} thresholds tie for lowest CV cost, range "
          f"[{tied_range.min():.2f}, {tied_range.max():.2f}]. Lowest threshold in the tie reported above.")

# --- 2. Full cost-vs-threshold curve ---
print("\n=== 2. COST-VS-THRESHOLD CURVE (every 0.05, plus the chosen threshold and 0.5) ===")
report_thresholds = sorted(set(np.round(np.arange(0.0, 1.001, 0.05), 2)) | {best_threshold, 0.5})
rows = []
for t in report_thresholds:
    idx = int(np.argmin(np.abs(thresholds - t)))
    rows.append({
        "threshold": thresholds[idx],
        "mean_cv_cost": round(mean_costs[idx], 4),
        "std": round(std_costs[idx], 4),
        "marker": ("<- CHOSEN" if thresholds[idx] == best_threshold else "") +
                  (" <- DEFAULT (0.5)" if thresholds[idx] == 0.5 else ""),
    })
curve_table = pd.DataFrame(rows)
print(curve_table.to_string(index=False))

default_idx = int(np.argmin(np.abs(thresholds - 0.5)))
print(f"\nDefault threshold (0.5) CV cost: {mean_costs[default_idx]:.4f} +/- {std_costs[default_idx]:.4f}")
print(f"Chosen threshold ({best_threshold:.2f}) CV cost: {mean_costs[best_idx]:.4f} +/- {std_costs[best_idx]:.4f}")
improvement = mean_costs[default_idx] - mean_costs[best_idx]
print(f"Choosing the threshold deliberately saves {improvement:.4f} average cost per applicant "
      f"vs leaving it at 0.5 ({improvement / mean_costs[default_idx] * 100:.1f}% reduction).")

# --- 3. Open the test set, once. Final model refit on the full training set. ---
X_test = build_features(test_df)
y_test = (test_df[TARGET_COL] == BAD).astype(int).values

final_model = make_calibrated_model()
final_model.fit(X_train, y_train)
test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

tp = int(((y_test == 1) & (test_preds == 1)).sum())
tn = int(((y_test == 0) & (test_preds == 0)).sum())
fp = int(((y_test == 0) & (test_preds == 1)).sum())
fn = int(((y_test == 1) & (test_preds == 0)).sum())
n_test = len(y_test)
total_cost = 5 * fn + fp
avg_cost = total_cost / n_test

print(f"\n=== 3. TEST SET (opened once, threshold = {best_threshold:.2f}) ===")
print(f"n_test = {n_test}")
print("Confusion matrix (positive = bad credit):")
print(f"  TP (predicted bad, actually bad):   {tp}")
print(f"  TN (predicted good, actually good): {tn}")
print(f"  FP (predicted bad, actually good):  {fp}  <- applicant harm, cost 1 each")
print(f"  FN (predicted good, actually bad):  {fn}  <- lender's costly error, cost 5 each")
print(f"Total cost: 5*{fn} + 1*{fp} = {total_cost}")
print(f"Average cost per applicant: {avg_cost:.4f}")

# --- 4. Verdict ---
accuracy = (tp + tn) / n_test
auc = roc_auc_score(y_test, test_probs)
baseline = 0.70

print("\n=== 4. VERDICT ===")
print(f"Test-set average cost: {avg_cost:.4f}")
print(f"Baseline (reject-everyone): {baseline:.4f}")
diff = baseline - avg_cost
pct = diff / baseline * 100
if avg_cost < baseline:
    print(f"BEATS the baseline by {diff:.4f} average cost per applicant ({pct:.1f}% lower).")
else:
    print(f"DOES NOT beat the baseline. Model cost is {avg_cost - baseline:.4f} higher "
          f"({-pct:.1f}% worse) than reject-everyone.")
print(f"\nFor context only (cost is the metric the claim is judged on, not these):")
print(f"  Test accuracy: {accuracy:.4f}")
print(f"  Test ROC AUC:  {auc:.4f}")
