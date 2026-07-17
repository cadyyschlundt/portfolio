"""
Generate public/german-credit/results.json - the only handoff from the Python
analysis to the Next.js page. Only the fine-grained cost-vs-threshold curve is
RECOMPUTED here (0.01 resolution, deterministic given random_state=42 and the
same model config as the threshold step). Every other value is a HARDCODED
constant carried over exactly from prior steps - each is printed, grouped and
labeled, before being written, so it can be checked against what was already
confirmed.

Positive class: bad credit (class == 2). Model: calibrated linear SVC, C=0.01,
class_weight="balanced", method="sigmoid".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ANALYSIS_DIR = Path(__file__).parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
DATA_PATH = ANALYSIS_DIR / "german_credit_raw.csv"
NOTES_PATH = REPO_ROOT / "german-credit-notes.md"
OUT_PATH = REPO_ROOT / "src" / "app" / "projects" / "german-credit" / "results.json"

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

# =========================================================================
# RECOMPUTED: cost-vs-threshold curve, training set only, same split/model
# as the threshold step
# =========================================================================
df = pd.read_csv(DATA_PATH)
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df[TARGET_COL], random_state=42
)

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


thresholds = np.round(np.arange(0.0, 1.001, 0.01), 2)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_costs = []
for tr_idx, val_idx in outer_cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx].values, y_train.iloc[val_idx].values
    model = make_calibrated_model()
    model.fit(X_tr, y_tr)
    probs_val = model.predict_proba(X_val)[:, 1]
    n_val = len(y_val)
    costs = []
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        fn = int(((y_val == 1) & (preds == 0)).sum())
        fp = int(((y_val == 0) & (preds == 1)).sum())
        costs.append((5 * fn + fp) / n_val)
    fold_costs.append(costs)

mean_costs = np.array(fold_costs).mean(axis=0)
cost_curve_points = [
    {"threshold": round(float(t), 2), "cost": round(float(c), 4)}
    for t, c in zip(thresholds, mean_costs)
]

idx_020 = int(np.argmin(np.abs(thresholds - 0.20)))
idx_050 = int(np.argmin(np.abs(thresholds - 0.50)))
cost_at_020 = round(float(mean_costs[idx_020]), 4)
cost_at_050 = round(float(mean_costs[idx_050]), 4)
pct_improvement = round((cost_at_050 - cost_at_020) / cost_at_050 * 100, 2)

print("=== RECOMPUTED: cost-vs-threshold curve ===")
print(f"{len(cost_curve_points)} points, threshold 0.00 to 1.00 step 0.01")
print(f"cost at threshold=0.20: {cost_at_020}  (previously reported: 0.5200)")
print(f"cost at threshold=0.50: {cost_at_050}  (previously reported: 0.9957)")
print(f"pct_improvement_vs_default (recomputed): {pct_improvement}%  (previously reported: 47.8%)")

cost_curve = {
    "points": cost_curve_points,
    "markers": {
        "minimum": {"threshold": 0.20, "cost": cost_at_020},
        "default": {"threshold": 0.50, "cost": cost_at_050},
    },
}

# =========================================================================
# HARDCODED: everything else, carried over exactly from prior steps
# =========================================================================
claim_text = NOTES_PATH.read_text(encoding="utf-8").strip()
print("\n=== HARDCODED: claim (verbatim from german-credit-notes.md) ===")
print(claim_text)

baseline = {
    "cost_rule": "reject everyone",
    "avg_cost": 0.70,
    "accuracy_rule": "approve everyone",
    "accuracy": 0.70,
    "note": ("These are different rules under different metrics that happen to share the "
             "same number: reject-everyone minimizes the 5x cost-weighted metric (0.70 avg "
             "cost), while approve-everyone maximizes plain accuracy (70%, the majority-class "
             "rate). They are not the same baseline."),
}
print("\n=== HARDCODED: baseline ===")
print(json.dumps(baseline, indent=2))

model_info = {
    "kernel": "linear",
    "C": 0.01,
    "class_weight": "balanced",
    "calibration": "sigmoid",
    "why_linear": ("Linear and RBF cross-validated ROC AUC were within one standard deviation "
                   "of each other (linear 0.7739 +/- 0.0318 vs RBF 0.7783 +/- 0.0303), so "
                   "linear was chosen as the simpler model."),
}
print("\n=== HARDCODED: model ===")
print(json.dumps(model_info, indent=2))

threshold_info = {
    "chosen": 0.20,
    "default": 0.50,
    "cv_cost_at_chosen": cost_at_020,
    "cv_cost_at_default": cost_at_050,
    "pct_improvement_vs_default": pct_improvement,
}
print("\n=== threshold (uses the recomputed curve's values, for internal consistency) ===")
print(json.dumps(threshold_info, indent=2))

test_result = {
    "threshold": 0.20,
    "confusion": {"TN": 107, "FP": 103, "FN": 9, "TP": 81},
    "total_cost": 148,
    "avg_cost": 0.4933,
    "accuracy": 0.6267,
    "roc_auc": 0.7991,
    "beats_baseline": True,
    "pct_reduction_vs_baseline": 29.5,
}
print("\n=== HARDCODED: test_result ===")
print(json.dumps(test_result, indent=2))

fairness = {
    "frame_note": ("Group rates (full_model and dropped_model sex/age tables) are "
                    "cross-validated out-of-fold predictions across all 1000 rows, refitting "
                    "per fold, so group sizes are stable. Test costs (test_result above, and "
                    "dropped_model.test_avg_cost below) are single evaluations on the sealed "
                    "300-row test set, opened once each."),
    "full_model": {
        "sex": [
            {"group": "female", "n": 310, "n_good": 201, "fp_rate": 0.5323, "n_bad": 109, "fn_rate": 0.1193},
            {"group": "male", "n": 690, "n_good": 499, "fp_rate": 0.4389, "n_bad": 191, "fn_rate": 0.1728},
        ],
        "age": [
            {"group": "under_25", "n": 149, "n_good": 88, "fp_rate": 0.6932, "n_bad": 61, "fn_rate": 0.0656},
            {"group": "25_to_40", "n": 577, "n_good": 410, "fp_rate": 0.4829, "n_bad": 167, "fn_rate": 0.1617},
            {"group": "over_40", "n": 274, "n_good": 202, "fp_rate": 0.3317, "n_bad": 72, "fn_rate": 0.2083},
        ],
    },
    "dropped_model": {
        "threshold": 0.19,
        "test_avg_cost": 0.53,
        "sex": [
            {"group": "female", "n": 310, "n_good": 201, "fp_rate": 0.4677, "n_bad": 109, "fn_rate": 0.1560},
            {"group": "male", "n": 690, "n_good": 499, "fp_rate": 0.4870, "n_bad": 191, "fn_rate": 0.1204},
        ],
        "age": [
            {"group": "under_25", "n": 149, "n_good": 88, "fp_rate": 0.6591, "n_bad": 61, "fn_rate": 0.0820},
            {"group": "25_to_40", "n": 577, "n_good": 410, "fp_rate": 0.4780, "n_bad": 167, "fn_rate": 0.1497},
            {"group": "over_40", "n": 274, "n_good": 202, "fp_rate": 0.4109, "n_bad": 72, "fn_rate": 0.1389},
        ],
    },
    "comparison_note": (
        "Sex disparity mostly disappeared when the attribute was dropped: the male-female "
        "FP-rate gap fell from 0.0934 to 0.0193 (an 80% reduction), consistent with the full "
        "model using personal-status/sex fairly directly. Age disparity barely changed: "
        "under-25's FP rate went from 0.6932 to 0.6591, still far above the other age bands, "
        "indicating other features (employment duration, credit history length, residence "
        "duration, existing credit count) act as proxies for age. Removing both attributes "
        "also cost real performance: overall test-set average cost rose from 0.4933 to 0.5300, "
        "a 7.4% relative increase."
    ),
    "confounds": (
        "Sex is derived from a field fused with marital status (A92 covers divorced, "
        "separated, and married women together, with no female-single category), so any sex "
        "gap is confounded with marital status, not attributable to sex alone. Age has no "
        "single input to drop cleanly: features like employment duration, credit history "
        "length, residence duration, and existing credit count correlate with age and can "
        "proxy for it."
    ),
}
print("\n=== HARDCODED: fairness ===")
print(json.dumps(fairness, indent=2))

limitations = [
    "Dataset has only 1000 rows total; group subsets are small and rates are noisy, especially under-25 (n=88 good applicants).",
    "Sex is derived from a fused personal-status-and-sex field with no female-single category, confounding sex with marital status.",
    "Age disparity persists after dropping the age feature, likely via correlated proxy features (employment duration, credit history length, residence duration, existing credit count).",
    "Data collected in Germany in 1994; may not reflect current lending patterns or populations.",
    "Removing protected attributes increased overall cost by 7.4% relative (0.4933 to 0.5300), a real fairness-performance tradeoff, not a free fix.",
]
print("\n=== HARDCODED: limitations ===")
for item in limitations:
    print(f"- {item}")

results = {
    "claim": claim_text,
    "baseline": baseline,
    "model": model_info,
    "threshold": threshold_info,
    "cost_curve": cost_curve,
    "test_result": test_result,
    "fairness": fairness,
    "limitations": limitations,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n=== WROTE {OUT_PATH} ===")

with open(OUT_PATH, encoding="utf-8") as f:
    reparsed = json.load(f)
print("Valid JSON: parsed successfully with the standard library json module.")

print("\n=== FULL FILE CONTENTS ===")
print(json.dumps(reparsed, indent=2))
