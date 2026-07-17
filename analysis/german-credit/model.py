"""
Train linear and RBF SVCs, select hyperparameters by nested cross-validated
ROC AUC on the training set only, compare kernels, and report a training-set
CV checkpoint for the chosen model.

No cost weighting here (class_weight="balanced" is a class-frequency
correction only, not the 5x cost - that is applied at the threshold step, and
applying it here would double-count it). No threshold tuning. No test-set
evaluation - X_test/y_test are never constructed in this script, so the test
set stays sealed.

Positive class: bad credit (class == 2), relabeled to 1/0 for clarity so
predict_proba[:, 1] and roc_auc_score are unambiguous about which class they
score.
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
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

# Same split as baseline.py / encode.py. test_df exists only because
# train_test_split returns it - it is never referenced again below.
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

print("=== TRAINING DATA ===")
print(f"X_train shape: {X_train.shape}")
print(f"y_train positive rate (bad): {y_train.mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grids = {
    "linear": {"C": [0.01, 0.1, 1, 10, 100]},
    "rbf": {"C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1, 1]},
}

results = {}
for kernel, grid_params in param_grids.items():
    base_estimator = SVC(kernel=kernel, class_weight="balanced", probability=True, random_state=42)
    search = GridSearchCV(
        base_estimator, grid_params, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    best_std = search.cv_results_["std_test_score"][search.best_index_]
    results[kernel] = {
        "best_params": search.best_params_,
        "mean_auc": search.best_score_,
        "std_auc": best_std,
    }

    print(f"\n=== {kernel.upper()} GRID SEARCH ===")
    print(f"Best params: {search.best_params_}")
    print(f"CV ROC AUC: {search.best_score_:.4f} +/- {best_std:.4f}")

    c_grid = grid_params["C"]
    if search.best_params_["C"] in (c_grid[0], c_grid[-1]):
        print(f"FLAG: best C ({search.best_params_['C']}) is at the edge of the grid {c_grid} - "
              f"the true optimum may lie outside this range.")
    if "gamma" in grid_params:
        numeric_gammas = [g for g in grid_params["gamma"] if isinstance(g, (int, float))]
        best_gamma = search.best_params_["gamma"]
        if isinstance(best_gamma, (int, float)) and best_gamma in (min(numeric_gammas), max(numeric_gammas)):
            print(f"FLAG: best gamma ({best_gamma}) is at the edge of the numeric grid {numeric_gammas} - "
                  f"the true optimum may lie outside this range.")

print("\n=== LINEAR vs RBF ===")
lin, rbf = results["linear"], results["rbf"]
print(f"linear: {lin['mean_auc']:.4f} +/- {lin['std_auc']:.4f}  (params: {lin['best_params']})")
print(f"rbf:    {rbf['mean_auc']:.4f} +/- {rbf['std_auc']:.4f}  (params: {rbf['best_params']})")

lin_interval = (lin["mean_auc"] - lin["std_auc"], lin["mean_auc"] + lin["std_auc"])
rbf_interval = (rbf["mean_auc"] - rbf["std_auc"], rbf["mean_auc"] + rbf["std_auc"])
overlap = lin_interval[0] <= rbf_interval[1] and rbf_interval[0] <= lin_interval[1]

if overlap:
    print("Linear and RBF's mean +/- 1 std intervals overlap: the spread does not support "
          "crowning a winner. Preferring linear as the simpler model.")
    chosen_kernel = "linear"
else:
    chosen_kernel = "linear" if lin["mean_auc"] > rbf["mean_auc"] else "rbf"
    print(f"Intervals do not overlap: {chosen_kernel} has the higher CV ROC AUC outside the other's spread.")

print(f"\nCarrying forward: {chosen_kernel} (params: {results[chosen_kernel]['best_params']})")

# --- Checkpoint: refit chosen model on full training set, report CV performance ---
chosen_params = results[chosen_kernel]["best_params"]
final_estimator = SVC(
    kernel=chosen_kernel, class_weight="balanced", probability=True, random_state=42, **chosen_params
)

checkpoint_scores = cross_val_score(clone(final_estimator), X_train, y_train, cv=cv, scoring="roc_auc")
final_estimator.fit(X_train, y_train)  # fit on full training set, for later steps - not scored on test here

print(f"\n=== CHECKPOINT: {chosen_kernel} refit on full training set ===")
print(f"CV ROC AUC on training set: {checkpoint_scores.mean():.4f} +/- {checkpoint_scores.std():.4f}")
print(f"Per-fold scores: {[round(s, 4) for s in checkpoint_scores]}")
print("This is a training-set checkpoint, not a test-set evaluation. No threshold tuning or fairness audit yet.")
