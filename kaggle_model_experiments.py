"""
Kaggle tabular classification experiment runner
----------------------------------------------
Expected files in the same folder:
  student_dataset.csv   training data, target column: condition
  test.csv              test data, includes unit_id
  sample_submission.csv sample submission with unit_id, condition

Install the optional boosting libraries you want to try:
  pip install pandas numpy scikit-learn catboost xgboost lightgbm

This script:
  1. Keeps the existing catboost_submission.csv untouched.
  2. Evaluates multiple model configurations with the same stratified CV folds.
  3. Reports macro-F1 (change SCORING if Kaggle uses a different metric).
  4. Saves OOF probabilities, a results table, and a submission for each model.
  5. Creates a probability blend of the best CV candidates for further testing.

IMPORTANT:
CV is a screening tool. Kaggle's public score can differ. Do not assume a higher
CV score guarantees beating the saved 0.89125 submission.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# -------------------- SETTINGS --------------------
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "student_dataset.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

TARGET = "condition"
ID_COL = "unit_id"
N_SPLITS = 5
SEED = 42
SCORING = "macro_f1"  # Confirm this matches the competition metric.
ORIGINAL_SCORE = 0.89125

# -------------------- LOAD DATA --------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} was not found in {TRAIN_PATH}")
if ID_COL not in test.columns:
    raise ValueError(f"ID column {ID_COL!r} was not found in {TEST_PATH}")

y = train[TARGET].astype(int)
X = train.drop(columns=[TARGET])
X_test = test.copy()

# Remove ID from model features; retain it for the final submission.
train_ids = X[ID_COL].copy() if ID_COL in X.columns else None
test_ids = X_test[ID_COL].copy()
X = X.drop(columns=[ID_COL], errors="ignore")
X_test = X_test.drop(columns=[ID_COL], errors="ignore")

# Match train/test columns and order.
missing_test = set(X.columns) - set(X_test.columns)
extra_test = set(X_test.columns) - set(X.columns)
if missing_test or extra_test:
    raise ValueError(f"Train/test feature mismatch. Missing in test: {missing_test}; extra in test: {extra_test}")
X_test = X_test[X.columns]

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
classes = np.sort(y.unique())
n_classes = len(classes)

print(f"Train: {X.shape}; test: {X_test.shape}; classes: {classes.tolist()}")
print(f"Categorical columns: {cat_cols}")
print(f"Original best Kaggle score to beat: {ORIGINAL_SCORE:.5f}")
print("Metric used below: macro F1. Confirm this is the competition metric.\n")

# -------------------- PREPROCESSORS --------------------
# Tree models do not need scaling. Impute numeric values and one-hot encode categoricals.
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:  # compatibility with older scikit-learn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

sparse_preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), cat_cols),
    ],
    remainder="drop",
)

# Dense, ordinal-encoded version for sklearn estimators that need dense arrays.
ordinal_preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ],
    remainder="drop",
)

# -------------------- OPTIONAL LIBRARIES --------------------
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed; skipping CatBoost. Install with: pip install catboost")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed; skipping XGBoost. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed; skipping LightGBM. Install with: pip install lightgbm")

# -------------------- MODEL CANDIDATES --------------------
# Each candidate is fit/evaluated using exactly the same CV folds.
candidates = []

if HAS_CATBOOST:
    cat_configs = [
        # Baseline close to your previous strong setup
        ("catboost_d6_base", dict(iterations=1500, depth=6, learning_rate=0.04,
             l2_leaf_reg=5, random_strength=1, bootstrap_type="Bernoulli",
             subsample=0.8)),
        ("catboost_d5", dict(iterations=1800, depth=5, learning_rate=0.04,
             l2_leaf_reg=5, random_strength=1, bootstrap_type="Bernoulli",
             subsample=0.8)),
        ("catboost_d7", dict(iterations=1500, depth=7, learning_rate=0.035,
             l2_leaf_reg=5, random_strength=1, bootstrap_type="Bernoulli",
             subsample=0.8)),
        ("catboost_d8_reg", dict(iterations=1400, depth=8, learning_rate=0.03,
             l2_leaf_reg=7, random_strength=1.2, bootstrap_type="Bernoulli",
             subsample=0.8)),
        ("catboost_d6_reg10", dict(iterations=1800, depth=6, learning_rate=0.035,
             l2_leaf_reg=10, random_strength=1.2, bootstrap_type="Bernoulli",
             subsample=0.8)),
    ]
    for name, params in cat_configs:
        def make_cat(params=params):
            return CatBoostClassifier(
                **params,
                loss_function="MultiClass",
                eval_metric="TotalF1:average=Macro",
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
        candidates.append((name, "catboost", make_cat))

if HAS_XGBOOST:
    xgb_configs = [
        ("xgb_depth4", dict(n_estimators=1200, max_depth=4, learning_rate=0.035,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5, reg_alpha=0.05, gamma=0.0)),
        ("xgb_depth5", dict(n_estimators=1000, max_depth=5, learning_rate=0.03,
            min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=7, reg_alpha=0.1, gamma=0.0)),
        ("xgb_depth3_more_trees", dict(n_estimators=1600, max_depth=3, learning_rate=0.025,
            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=5, reg_alpha=0.0, gamma=0.0)),
    ]
    for name, params in xgb_configs:
        def make_xgb(params=params):
            return Pipeline([
                ("prep", sparse_preprocessor),
                ("model", XGBClassifier(
                    **params, objective="multi:softprob", num_class=n_classes,
                    eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
                    tree_method="hist",
                )),
            ])
        candidates.append((name, "sklearn", make_xgb))

if HAS_LIGHTGBM:
    lgb_configs = [
        ("lightgbm_leaves31", dict(n_estimators=1200, learning_rate=0.025,
            num_leaves=31, max_depth=-1, min_child_samples=25,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5, reg_alpha=0.05)),
        ("lightgbm_leaves20", dict(n_estimators=1500, learning_rate=0.025,
            num_leaves=20, max_depth=-1, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=7, reg_alpha=0.1)),
        ("lightgbm_depth7", dict(n_estimators=1000, learning_rate=0.03,
            num_leaves=24, max_depth=7, min_child_samples=30,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5, reg_alpha=0.0)),
    ]
    for name, params in lgb_configs:
        def make_lgb(params=params):
            return Pipeline([
                ("prep", sparse_preprocessor),
                ("model", LGBMClassifier(
                    **params, objective="multiclass", random_state=SEED,
                    n_jobs=-1, verbosity=-1,
                )),
            ])
        candidates.append((name, "sklearn", make_lgb))

# Strong non-boosting baselines; these can sometimes add value in a blend.
candidates.extend([
    ("extra_trees", "sklearn", lambda: Pipeline([
        ("prep", sparse_preprocessor),
        ("model", ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=1, max_features=0.9,
            class_weight=None, random_state=SEED, n_jobs=-1,
        )),
    ])),
    ("random_forest", "sklearn", lambda: Pipeline([
        ("prep", sparse_preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=1, max_features=0.9,
            class_weight=None, random_state=SEED, n_jobs=-1,
        )),
    ])),
    ("hist_gradient_boosting", "sklearn", lambda: Pipeline([
        ("prep", ordinal_preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=25,
            l2_regularization=2.0, random_state=SEED,
        )),
    ])),
])

# GradientBoostingClassifier needs dense OHE output. Include as another comparison.
try:
    dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

dense_preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", dense_ohe),
        ]), cat_cols),
    ],
    remainder="drop",
)
candidates.append(("sklearn_gradient_boosting", "sklearn", lambda: Pipeline([
    ("prep", dense_preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.04, max_depth=2,
        min_samples_leaf=10, random_state=SEED,
    )),
])))

if not candidates:
    raise RuntimeError("No models available. Install scikit-learn and at least one boosting library.")

# -------------------- CROSS-VALIDATION --------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
results = []
oof_by_model = {}
folds = list(cv.split(X, y))

for model_name, model_type, factory in candidates:
    print(f"Training {model_name} ...")
    oof = np.zeros((len(X), n_classes), dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = factory()

        if model_type == "catboost":
            # CatBoost receives original columns and handles categorical features directly.
            X_tr = X.iloc[tr_idx].copy()
            X_va = X.iloc[va_idx].copy()
            for col in cat_cols:
                X_tr[col] = X_tr[col].fillna("__MISSING__").astype(str)
                X_va[col] = X_va[col].fillna("__MISSING__").astype(str)
            model.fit(X_tr, y.iloc[tr_idx], cat_features=cat_cols)
            probs = model.predict_proba(X_va)
        else:
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = model.predict_proba(X.iloc[va_idx])

        # Align probability columns with the global class order.
        model_classes = model.classes_ if model_type == "catboost" else model.named_steps["model"].classes_
        aligned = np.zeros((len(va_idx), n_classes), dtype=float)
        for j, cls in enumerate(model_classes):
            aligned[:, np.where(classes == cls)[0][0]] = probs[:, j]
        oof[va_idx] = aligned

        pred = classes[np.argmax(aligned, axis=1)]
        fold_score = f1_score(y.iloc[va_idx], pred, average="macro")
        fold_scores.append(fold_score)
        print(f"  fold {fold}: macro-F1={fold_score:.5f}")

    overall = f1_score(y, classes[np.argmax(oof, axis=1)], average="macro")
    oof_by_model[model_name] = oof
    results.append({
        "model": model_name,
        "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "std_fold_macro_f1": float(np.std(fold_scores)),
        "oof_macro_f1": float(overall),
        "fold_scores": ", ".join(f"{s:.5f}" for s in fold_scores),
    })
    print(f"  OOF macro-F1={overall:.5f}\n")

results_df = pd.DataFrame(results).sort_values("oof_macro_f1", ascending=False)
results_df.to_csv("experiment_results.csv", index=False)
print("\n================ RESULTS (sorted by OOF macro-F1) ================")
print(results_df[["model", "mean_fold_macro_f1", "std_fold_macro_f1", "oof_macro_f1"]].to_string(index=False))
print("\nSaved experiment_results.csv")

# Save OOF probability arrays for later blending / error analysis.
np.savez_compressed(
    "oof_probabilities.npz",
    **{name: probs for name, probs in oof_by_model.items()}
)

# -------------------- TRAIN TOP MODELS ON ALL DATA --------------------
# Create one submission per candidate, so no new experiment overwrites the original.
def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

submission_predictions = {}
for model_name, model_type, factory in candidates:
    print(f"Fitting full-data model for submission: {model_name}")
    model = factory()

    if model_type == "catboost":
        X_full = X.copy()
        X_test_full = X_test.copy()
        for col in cat_cols:
            X_full[col] = X_full[col].fillna("__MISSING__").astype(str)
            X_test_full[col] = X_test_full[col].fillna("__MISSING__").astype(str)
        model.fit(X_full, y, cat_features=cat_cols)
        probs = model.predict_proba(X_test_full)
    else:
        model.fit(X, y)
        probs = model.predict_proba(X_test)

    model_classes = model.classes_ if model_type == "catboost" else model.named_steps["model"].classes_
    aligned = np.zeros((len(X_test), n_classes), dtype=float)
    for j, cls in enumerate(model_classes):
        aligned[:, np.where(classes == cls)[0][0]] = probs[:, j]
    submission_predictions[model_name] = aligned
    preds = classes[np.argmax(aligned, axis=1)]

    # Preserve sample_submission row order if possible.
    out = pd.DataFrame({ID_COL: test_ids, TARGET: preds})
    if ID_COL in sample.columns:
        out = sample[[ID_COL]].merge(out, on=ID_COL, how="left", validate="one_to_one")
    out[TARGET] = out[TARGET].astype(int)
    out.to_csv(f"submission_{safe_name(model_name)}.csv", index=False)

print("\nSaved one submission CSV per model. The existing catboost_submission.csv was not modified.")

# -------------------- BLENDS: OOF-BASED SHORTLIST --------------------
# Average the probability predictions of the top 2 or top 3 models by OOF score.
# This is a useful candidate, but blending weights must be chosen with care to avoid
# overfitting the validation folds. Compare on Kaggle before replacing your best.
top_names = results_df["model"].head(min(3, len(results_df))).tolist()
if len(top_names) >= 2:
    for k in [2, 3]:
        names = top_names[:min(k, len(top_names))]
        blend_oof = np.mean([oof_by_model[n] for n in names], axis=0)
        blend_score = f1_score(y, classes[np.argmax(blend_oof, axis=1)], average="macro")
        blend_test = np.mean([submission_predictions[n] for n in names], axis=0)
        blend_preds = classes[np.argmax(blend_test, axis=1)]

        out = pd.DataFrame({ID_COL: test_ids, TARGET: blend_preds})
        if ID_COL in sample.columns:
            out = sample[[ID_COL]].merge(out, on=ID_COL, how="left", validate="one_to_one")
        out[TARGET] = out[TARGET].astype(int)
        out.to_csv(f"submission_blend_top{k}.csv", index=False)
        print(f"Blend top {k}: OOF macro-F1={blend_score:.5f}; models={names}")

print("\nDONE.")
print("1) Keep catboost_submission.csv untouched.")
print("2) Inspect experiment_results.csv.")
print("3) Submit only a small shortlist of promising CSVs.")
print("4) Compare Kaggle scores against 0.89125; save any better file under a new name.")
