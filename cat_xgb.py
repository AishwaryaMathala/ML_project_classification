"""Train the current best OOF blend and create a NEW Kaggle submission.

Expected in the same folder:
  student_dataset.csv
  test.csv
  sample_submission.csv

Creates:
  submission_blend_catboost_d5_xgb_d4.csv
  submission_catboost_d5_new.csv
  submission_xgb_depth4_new.csv

Does NOT overwrite your original catboost_submission.csv or
submission_catboost_d6_base.csv.

Install dependencies if needed:
  pip install pandas numpy scikit-learn catboost xgboost
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "student_dataset.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
TARGET = "condition"
ID_COL = "unit_id"
SEED = 42
CAT_WEIGHT = 0.75
XGB_WEIGHT = 0.25

# ---- Load data ----
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

if TARGET not in train.columns:
    raise ValueError(f"Missing target column {TARGET!r} in {TRAIN_PATH}")
if ID_COL not in train.columns or ID_COL not in test.columns:
    raise ValueError(f"Expected ID column {ID_COL!r} in both train and test")
if TARGET not in sample.columns or ID_COL not in sample.columns:
    raise ValueError(f"Sample submission must contain {ID_COL!r} and {TARGET!r}")

# Preserve original row order. IDs are not used as model features.
y = train[TARGET].astype(int)
X = train.drop(columns=[TARGET, ID_COL]).copy()
X_test = test.drop(columns=[ID_COL]).copy()
test_ids = test[ID_COL].copy()

# Ensure identical feature columns and order.
missing = set(X.columns) - set(X_test.columns)
extra = set(X_test.columns) - set(X.columns)
if missing or extra:
    raise ValueError(f"Train/test feature mismatch. Missing in test: {missing}; extra in test: {extra}")
X_test = X_test[X.columns]

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
classes = np.sort(y.unique())
n_classes = len(classes)
print(f"Train shape: {X.shape}; test shape: {X_test.shape}")
print(f"Classes: {classes.tolist()}; categorical columns: {cat_cols}")

# ---- CatBoost depth 5, matching the experiment configuration ----
try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise SystemExit("CatBoost is missing. Install it with: pip install catboost") from exc

cat_model = CatBoostClassifier(
    iterations=1800,
    depth=5,
    learning_rate=0.04,
    l2_leaf_reg=5,
    random_strength=1,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    random_seed=SEED,
    verbose=100,
    allow_writing_files=False,
    thread_count=-1,
)

X_cat = X.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_cat[col] = X_cat[col].fillna("__MISSING__").astype(str)
    X_test_cat[col] = X_test_cat[col].fillna("__MISSING__").astype(str)

print("\nFitting CatBoost depth 5 on all training data...")
cat_model.fit(X_cat, y, cat_features=cat_cols)
cat_probs_raw = cat_model.predict_proba(X_test_cat)
cat_probs = np.zeros((len(X_test), n_classes), dtype=float)
for j, cls in enumerate(cat_model.classes_):
    cat_probs[:, np.where(classes == cls)[0][0]] = cat_probs_raw[:, j]

# ---- XGBoost depth 4, with the same preprocessing/configuration as CV ----
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit("XGBoost is missing. Install it with: pip install xgboost") from exc

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:  # compatibility with older scikit-learn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), cat_cols),
    ],
    remainder="drop",
)

xgb_model = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5,
        reg_alpha=0.05,
        gamma=0.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )),
])

print("\nFitting XGBoost depth 4 on all training data...")
xgb_model.fit(X, y)
xgb_probs_raw = xgb_model.predict_proba(X_test)
xgb_classes = xgb_model.named_steps["model"].classes_
xgb_probs = np.zeros((len(X_test), n_classes), dtype=float)
for j, cls in enumerate(xgb_classes):
    xgb_probs[:, np.where(classes == cls)[0][0]] = xgb_probs_raw[:, j]

# ---- Blend probabilities using the best OOF weight found so far: 75:25 ----
blend_probs = CAT_WEIGHT * cat_probs + XGB_WEIGHT * xgb_probs
blend_pred = classes[np.argmax(blend_probs, axis=1)]
cat_pred = classes[np.argmax(cat_probs, axis=1)]
xgb_pred = classes[np.argmax(xgb_probs, axis=1)]

# Build each submission from sample_submission to preserve required format/order.
def save_submission(predictions, filename):
    sub = sample.copy()
    if len(sub) != len(predictions):
        raise ValueError(f"Sample submission has {len(sub)} rows but predictions have {len(predictions)}")
    # Use test IDs in test.csv order, which should match sample_submission order.
    sub[ID_COL] = test_ids.to_numpy()
    sub[TARGET] = predictions.astype(int)
    sub.to_csv(DATA_DIR / filename, index=False)
    print(f"Saved: {filename}")
    print(sub.head())

save_submission(cat_pred, "submission_catboost_d5_new.csv")
save_submission(xgb_pred, "submission_xgb_depth4_new.csv")
save_submission(blend_pred, "submission_blend_catboost_d5_xgb_d4.csv")
print("\nDone. Your original submissions were not overwritten.")
print(f"Blend weights: CatBoost={CAT_WEIGHT:.2f}, XGBoost={XGB_WEIGHT:.2f}")

