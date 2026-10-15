
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "student_dataset.csv"
TEST_PATH = BASE_DIR / "test.csv"
SAMPLE_PATH = BASE_DIR / "sample_submission.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "condition"
ID_COL = "unit_id"

RANDOM_STATE = 42
N_SPLITS = 5

CATBOOST_PARAMS = {
    "iterations": 1500,
    "depth": 6,
    "learning_rate": 0.04,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "thread_count": -1,
    "allow_writing_files": False,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8
}


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

if TARGET not in train_df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

if TARGET not in sample_submission.columns:
    raise ValueError(
        f"Missing target column in sample submission: {TARGET}"
    )

if len(test_df) != len(sample_submission):
    raise ValueError("Test and submission row counts do not match.")

if train_df[TARGET].isna().any():
    raise ValueError("Training target contains missing values.")


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

X = train_df.drop(
    columns=[TARGET, ID_COL],
    errors="ignore"
).copy()

y = train_df[TARGET].copy()

# Preserve the test row order and match training feature columns.
X_test = test_df.reindex(columns=X.columns).copy()

categorical_cols = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

print("Feature count:", X.shape[1])
print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)
print("\nClass distribution:")
print(y.value_counts().sort_index())

# Convert categorical values to strings and represent missing values.
for col in categorical_cols:
    X[col] = X[col].fillna("__MISSING__").astype(str)
    X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

# Convert numeric values safely.
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# CatBoost can handle NaN in numerical columns.
X[numeric_cols] = X[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_test[numeric_cols] = X_test[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)


# ============================================================
# 4. TRAIN FINAL MODEL AND SAVE SUBMISSION FIRST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL CATBOOST MODEL")
print("=" * 60)

final_model = CatBoostClassifier(**CATBOOST_PARAMS)

final_model.fit(
    X,
    y,
    cat_features=categorical_cols,
    verbose=100
)

# Save the trained model.
model_path = OUTPUT_DIR / "best_catboost_model.cbm"
final_model.save_model(str(model_path))

print("Model saved to:", model_path)


# ============================================================
# 5. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("PREDICTING TEST DATA")
print("=" * 60)

test_predictions = final_model.predict(X_test).ravel()

print("\nTest prediction distribution:")
print(
    pd.Series(test_predictions)
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. CREATE KAGGLE SUBMISSION
# ============================================================

print("\n" + "=" * 60)
print("CREATING KAGGLE SUBMISSION")
print("=" * 60)

submission = sample_submission.copy()

if len(test_predictions) != len(submission):
    raise ValueError("Prediction count does not match submission rows.")

# Assign predictions while preserving the sample submission's row order.
submission[TARGET] = test_predictions

submission_path = OUTPUT_DIR / "catboost_submission.csv"

submission.to_csv(
    submission_path,
    index=False
)

# Verify that the CSV exists and has the expected structure.
check_submission = pd.read_csv(submission_path)

assert len(check_submission) == len(test_df)
assert TARGET in check_submission.columns
assert not check_submission[TARGET].isna().any()

print("\nSUCCESS! Submission saved.")
print("File:", submission_path)
print("Shape:", check_submission.shape)
print("\nFirst 10 rows:")
print(check_submission.head(10))


# ============================================================
# 7. FIVE-FOLD CROSS-VALIDATION
# ============================================================

# This happens AFTER saving the submission. If validation fails,
# the submission file has already been written.

print("\n" + "=" * 60)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Initialize with valid target values to avoid unknown/uninitialized
# entries in the out-of-fold prediction array.
oof_predictions = y.to_numpy().copy()

fold_f1_scores = []
fold_accuracy_scores = []

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):
    print(f"\nFOLD {fold}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostClassifier(**CATBOOST_PARAMS)

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=150,
        verbose=100
    )

    valid_predictions = model.predict(X_valid).ravel()

    oof_predictions[valid_idx] = valid_predictions

    fold_f1 = f1_score(
        y_valid,
        valid_predictions,
        average="macro",
        zero_division=0
    )

    fold_accuracy = accuracy_score(
        y_valid,
        valid_predictions
    )

    fold_f1_scores.append(fold_f1)
    fold_accuracy_scores.append(fold_accuracy)

    print(f"Fold {fold} macro F1: {fold_f1:.5f}")
    print(f"Fold {fold} accuracy: {fold_accuracy:.5f}")

    model.save_model(
        str(OUTPUT_DIR / f"catboost_fold_{fold}.cbm")
    )


# ============================================================
# 8. OVERALL VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

overall_f1 = f1_score(
    y,
    oof_predictions,
    average="macro",
    zero_division=0
)

overall_accuracy = accuracy_score(y, oof_predictions)

print("Individual fold macro F1:", np.round(fold_f1_scores, 5))
print("Mean fold macro F1:", round(np.mean(fold_f1_scores), 5))
print("Standard deviation:", round(np.std(fold_f1_scores), 5))
print("Overall OOF macro F1:", round(overall_f1, 5))
print("Overall OOF accuracy:", round(overall_accuracy, 5))

print("\nClassification report:")
print(
    classification_report(
        y,
        oof_predictions,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y, oof_predictions))

# Save out-of-fold predictions.
oof_df = pd.DataFrame({
    "actual": y.to_numpy(),
    "predicted": oof_predictions
})

oof_df.to_csv(
    OUTPUT_DIR / "catboost_oof_predictions.csv",
    index=False
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "catboost_feature_importance.csv",
    index=False
)


# ============================================================
# 10. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print("Kaggle submission:")
print(submission_path)
print("Upload catboost_submission.csv to Kaggle.")
