
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)


# ==================================================
# 1. FILE PATHS — CHANGE THESE IF NEEDED
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "student_dataset.csv"
TEST_PATH = BASE_DIR / "test.csv"
SAMPLE_PATH = BASE_DIR / "sample_submission.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "condition"
ID_COL = "unit_id"


# ==================================================
# 2. LOAD DATA
# ==================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_PATH)

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ==================================================
# 3. FEATURES AND TARGET
# ==================================================

X = train_df.drop(columns=[TARGET, ID_COL], errors="ignore")
y = train_df[TARGET]

# Align test features with training columns
X_test = test_df.reindex(columns=X.columns)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


# ==================================================
# 4. PREPROCESSING
# ==================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# ==================================================
# 5. BASE MODEL
# ==================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])


# ==================================================
# 6. HYPERPARAMETER SEARCH
# ==================================================

param_distributions = {
    "classifier__n_estimators": [200, 300, 400, 600, 800],
    "classifier__max_depth": [15, 20, 25, 30, 40, None],
    "classifier__min_samples_split": [2, 5, 10, 15, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 6, 8],
    "classifier__max_features": ["sqrt", 0.5, 0.7, 1.0],
    "classifier__class_weight": [None, "balanced"],
    "classifier__bootstrap": [True]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=35,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)


# ==================================================
# 7. RUN SEARCH
# ==================================================

print("\nStarting hyperparameter search...")
search.fit(X, y)

print("\nBEST PARAMETERS:")
print(search.best_params_)

print("\nBEST 5-FOLD CV MACRO F1:")
print(round(search.best_score_, 5))


# ==================================================
# 8. SAVE SEARCH RESULTS
# ==================================================

results = pd.DataFrame(search.cv_results_)

results = results.sort_values("rank_test_score")

results.to_csv(
    OUTPUT_DIR / "classification_tuning_results.csv",
    index=False
)

print("\nTop 10 parameter combinations:")
print(
    results[
        [
            "rank_test_score",
            "mean_train_score",
            "mean_test_score",
            "std_test_score",
            "params"
        ]
    ].head(10).to_string(index=False)
)


# ==================================================
# 9. BEST MODEL IS ALREADY REFITTED ON ALL X, y
# ==================================================

best_model = search.best_estimator_

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_classification_model.joblib"
)


# ==================================================
# 10. PREDICT TEST DATA WITH BEST MODEL
# ==================================================

predictions = best_model.predict(X_test)

print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts().sort_index())


# ==================================================
# 11. GENERATE SUBMISSION
# ==================================================

if len(submission) != len(predictions):
    raise ValueError(
        "Sample submission and test predictions have different row counts."
    )

if TARGET not in submission.columns:
    raise ValueError(
        f"'{TARGET}' column is missing from sample_submission.csv"
    )

submission[TARGET] = predictions

output_path = OUTPUT_DIR / "best_classification_submission.csv"

submission.to_csv(output_path, index=False)

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nSubmission shape:", submission.shape)
print("Saved submission:", output_path)
print("Saved best model:", OUTPUT_DIR / "best_classification_model.joblib")
print("Saved tuning results:", OUTPUT_DIR / "classification_tuning_results.csv")
print("\nDONE!")
