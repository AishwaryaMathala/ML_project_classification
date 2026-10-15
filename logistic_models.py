import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("student_dataset.csv")
y = df["condition"]
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
test = pd.read_csv("test.csv")
X_test = test.drop(columns=["unit_id"], errors="ignore")
models = {"baseline": LogisticRegression(max_iter=2000, C=1.0, random_state=42), "tuned": LogisticRegression(max_iter=3000, C=2.0, random_state=42)}
for name, estimator in models.items():
    model = Pipeline([("prep", prep), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    print(name, "Accuracy:", accuracy_score(y_valid, pred), "Macro-F1:", f1_score(y_valid, pred, average="macro"))
    print(classification_report(y_valid, pred, digits=5))
    model.fit(X, y)
    pd.DataFrame({"unit_id": test["unit_id"], "condition": model.predict(X_test)}).to_csv(f"logistic_regression_{name}_submission.csv", index=False)
