import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("student_dataset.csv")
y = df["condition"]
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([("prep", prep), ("model", GaussianNB())])
model.fit(X_train, y_train)
pred = model.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, pred))
print("Macro-F1:", f1_score(y_valid, pred, average="macro"))
print("Weighted-F1:", f1_score(y_valid, pred, average="weighted"))
print(classification_report(y_valid, pred, digits=5))
print(confusion_matrix(y_valid, pred))
test = pd.read_csv("test.csv")
model.fit(X, y)
pred_test = model.predict(test.drop(columns=["unit_id"], errors="ignore"))
pd.DataFrame({"unit_id": test["unit_id"], "condition": pred_test}).to_csv("gaussian_naive_bayes_submission.csv", index=False)
