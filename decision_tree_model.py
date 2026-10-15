import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("student_dataset.csv")
y = df["condition"]
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([("prep", prep), ("model", DecisionTreeClassifier(max_depth=12, min_samples_split=10, min_samples_leaf=4, random_state=42))])
model.fit(X_train, y_train)
pred = model.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, pred))
print("Macro-F1:", f1_score(y_valid, pred, average="macro"))
print(classification_report(y_valid, pred, digits=5))
test = pd.read_csv("test.csv")
model.fit(X, y)
pd.DataFrame({"unit_id": test["unit_id"], "condition": model.predict(test.drop(columns=["unit_id"], errors="ignore"))}).to_csv("decision_tree_tuned_submission.csv", index=False)
