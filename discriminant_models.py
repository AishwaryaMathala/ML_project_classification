import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("student_dataset.csv")
y = df["condition"]
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
A = prep.fit_transform(X_train)
B = prep.transform(X_valid)
models = {"lda": LinearDiscriminantAnalysis(), "qda_default": QuadraticDiscriminantAnalysis(), "qda_regularized_01": QuadraticDiscriminantAnalysis(reg_param=0.1)}
test = pd.read_csv("test.csv")
for name, model in models.items():
    model.fit(A, y_train)
    pred = model.predict(B)
    print(name, "Accuracy:", accuracy_score(y_valid, pred), "Macro-F1:", f1_score(y_valid, pred, average="macro"))
    print(classification_report(y_valid, pred, digits=5))
    prep.fit(X)
    all_x = prep.transform(X)
    test_x = prep.transform(test.drop(columns=["unit_id"], errors="ignore"))
    model.fit(all_x, y)
    pd.DataFrame({"unit_id": test["unit_id"], "condition": model.predict(test_x)}).to_csv(f"{name}_submission.csv", index=False)
