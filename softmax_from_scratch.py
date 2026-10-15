import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def fit_model(X, y, lr=0.05, epochs=2500, reg=0.001):
    classes = np.sort(np.unique(y))
    idx = np.searchsorted(classes, y)
    Y = np.eye(len(classes))[idx]
    X = np.c_[np.ones(len(X)), X]
    W = np.zeros((X.shape[1], len(classes)))
    for _ in range(epochs):
        P = softmax(X @ W)
        grad = X.T @ (P - Y) / len(X)
        grad[1:] += reg * W[1:]
        W -= lr * grad
    return W, classes

def predict_model(X, W, classes):
    return classes[np.argmax(softmax(np.c_[np.ones(len(X)), X] @ W), axis=1)]

df = pd.read_csv("student_dataset.csv")
y = df["condition"].to_numpy()
X = df.drop(columns=["condition", "unit_id"], errors="ignore")
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
A = prep.fit_transform(X_train)
B = prep.transform(X_valid)
W, classes = fit_model(A, y_train)
pred = predict_model(B, W, classes)
print("Accuracy:", accuracy_score(y_valid, pred))
print("Macro-F1:", f1_score(y_valid, pred, average="macro"))
prep.fit(X)
W, classes = fit_model(prep.transform(X), y)
test = pd.read_csv("test.csv")
T = prep.transform(test.drop(columns=["unit_id"], errors="ignore"))
pd.DataFrame({"unit_id": test["unit_id"], "condition": predict_model(T, W, classes)}).to_csv("scratch_softmax_regression_submission.csv", index=False)
