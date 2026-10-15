from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_csv("student_dataset.csv")
test = pd.read_csv("test.csv")
y = train["condition"].copy()
X = train.drop(columns=["condition", "unit_id"], errors="ignore")
X_test = test.drop(columns=["unit_id"], errors="ignore")
numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
A = prep.fit_transform(X_train)
B = prep.transform(X_valid)
C = prep.transform(X_test)
out = Path("preprocessed")
out.mkdir(exist_ok=True)
cols = prep.get_feature_names_out()
pd.DataFrame(A, columns=cols).to_csv(out / "X_train.csv", index=False)
pd.DataFrame(B, columns=cols).to_csv(out / "X_valid.csv", index=False)
pd.DataFrame(C, columns=cols).to_csv(out / "X_test.csv", index=False)
y_train.to_csv(out / "y_train.csv", index=False)
y_valid.to_csv(out / "y_valid.csv", index=False)
print(A.shape, B.shape, C.shape)
