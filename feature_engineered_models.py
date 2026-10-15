import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def add_features(df):
    data = df.copy()
    if {"process_temp", "ambient_temp"}.issubset(data.columns):
        data["temperature_gap"] = data["process_temp"] - data["ambient_temp"]
        data["temperature_mean"] = (data["process_temp"] + data["ambient_temp"]) / 2
    if {"drive_torque", "shaft_speed"}.issubset(data.columns):
        data["torque_speed_product"] = data["drive_torque"] * data["shaft_speed"]
    ratios = [("power_load", "drive_torque", "power_torque_ratio"), ("thermal_stress", "process_temp", "stress_temperature_ratio"), ("mechanical_load", "component_wear", "wear_load_ratio"), ("vibration_index", "shaft_speed", "vibration_speed_ratio"), ("coolant_flow", "power_load", "coolant_power_ratio")]
    for a, b, name in ratios:
        if a in data and b in data:
            den = data[b].where(data[b].abs() > 1e-8, np.nan)
            data[name] = data[a] / den
    if {"power_load", "shaft_speed"}.issubset(data.columns):
        data["power_speed_product"] = data["power_load"] * data["shaft_speed"]
    return data.replace([np.inf, -np.inf], np.nan)

df = pd.read_csv("student_dataset.csv")
test = pd.read_csv("test.csv")
y = df["condition"]
X = add_features(df.drop(columns=["condition", "unit_id"], errors="ignore"))
X_test = add_features(test.drop(columns=["unit_id"], errors="ignore")).reindex(columns=X.columns)
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat)])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {"engineered_random_forest": RandomForestClassifier(n_estimators=500, max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1), "engineered_extra_trees": ExtraTreesClassifier(n_estimators=700, max_features=0.9, min_samples_leaf=1, random_state=42, n_jobs=-1)}
for name, estimator in models.items():
    model = Pipeline([("prep", prep), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    print(name, "Accuracy:", accuracy_score(y_valid, pred), "Macro-F1:", f1_score(y_valid, pred, average="macro"))
    print(classification_report(y_valid, pred, digits=5))
    model.fit(X, y)
    pd.DataFrame({"unit_id": test["unit_id"], "condition": model.predict(X_test)}).to_csv(f"{name}_submission.csv", index=False)
