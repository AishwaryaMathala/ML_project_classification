"""Generate EDA summaries and figures for the condition classification project.

Expected files beside this script:
  student_dataset.csv, test.csv
Outputs:
  eda_summary.txt
  eda_target_distribution.png
  eda_numeric_distributions.png
  eda_numeric_correlation.png
  eda_unit_type_counts.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
train = pd.read_csv(BASE / "student_dataset.csv")
test = pd.read_csv(BASE / "test.csv")
TARGET = "condition"
ID_COL = "unit_id"

X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

summary = []
summary.append(f"Training shape: {train.shape}")
summary.append(f"Test shape: {test.shape}")
summary.append(f"Training duplicate rows: {int(train.duplicated().sum())}")
summary.append(f"Training missing cells: {int(train.isna().sum().sum())}")
summary.append(f"Test missing cells: {int(test.isna().sum().sum())}")
summary.append("\\nMissing values by training column:")
summary.extend(train.isna().sum().to_string().splitlines())
summary.append("\\nTarget counts:")
summary.extend(train[TARGET].value_counts(dropna=False).sort_index().to_string().splitlines())
summary.append("\\nNumeric summary:")
summary.append(X[num_cols].describe().round(3).to_string())
summary.append("\\nCategorical cardinality:")
for col in cat_cols:
    summary.append(f"{col}: {X[col].nunique(dropna=False)} unique values")
(BASE / "eda_summary.txt").write_text("\\n".join(summary), encoding="utf-8")

# Target distribution
ax = train[TARGET].value_counts().sort_index().plot(kind="bar", rot=0, figsize=(6,4), title="Target Distribution: condition")
ax.set_xlabel("condition")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(BASE / "eda_target_distribution.png", dpi=160)
plt.close()

# Numeric feature distributions
if num_cols:
    ncols = 2
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows))
    axes = axes.flatten()
    for ax, col in zip(axes, num_cols):
        ax.hist(X[col].dropna(), bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    for ax in axes[len(num_cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(BASE / "eda_numeric_distributions.png", dpi=160)
    plt.close(fig)

# Correlation heatmap
if len(num_cols) >= 2:
    corr = X[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Predictors")
    fig.tight_layout()
    fig.savefig(BASE / "eda_numeric_correlation.png", dpi=160)
    plt.close(fig)

# Categorical distribution
for col in cat_cols:
    counts = X[col].fillna("__MISSING__").astype(str).value_counts().head(30)
    ax = counts.plot(kind="bar", figsize=(7,4), title=f"Category counts: {col}", rot=30)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    safe = "".join(ch if ch.isalnum() else "_" for ch in col)
    plt.savefig(BASE / f"eda_{safe}_counts.png", dpi=160)
    plt.close()

print(f"Saved EDA outputs in {BASE}")
print("Review eda_summary.txt and the PNGs before writing numeric EDA findings into the report.")
