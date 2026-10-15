"""Evaluate weighted OOF probability blends for the multiclass condition task.

Place this script beside student_dataset.csv and oof_probabilities.npz, then run:
    python blend_oof_experiments.py

It does not overwrite any existing submission files.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "student_dataset.csv"
OOF_PATH = DATA_DIR / "oof_probabilities.npz"
TARGET = "condition"

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Could not find {TRAIN_PATH}. Put this script beside student_dataset.csv.")
if not OOF_PATH.exists():
    raise FileNotFoundError(f"Could not find {OOF_PATH}.")

train = pd.read_csv(TRAIN_PATH)
y = train[TARGET].to_numpy()
npz = np.load(OOF_PATH)
probs = {name: npz[name] for name in npz.files}

for name, p in probs.items():
    if len(p) != len(y):
        raise ValueError(f"Row mismatch for {name}: {len(p)} OOF rows vs {len(y)} labels. Check row order/data version.")

# Keep the most promising diverse candidates from the current experiment results.
candidates = [
    "catboost_d6_base", "catboost_d5", "catboost_d6_reg10",
    "xgb_depth3_more_trees", "xgb_depth4", "xgb_depth5",
]
candidates = [name for name in candidates if name in probs]

rows = []
def score(p):
    return f1_score(y, np.argmax(p, axis=1), average="macro")

for name, p in probs.items():
    rows.append({"kind": "single", "model_1": name, "weight_1": 1.0,
                 "model_2": "", "weight_2": 0.0, "oof_macro_f1": score(p)})

# Pairwise blends, testing 5% increments. Weight_1 is the share assigned to model_1.
for a, b in combinations(candidates, 2):
    for w in np.arange(0.0, 1.0001, 0.05):
        blend = w * probs[a] + (1.0 - w) * probs[b]
        rows.append({"kind": "pair", "model_1": a, "weight_1": round(float(w), 2),
                     "model_2": b, "weight_2": round(float(1-w), 2),
                     "oof_macro_f1": score(blend)})

results = pd.DataFrame(rows).sort_values("oof_macro_f1", ascending=False)
out = DATA_DIR / "blend_oof_results.csv"
results.to_csv(out, index=False)
print("\nTop 25 OOF blend results (macro F1):")
print(results.head(25).to_string(index=False))
print(f"\nSaved all results to: {out}")
print("\nReminder: these are OOF validation results. Do not blend hard-label submission CSVs; final test blending should use probability predictions from refitted models.")

