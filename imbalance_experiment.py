from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from data import CREDIT_COLS, TARGET
from data import load_credit, clean, split_data
ALL_FEATURES = [c for c in CREDIT_COLS if c != TARGET]
BASE_CAT = ["SEX", "EDUCATION", "MARRIAGE"]
PAY = [f"PAY_{i}" for i in range(1, 7)]

df= cleaned(load.credit())
X = df
y = df[TARGET]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["average_precision", "f1", "recall", "precision"]

def make_model(strategy):
    if strategy == "none":
        return Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=1000))])
    if strategy == "class_weight":
        return Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    if strategy == "smote":
        return Pipeline([("scale", StandardScaler()),
                         ("smote", SMOTE(random_state=42)),
                         ("model", LogisticRegression(max_iter=1000))])

feature_sets = {
    "original only":       base_cols,
    "original + created":  base_cols + eng_cols,
}

rows = []
for fs_name, cols in feature_sets.items():
    for strat in ["none", "class_weight", "smote"]:
        res = cross_validate(make_model(strat), X[cols], y, cv=cv, scoring=scoring)
        rows.append({
            "features": fs_name,
            "imbalance": strat,
            "PR-AUC":    res["test_average_precision"].mean(),
            "F1":        res["test_f1"].mean(),
            "recall":    res["test_recall"].mean(),
            "precision": res["test_precision"].mean(),
            "F1 std":    res["test_f1"].std(),
        })

results = pd.DataFrame(rows).round(3)