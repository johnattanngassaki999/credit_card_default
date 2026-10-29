import joblib
import time
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pathlib import Path
from .evaluate import cross_validate_model



def build_pipeline(preprocesser:Pipeline, model) -> Pipeline:
    pipeline = [
        ("prep",preprocesser),
        ("model", model)
    ]
    return pipeline

def get_model(random_state: int) -> dict:
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=50),
        "linear_svm": LinearSVC(
            class_weight="balanced", max_iter=5000, random_state=random_state),
    }
    return models

def tune_model(pipeline: Pipeline, param_grid: dict, X_train, y_train,
               n_splits: int , random_state: int) -> tuple[Pipeline, dict]:
    """Grid search with stratified CV on the training set only, scored by PR-AUC.

    Returns the best pipeline (already refitted on all of X_train) and its parameters.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid, scoring="average_precision",
                          cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def compare_models(pipelines: dict[str, Pipeline], X_train, y_train,
                   n_splits: int, random_state: int) -> pd.DataFrame:
    """One row per model: PR-AUC, ROC-AUC (mean, std), time, and gain over the dummy.

    `pipelines` must contain a "dummy" entry, which is the baseline.
    """
    rows = []
    for name, pipeline in pipelines.items():
        start = time.perf_counter()
        scores = cross_validate_model(pipeline, X_train, y_train, n_splits, random_state)
        rows.append({
            "model": name,
            "pr_auc_mean": scores["pr_auc"]["mean"],
            "pr_auc_std": scores["pr_auc"]["std"],
            "roc_auc_mean": scores["roc_auc"]["mean"],
            "roc_auc_std": scores["roc_auc"]["std"],
            "time_s": time.perf_counter() - start,
        })
    table = pd.DataFrame(rows).set_index("model")
    table["pr_auc_gain_vs_dummy"] = table["pr_auc_mean"] - table.loc["dummy", "pr_auc_mean"]
    return table


def save_model(pipeline: Pipeline, threshold: float, path: str | Path) -> None:
    """Save the full pipeline together with its threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, path)


def load_model(path: str | Path) -> tuple[Pipeline, float]:
    """Load a pipeline and its threshold saved by save_model."""
    saved = joblib.load(Path(path))
    return saved["pipeline"], saved["threshold"]


    
