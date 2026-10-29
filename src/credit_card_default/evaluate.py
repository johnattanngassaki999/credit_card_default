import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, roc_auc_score, average_precision_score )



def compute_metrics (y_true, y_proba, threshold=0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    return {"precision":precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc}

def cross_validate_model (model, X, y, n_splits: int, random_state: int) -> dict[str,dict[str,float]]:
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scores = cross_validate(model, X, y, cv=cv, scoring=["average_precision", "roc_auc"])
    return {
        "pr_auc": {"mean": scores["test_average_precision"].mean(),
                   "std": scores["test_average_precision"].std()},
        "roc_auc": {"mean": scores["test_roc_auc"].mean(),
                    "std": scores["test_roc_auc"].std()},
    }

def get_oof_proba(model, X, y, n_splits: int, random_state: int):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:,1]

def find_best_threshold(y_true, y_proba, metric="f1", min_recall=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1] 
    if metric == "f1":
        scores = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    elif metric == "min_recall":
        if min_recall is None:
            raise ValueError("min_recall must be provided with metric='min_recall'")
        scores = np.where(recall >= min_recall, precision, -1.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(thresholds[np.argmax(scores)])

def plot_pr_curve(y_true, y_proba, label=None, threshold=None, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, label=f"{label or 'model'} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(np.mean(y_true), color="gray", linestyle="--", label="baseline (default rate)")

    if threshold is not None:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred)
        ax.scatter(r, p, color="red", zorder=3, label=f"threshold = {threshold:.2f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    return ax


def plot_roc_curve(y_true, y_proba, label=None, threshold=None, ax=None):
    fpr, recall, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    first_call = not ax.lines  
    ax.plot(fpr, recall, label=f"{label or 'model'} (ROC-AUC = {auc:.3f})")
    if first_call:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random (AUC = 0.5)")

    if threshold is not None:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        ax.scatter(fp / (fp + tn), tp / (tp + fn), color="red", zorder=3,
                   label=f"threshold = {threshold:.2f}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    return ax


def plot_confusion_matrix(y_true, y_proba, threshold, ax=None, save_path=None):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)  

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1%})",
                    ha="center", va="center", color=color)

    classes = ["No default", "Default"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")

    return ax



    










