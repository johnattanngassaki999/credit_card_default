"""Command line interface of the project.

It reads config.yaml and calls the functions of data.py, features.py, models.py and
evaluate.py. It contains no modelling logic of its own.

Usage, from the project root with the virtual environment active:
    python -m credit_card_default.cli train
    python -m credit_card_default.cli evaluate
    python -m credit_card_default.cli predict clients_sample.csv --output predictions.csv
or, once the package is installed (pip install -e ".[dev]"):
    credit-card-default train
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# The encoder warns when a validation fold contains a rare category that the training
# folds never saw. handle_unknown="ignore" handles this on purpose, so the message is
# silenced here, in this process and in the parallel worker processes started by joblib.
_UNKNOWN_CATEGORIES = "Found unknown categories"
warnings.filterwarnings("ignore", message=_UNKNOWN_CATEGORIES)
os.environ["PYTHONWARNINGS"] = f"ignore:{_UNKNOWN_CATEGORIES}"

import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .config import PROJECT_ROOT, load_config  # noqa: E402
from .data import load_credit, split_data  # noqa: E402  (ADAPT if data.py changes)
from .evaluate import (compute_metrics, find_best_threshold,  # noqa: E402
                       get_oof_scores, get_scores)
from .features import make_prep  # noqa: E402
from .models import (build_pipeline, get_model, get_param_grid,  # noqa: E402
                     load_model, save_model, tune_model)

FINAL_MODEL = "logistic_regression"  # chosen from the cross-validation comparison


def _model_path(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["paths"]["models_dir"] / "final_model.joblib"


def _split(cfg: dict):
    """Return X_train, X_test, y_train, y_test (ADAPT to what data.py provides).

    The split must be reproducible (same seed from config.yaml), so that `train`
    and `evaluate` see the same train and test sets.
    """
    return split_data(load_credit(), test_size=cfg["data"]["test_size"],
                      random_state=cfg["random_state"])


def _title(text: str) -> None:
    print(f"\n{text}\n{'=' * len(text)}")


def _show_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _metrics_table(metrics: dict[str, float]) -> str:
    return pd.Series(metrics).round(3).to_frame("value").to_string()


def cmd_train(cfg: dict) -> None:
    """Tune the final model on the training set, choose its threshold, save both."""
    X_train, _, y_train, _ = _split(cfg)  # the test set is not used here
    seed, n_splits = cfg["random_state"], cfg["cv"]["n_splits"]

    pipeline = build_pipeline(make_prep(), get_model(seed)[FINAL_MODEL])
    grid = get_param_grid(FINAL_MODEL, cfg["param_grids"])
    model, best_params = tune_model(pipeline, grid, X_train, y_train, n_splits, seed)

    rule = cfg["evaluation"]["threshold_metric"]
    min_recall = cfg["evaluation"]["min_recall"]
    oof = get_oof_scores(model, X_train, y_train, n_splits, seed)
    threshold = find_best_threshold(y_train, oof, metric=rule, min_recall=min_recall)
    metrics = compute_metrics(y_train, oof, threshold)

    save_model(model, threshold, _model_path(cfg))

    params = ", ".join(f"{k.removeprefix('model__')} = {v}" for k, v in best_params.items())
    rule_text = f"min recall {min_recall}" if rule == "min_recall" else rule
    _title(f"Training: {FINAL_MODEL}")
    print(f"Training clients : {len(y_train):,}")
    print(f"Best parameters  : {params}")
    print(f"Threshold        : {threshold:.3f}  (rule: {rule_text})")
    print("\nOut-of-fold metrics on the training set, at this threshold:")
    print(_metrics_table(metrics))
    print(f"\nModel saved to   : {_show_path(_model_path(cfg))}")


def cmd_evaluate(cfg: dict) -> None:
    """Evaluate the saved model on the test set. Run this once, on the final model."""
    _, X_test, _, y_test = _split(cfg)
    model, threshold = load_model(_model_path(cfg))
    scores = get_scores(model, X_test)
    metrics = compute_metrics(y_test, scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, (scores >= threshold).astype(int)).ravel()

    _title("Test set evaluation")
    print(f"Threshold : {threshold:.3f}")
    print(f"Clients   : {len(y_test):,} ({int(tp + fn):,} defaulters)")
    print("\nMetrics:")
    print(_metrics_table(metrics))
    print("\nConfusion matrix (number of clients):")
    print(pd.DataFrame([[tn, fp], [fn, tp]],
                       index=["actual: no default", "actual: default"],
                       columns=["predicted: no default", "predicted: default"]).to_string())
    print(f"\nFlagged as defaulters: {(tp + fp) / len(y_test):.1%} of clients")


def cmd_predict(cfg: dict, input_csv: str, output_csv: str) -> None:
    """Score the clients of a CSV file (raw feature columns) with the saved model."""
    model, threshold = load_model(_model_path(cfg))
    clients = pd.read_csv(input_csv)
    scores = get_scores(model, clients)
    result = clients.assign(score=scores, flagged=(scores >= threshold).astype(int))
    result.to_csv(output_csv, index=False)

    n_flagged = int(result["flagged"].sum())
    shown = [c for c in ("ID",) if c in result.columns] + ["score", "flagged"]
    _title("Predictions")
    print(f"Threshold : {threshold:.3f}")
    print(f"Flagged   : {n_flagged} of {len(result)} clients ({n_flagged / len(result):.1%})")
    print(f"Saved to  : {output_csv}")
    print("\nFirst clients:")
    print(result[shown].round(3).head(15).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-card-default",
        description="Credit card default prediction: train, evaluate, predict.",
    )
    parser.add_argument("--config", default=None,
                        help="path to a config.yaml (default: the one at the project root)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("train", help="tune the final model, choose the threshold, save them")
    commands.add_parser("evaluate", help="evaluate the saved model on the test set (run once)")

    predict = commands.add_parser("predict", help="score the clients of a CSV file")
    predict.add_argument("input_csv", help="CSV file with the client features")
    predict.add_argument("--output", default="predictions.csv",
                         help="where to write the predictions (default: predictions.csv)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "train":
        cmd_train(cfg)
    elif args.command == "evaluate":
        cmd_evaluate(cfg)
    elif args.command == "predict":
        cmd_predict(cfg, args.input_csv, args.output)


if __name__ == "__main__":
    main()