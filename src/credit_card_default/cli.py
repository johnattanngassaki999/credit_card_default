"""Command line interface of the project.

It reads config.yaml and calls the functions of data.py, features.py, models.py and
evaluate.py. It contains no modelling logic of its own.

Usage, from the project root with the virtual environment active:
    python -m credit_card_default.cli train
    python -m credit_card_default.cli evaluate
    python -m credit_card_default.cli predict clients.csv --output predictions.csv
or, once the package is installed (pip install -e ".[dev]"):
    credit-card-default train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, load_config
from .data import load_credit, split_data 
from .evaluate import compute_metrics, find_best_threshold, get_oof_scores, get_scores
from .features import make_prep
from .models import (build_pipeline, get_model, get_param_grid, load_model,
                     save_model, tune_model)

FINAL_MODEL = "logistic_regression"  # chosen from the cross-validation comparison


def _model_path(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["paths"]["models_dir"] / "final_model.joblib"


def _split(cfg: dict):
    """Return X_train, X_test, y_train, y_test (ADAPT to what data.py provides).

    The split must be reproducible (same seed from config.yaml), so that `train`
    and `evaluate` see the same train and test sets.
    """
    return split_data(load_credit(cfg), cfg)


def cmd_train(cfg: dict) -> None:
    """Tune the final model on the training set, choose its threshold, save both."""
    X_train, _, y_train, _ = _split(cfg) 
    random_state, n_splits = cfg["random_state"], cfg["cv"]["n_splits"]

    pipeline = build_pipeline(make_prep(), get_model(random_state)[FINAL_MODEL])
    grid = get_param_grid(FINAL_MODEL, cfg["param_grids"])
    model, best_params = tune_model(pipeline, grid, X_train, y_train, n_splits, random_state)

    oof = get_oof_scores(model, X_train, y_train, n_splits, random_state)
    threshold = find_best_threshold(
        y_train, oof,
        metric=cfg["evaluation"]["threshold_metric"],
        min_recall=cfg["evaluation"]["min_recall"],
    )

    save_model(model, threshold, _model_path(cfg))
    print(f"Best parameters: {best_params}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Model saved to: {_model_path(cfg)}")


def cmd_evaluate(cfg: dict) -> None:
    """Evaluate the saved model on the test set. Run this once, on the final model."""
    _, X_test, _, y_test = _split(cfg)
    model, threshold = load_model(_model_path(cfg))
    metrics = compute_metrics(y_test, get_scores(model, X_test), threshold)
    print(f"Test set evaluation (threshold = {threshold:.3f})")
    print(pd.Series(metrics).round(3).to_string())


def cmd_predict(cfg: dict, input_csv: str, output_csv: str) -> None:
    """Score the clients of a CSV file (raw feature columns) with the saved model."""
    model, threshold = load_model(_model_path(cfg))
    clients = pd.read_csv(input_csv)
    scores = get_scores(model, clients)
    result = clients.assign(score=scores, flagged=(scores >= threshold).astype(int))
    result.to_csv(output_csv, index=False)
    print(f"{int(result['flagged'].sum())} of {len(result)} clients flagged -> {output_csv}")


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