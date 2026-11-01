import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from credit_card_default.evaluate import (compute_metrics, cross_validate_model,
                                          find_best_threshold, get_oof_scores)

N_SPLITS, RANDOM_STATE = 5, 42


@pytest.fixture
def scored():
    """Labels (about 22% positives) and informative scores for threshold tests."""
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.22).astype(int)
    scores = 0.3 * y + 0.7 * rng.random(1000)
    return y, scores


def test_compute_metrics_perfect_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y, scores, threshold=0.5)
    for name in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert metrics[name] == pytest.approx(1.0)


def test_compute_metrics_when_nobody_is_flagged():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics(y, scores, threshold=0.9)  # no score reaches 0.9
    assert metrics["recall"] == 0
    assert metrics["precision"] == 0  # zero_division=0, not a warning or an error


def test_threshold_moves_recall_but_not_auc(scored):
    y, scores = scored
    strict = compute_metrics(y, scores, threshold=0.8)
    loose = compute_metrics(y, scores, threshold=0.4)
    assert loose["recall"] > strict["recall"]
    assert loose["roc_auc"] == pytest.approx(strict["roc_auc"])
    assert loose["pr_auc"] == pytest.approx(strict["pr_auc"])


def test_dummy_baseline_scores_what_no_skill_should_score(data):
    X, y = data
    result = cross_validate_model(DummyClassifier(strategy="prior"), X, y, N_SPLITS, RANDOM_STATE)
    assert result["roc_auc"]["mean"] == pytest.approx(0.5)
    assert result["pr_auc"]["mean"] == pytest.approx(y.mean(), abs=0.02)


def test_oof_probabilities_cover_every_client(data):
    X, y = data
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof = get_oof_scores(model, X, y, N_SPLITS, RANDOM_STATE)
    assert oof.shape == (len(y),)
    assert oof.min() >= 0 and oof.max() <= 1


def test_oof_uses_decision_function_for_a_model_without_probabilities(data):
    X, y = data
    model = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, random_state=RANDOM_STATE))
    oof = get_oof_scores(model, X, y, N_SPLITS, RANDOM_STATE)
    assert oof.shape == (len(y),)  # 1D scores, one per client
    assert oof.min() < 0 < oof.max()  # margins on both sides of 0, not probabilities


def test_best_threshold_respects_the_minimum_recall(scored):
    y, scores = scored
    threshold = find_best_threshold(y, scores, metric="min_recall", min_recall=0.6)
    assert compute_metrics(y, scores, threshold)["recall"] >= 0.6


def test_best_f1_threshold_is_at_least_as_good_as_the_default(scored):
    y, scores = scored
    threshold = find_best_threshold(y, scores, metric="f1")
    assert compute_metrics(y, scores, threshold)["f1"] >= compute_metrics(y, scores, 0.5)["f1"]


def test_best_threshold_rejects_invalid_arguments(scored):
    y, scores = scored
    with pytest.raises(ValueError):
        find_best_threshold(y, scores, metric="unknown")
    with pytest.raises(ValueError):
        find_best_threshold(y, scores, metric="min_recall")  # min_recall missing