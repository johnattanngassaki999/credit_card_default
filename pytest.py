"""The whole chain: data -> features -> models -> evaluation, on fake data."""
import numpy as np
from sklearn.pipeline import Pipeline
 
from data import prepare_data
from evaluate import compute_metrics, find_best_threshold, get_oof_proba
from features import add_features, make_prep
from models import compare_models, get_model, load_model, save_model
 
 
def test_the_whole_pipeline_runs_end_to_end(fake_uci, tmp_path):
    df, X_train, X_test, y_train, y_test = prepare_data()
    X_train, X_test = add_features(X_train), add_features(X_test)
 
    pipes = {name: Pipeline([("prep", make_prep(True, True)), ("model", m)])
             for name, m in get_model(42).items()}
    table = compare_models(pipes, X_train, y_train, n_splits=3, random_state=42)
    assert set(table.index) == set(pipes)
 
    best = pipes["logistic_regression"]
    oof = get_oof_proba(best, X_train, y_train, n_splits=3, random_state=42)
    threshold = find_best_threshold(y_train, oof, metric="min_recall", min_recall=0.6)
 
    best.fit(X_train, y_train)
    save_model(best, threshold, tmp_path / "final.joblib")
    loaded, loaded_threshold = load_model(tmp_path / "final.joblib")
 
    proba = loaded.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, loaded_threshold)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert np.isclose(loaded_threshold, threshold)