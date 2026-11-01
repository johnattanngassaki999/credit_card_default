import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_card_default.models import (build_pipeline, compare_models, get_model,
                                        get_param_grid, load_model, save_model)


def test_get_models_returns_the_baseline_and_three_models():
    assert set(get_model(42)) == {"dummy", "logistic_regression", "knn", "linear_svm"}


def test_build_pipeline_returns_a_pipeline_with_named_steps(data):
    X, y = data
    pipeline = build_pipeline(StandardScaler(), get_model(42)["logistic_regression"])
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["prep", "model"]
    pipeline.fit(X, y)
    assert pipeline.predict_proba(X).shape == (len(X), 2)


def test_build_pipeline_gives_each_pipeline_its_own_preprocessor():
    prep = StandardScaler()
    models = get_model(42)
    first = build_pipeline(prep, models["dummy"])
    second = build_pipeline(prep, models["knn"])
    assert first.named_steps["prep"] is not second.named_steps["prep"]


def test_get_param_grid_adds_the_pipeline_step_prefix():
    grids = {"knn": {"n_neighbors": [25, 50]}}
    assert get_param_grid("knn", grids) == {"model__n_neighbors": [25, 50]}


def test_get_param_grid_is_empty_for_a_model_without_grid():
    assert get_param_grid("dummy", {"knn": {"n_neighbors": [25]}}) == {}


def test_compare_models_uses_the_dummy_as_reference(data):
    X, y = data
    models = get_model(42)
    pipelines = {name: build_pipeline(StandardScaler(), models[name])
                 for name in ("dummy", "logistic_regression")}
    table = compare_models(pipelines, X, y, 3, 42)
    assert table.loc["dummy", "pr_auc_gain_vs_dummy"] == 0
    assert table.loc["logistic_regression", "pr_auc_gain_vs_dummy"] > 0


def test_saved_model_reloads_with_the_same_predictions_and_threshold(tmp_path, data):
    X, y = data
    pipeline = build_pipeline(StandardScaler(), get_model(42)["logistic_regression"]).fit(X, y)
    path = tmp_path / "model.joblib"
    save_model(pipeline, 0.42, path)
    loaded, threshold = load_model(path)
    assert threshold == 0.42
    np.testing.assert_allclose(loaded.predict_proba(X), pipeline.predict_proba(X))