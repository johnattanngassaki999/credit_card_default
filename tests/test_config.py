from credit_card_default.config import load_config


def test_config_contains_the_values_the_code_reads():
    cfg = load_config()
    assert isinstance(cfg["random_state"], int)
    assert cfg["cv"]["n_splits"] >= 2
    assert "threshold_metric" in cfg["evaluation"]


def test_every_hyperparameter_grid_is_a_list_of_values():
    # GridSearchCV needs a list of candidates for each parameter, never a single value
    grids = load_config()["param_grids"]
    for model in ("logistic_regression", "knn", "linear_svm"):
        assert all(isinstance(values, list) for values in grids[model].values())