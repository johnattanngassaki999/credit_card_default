"""Tests for data.py. Put this file next to data.py and run:  pytest -q
No internet: we build 800 fake clients with the real column names (odd codes and negative bills included)."""
import types
 
import numpy as np
import pandas as pd
import pytest

from credit_card_default import data 
from credit_card_default.data import CREDIT_COLS, TARGET, clean, split_data, load_credit
 
 
def fake_credit(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"LIMIT_BAL": rng.integers(1, 60, n) * 10_000, "SEX": rng.integers(1, 3, n),
                       "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n), "MARRIAGE": rng.choice([0, 1, 2, 3], n),
                       "AGE": rng.integers(21, 70, n)})
    for i in range(1, 7):
        df[f"PAY_{i}"] = rng.choice([-2, -1, 0, 1, 2, 3], n)
        df[f"BILL_AMT{i}"] = rng.integers(-2_000, 150_000, n)
        df[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n)
    df[TARGET] = (rng.random(n) < 0.2).astype(int)
    df.index = pd.RangeIndex(1, n + 1, name="ID")
    return df[CREDIT_COLS]
 
 
@pytest.fixture
def credit():
    return fake_credit()
 
 
def test_clean_merges_odd_codes_and_changes_nothing_else(credit):
    out = clean(credit)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4} and set(out["MARRIAGE"]) <= {1, 2, 3}   # odd codes gone
    assert out.shape == credit.shape                                                    # no row dropped
    cols = ["EDUCATION", "MARRIAGE"]
    pd.testing.assert_frame_equal(out.drop(columns=cols), credit.drop(columns=cols))      # PAY_* etc. untouched
 
 
def test_split_is_80_20_without_overlap_and_without_target(credit):
    X_train, X_test, y_train, y_test = split_data(clean(credit))
    assert len(X_test) == pytest.approx(0.2 * len(credit), abs=1)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert TARGET not in X_train.columns
 
 
def test_split_is_stratified(credit):
    _, _, y_train, y_test = split_data(clean(credit))
    assert abs(y_train.mean() - y_test.mean()) < 0.01        # same share of defaulters in both
 
 
def test_split_is_reproducible(credit):
    df = clean(credit)
    assert split_data(df)[0].index.equals(split_data(df)[0].index)
    assert not split_data(df)[0].index.equals(split_data(df, random_state=7)[0].index)
 
 
def test_load_credit_renames_columns_and_keeps_id_as_index(monkeypatch, credit):
    raw = credit.reset_index()
    raw.columns = ["ID"] + [f"X{i}" for i in range(1, 24)] + ["Y"]                  # how UCI names them
    fake = types.SimpleNamespace(data=types.SimpleNamespace(original=raw))
    monkeypatch.setattr(data, "fetch_ucirepo", lambda id: fake)                       # no download
    pd.testing.assert_frame_equal(load_credit(), credit)