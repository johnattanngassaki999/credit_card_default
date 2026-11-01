"""Tests for features.py. Put this file next to features.py and data.py and run:  pytest -q
Same fake clients as in test_data.py (no internet)."""
import numpy as np
import pandas as pd
import pytest
 
from data import CREDIT_COLS, TARGET, clean
from features import ALL_FEATURES, ENGINEERED, add_features, make_prep
 
 
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
 
 
def client(**kw):
    """One client, all zeros except what the test sets (so the answer is easy to compute by hand)."""
    row = dict.fromkeys(ALL_FEATURES, 0)
    row.update(LIMIT_BAL=100_000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=30)
    row.update(kw)
    return pd.DataFrame([row])
 
 
@pytest.fixture
def credit():
    return fake_credit()
 
 
@pytest.fixture
def X(credit):
    return clean(credit).drop(columns=TARGET)
 
 
def test_add_features_adds_five_columns_without_nan_or_inf(X):
    before = X.copy()
    out = add_features(X)
    assert list(out.columns) == list(X.columns) + ENGINEERED
    assert np.isfinite(out[ENGINEERED].to_numpy()).all()
    pd.testing.assert_frame_equal(X, before)                  # the input table is not modified
 
 
def test_n_late_and_max_delay():
    out = add_features(client(PAY_1=2, PAY_2=0, PAY_3=-1, PAY_4=1, PAY_5=0, PAY_6=5))
    assert out["n_late"].iloc[0] == 3                         # only the statuses above 0 count
    assert out["max_delay"].iloc[0] == 5
 
 
def test_utilisation():
    assert add_features(client(BILL_AMT1=50_000))["util_1"].iloc[0] == pytest.approx(0.5)
    six = {f"BILL_AMT{i}": 10_000 for i in range(1, 7)}
    assert add_features(client(**six))["util_avg"].iloc[0] == pytest.approx(0.1)
 
 
def test_paid_share():
    share = lambda bill, paid: add_features(client(BILL_AMT2=bill, PAY_AMT1=paid))["paid_share_1"].iloc[0]
    assert share(1_000, 250) == pytest.approx(0.25)           # a quarter of the bill repaid
    assert share(1_000, 5_000) == 1.0                         # overpayment is capped at 1
    assert share(0, 300) == 1.0 and share(-500, 300) == 1.0   # nothing owed -> 1
 
 
def test_make_prep_has_no_nan_and_matches_its_names(X):
    prep = make_prep(True, False).fit(X)
    out = prep.transform(X)
    assert out.shape[1] == len(prep.get_feature_names_out()) and not np.isnan(out).any()
 
 
def test_pay_as_category_widens_the_table_and_engineered_adds_five_columns(X):
    X_fe = add_features(X)
    width = lambda **kw: make_prep(**kw).fit_transform(X_fe).shape[1]
    assert width(pay_as_category=True, engineered=False) > width(pay_as_category=False, engineered=False)
    assert width(pay_as_category=True, engineered=True) == width(pay_as_category=True, engineered=False) + 5
 
 
def test_scaler_learns_from_the_training_part_only(X):
    train = X.iloc[:600]
    prep = make_prep(True, False).fit(train)
    num = prep.transformers_[0][2]
    assert np.allclose(prep.named_transformers_["num"].mean_, train[num].mean())      # no leakage
 
 
def test_a_category_never_seen_in_training_does_not_crash(X):
    prep = make_prep(True, False).fit(X[X["PAY_6"] != 3])
    assert prep.transform(X[X["PAY_6"] == 3].head(5)).shape[0] == 5
 
 
@pytest.mark.xfail(reason="Known issue: an unseen PAY value is encoded like -2 ('no consumption'). "
                          "Fix: OneHotEncoder(..., handle_unknown='infrequent_if_exist', min_frequency=0.005)")
def test_an_unseen_pay_value_is_not_encoded_like_no_consumption(X):
    X = X.copy()
    X.iloc[:2, X.columns.get_loc("PAY_6")] = 8                # a rare value, so a 'rare' bucket exists
    prep = make_prep(True, False).fit(X)
    cols = [i for i, n in enumerate(prep.get_feature_names_out()) if "PAY_6" in n]
    a, b = X.iloc[[10]].copy(), X.iloc[[10]].copy()
    a["PAY_6"], b["PAY_6"] = -2, 7                            # -2 = no consumption, 7 = never seen
    assert not np.array_equal(prep.transform(a)[:, cols], prep.transform(b)[:, cols])
 