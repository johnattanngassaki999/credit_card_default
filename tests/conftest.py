import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def data():
    """Small synthetic dataset with about 22% positives, like the real default rate."""
    X, y = make_classification(n_samples=1500, n_features=8, n_informative=5,
                               weights=[0.78], flip_y=0.02, random_state=0)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]), pd.Series(y)