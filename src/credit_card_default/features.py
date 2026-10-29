from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data import CREDIT_COLS, TARGET

ALL_FEATURES = [c for c in CREDIT_COLS if c != TARGET]
BASE_CAT = ["SEX", "EDUCATION", "MARRIAGE"]
PAY = [f"PAY_{i}" for i in range(1, 7)]

def make_prep(pay_as_category=True):
    cat = BASE_CAT + (PAY if pay_as_category else [])
    num = [c for c in ALL_FEATURES if c not in cat]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat)])