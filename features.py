from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data import CREDIT_COLS, TARGET

ALL_FEATURES = [c for c in CREDIT_COLS if c != TARGET]
BASE_CAT = ["SEX", "EDUCATION", "MARRIAGE"]
PAY = [f"PAY_{i}" for i in range(1, 7)]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
ENGINEERED = ["n_late", "max_delay", "util_1", "util_avg", "paid_share_1"]

def add_features(df):
    df = df.copy()
    df["n_late"] = (df[PAY] > 0).sum(axis=1)                       # months with a delay
    df["max_delay"] = df[PAY].max(axis=1)                          # worst status in 6 months
    df["util_1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]               # share of the limit in use, last month
    df["util_avg"] = df[BILLS].mean(axis=1) / df["LIMIT_BAL"]      # same, averaged over 6 months
    owed = df["BILL_AMT2"] > 0                                     # bill <= 0: nothing owed, ratio undefined
    share = df["PAY_AMT1"] / df["BILL_AMT2"].where(owed, 1)
    df["paid_share_1"] = share.where(owed, 1.0).clip(upper=1.0)    # share of last bill repaid, capped at 1
    return df

def make_prep(pay_as_category=True, engineered=False):
    cat = BASE_CAT + (PAY if pay_as_category else [])
    num = [c for c in ALL_FEATURES if c not in cat] + (ENGINEERED if engineered else [])
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat)])