import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42
TARGET = "default"

CREDIT_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
    "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
    "PAY_AMT5", "PAY_AMT6", TARGET,
]




def load_credit():
    """Fetch the credit-card default dataset from the UCI repository."""
    dataset = fetch_ucirepo(id=350)
    df = dataset.data.original.copy()

    # Keep ID as the index when it is provided by UCI.
    if "ID" in df.columns:
        df = df.set_index("ID")

    if df.shape[1] != len(CREDIT_COLS):
        raise ValueError(
            f"Expected {len(CREDIT_COLS)} dataset columns, found {df.shape[1]}."
        )

    # Give the columns consistent names for the rest of the project.
    df.columns = CREDIT_COLS
    return df


def clean(df):
    """Clean the EDUCATION and MARRIAGE category values."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def split_data(df, random_state=RANDOM_STATE, test_size=0.2):
    """Return an 80/20 stratified train/test split by default."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test



def prepare_data(random_state=RANDOM_STATE):
    """Load, clean, and split the dataset; return the data and split sets."""
    df = clean(load_credit())
    X_train, X_test, y_train, y_test = split_data(
        df, random_state=random_state
    )
    return df, X_train, X_test, y_train, y_test


