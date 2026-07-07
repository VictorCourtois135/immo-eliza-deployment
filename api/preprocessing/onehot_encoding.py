"""
One-hot encoding utilities.

Categories are "fit" on the training set only; the test set is then
re-aligned on exactly the same dummy columns, with unknown categories
mapped to all-zero rows (no crash, no leakage).
"""
import pandas as pd

ONEHOT_COLS = ["property_type", "property_subtype", "region", "province"]


def fit_onehot_categories(train_df: pd.DataFrame) -> dict:
    """
    Extract the sorted list of unique categories per column, based
    ONLY on the training set.
    """
    return {col: sorted(train_df[col].dropna().unique().tolist()) for col in ONEHOT_COLS}


def apply_onehot(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """
    One-hot encode categorical columns using a fixed category set
    (from train). Categories not seen in train become NaN, and end up
    with all dummy columns set to 0 (i.e. treated as the baseline).
    """
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    df = pd.get_dummies(df, columns=list(categories.keys()), drop_first=True)
    return df