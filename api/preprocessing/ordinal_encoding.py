"""
Ordinal encoding for categorical features that have a natural order
(EPC energy score, building condition).
"""
import numpy as np
import pandas as pd

EPC_MAPPING = {
    "Unknown": np.nan,
    "G": 1, "F": 2, "E": 3, "E+": 4, "D": 5,
    "C": 6, "B": 7, "B+": 8, "A": 9, "A+": 10, "A++": 11
}

STATE_MAPPING = {
    "Unknown": np.nan,
    "To demolish": 1, "To restore": 2, "To renovate": 3,
    "Under construction": 4, "Normal": 5, "Fully renovated": 6,
    "Excellent": 7, "New": 8,
}


def fit_ordinal_medians(train_df: pd.DataFrame) -> dict:
    """
    Compute median values for ordinal-encoded columns using ONLY the
    training set, to be reused for imputation on train/test.
    """
    epc_num = train_df["epc_score"].map(EPC_MAPPING)
    state_num = train_df["state_of_the_building"].map(STATE_MAPPING)
    return {"epc_median": epc_num.median(), "state_median": state_num.median()}


def apply_ordinal_encoding(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """
    Map EPC score and building state strings to ordinal numeric values,
    filling unmapped/missing entries with train-set medians.
    """
    df["epc_score"] = df["epc_score"].map(EPC_MAPPING)
    df["epc_score"] = df["epc_score"].fillna(medians["epc_median"])

    df["state_of_the_building"] = df["state_of_the_building"].map(STATE_MAPPING)
    df["state_of_the_building"] = df["state_of_the_building"].fillna(medians["state_median"])

    return df