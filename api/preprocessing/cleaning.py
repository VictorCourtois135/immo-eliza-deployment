"""
Data cleaning utilities.

Handles column dropping and missing-value imputation.
Statistics (medians, groupby aggregations) must be fit ONLY on the
training set and then applied identically to train/test to avoid
data leakage.
"""
import numpy as np
import pandas as pd


def col_drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant or leaky columns and rows with missing coordinates.

    Columns are dropped for reasons ranging from redundancy (e.g. address
    duplicating lat/long/province/region) to data leakage (price_per_m2)
    or excessive missing values (kitchen_equipped, floor_number, etc.).

    Safe to call multiple times: only drops columns that actually exist.
    """
    COLS_TO_DROP = {
        "address": "irrelevant because we have lat, long, province, region",
        "property_url": "no impact on price",
        "coord_swapped": "no impact on price",
        "kitchen_equipped": "too many missing values",
        "floor_number": "too many missing values",
        "floors_total": "too many missing values",
        "parking_count": "too many missing values",
        "total_area_m2": "too many missing values",
        "postal_code": "not really important",
        "nearby_city": "too many categories for pd.get_dummies",
        "city": "too many categories for pd.get_dummies",
        "price_per_m2": "causes data leakage",
    }

    cols_drop_rows = ["latitude", "longitude"]

    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    df = df.dropna(subset=cols_drop_rows)
    return df


def fit_cleaning_stats(train_df: pd.DataFrame) -> dict:
    """
    Compute imputation statistics (group medians, global medians) using
    ONLY the training set.

    Returns a dict of stats to be reused later on both train and test
    via `apply_cleaning`, preventing any information leak from test data.
    """
    stats = {"group_medians": {}, "group_global": {}}

    for col in ["living_area_m2", "bedrooms", "bathrooms", "facades", "building_year"]:
        stats["group_medians"][col] = train_df.groupby("property_subtype")[col].median()
        stats["group_global"][col] = train_df[col].median()

    stats["garden_by_subtype"] = train_df.groupby("property_subtype")["garden_area_m2"].median()
    stats["garden_global"] = train_df["garden_area_m2"].median()

    return stats


def apply_cleaning(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply cleaning statistics (computed on train) to any dataframe
    (train or test).

    For each imputed column, also creates a `<col>_was_missing` flag so
    the model can learn from missingness itself.
    """
    df = df.copy()

    for col in ["living_area_m2", "bedrooms", "bathrooms", "facades", "building_year"]:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        fill_values = df["property_subtype"].map(stats["group_medians"][col])
        df[col] = df[col].fillna(fill_values)
        df[col] = df[col].fillna(stats["group_global"][col])  # subtype not seen in train

    df["garden_area_m2_was_missing"] = df["garden_area_m2"].isna().astype(int)
    fill_garden = df["property_subtype"].map(stats["garden_by_subtype"])
    df["garden_area_m2"] = np.where(
        df["garden_area_m2"].notna(), df["garden_area_m2"],
        np.where(df["has_garden"] == 0, 0, fill_garden)
    )
    df["garden_area_m2"] = df["garden_area_m2"].fillna(stats["garden_global"])

    for col in ["state_of_the_building", "epc_score"]:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna("Unknown")

    return df