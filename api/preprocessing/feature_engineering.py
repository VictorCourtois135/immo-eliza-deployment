"""
Feature engineering utilities.

All transformations here are row-wise formulas (no aggregation), so
they carry no data leakage risk and can be applied identically to
train and test sets.
"""
import numpy as np
import pandas as pd


def engeneering_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from existing columns:
    - Interaction terms (area x bedrooms, area x epc, etc.)
    - Ratios (area per bedroom, bed/bath ratio, etc.)
    - Quality composite scores (state + epc combinations)
    - Non-linear transforms (log, sqrt) useful for linear/log-target models
    - Building age categories
    """
    # --- interaction terms ---
    df["area_x_bedrooms"] = df["living_area_m2"] * df["bedrooms"]
    df["area_x_epc"] = df["living_area_m2"] * df["epc_score"]
    df["area_x_state"] = df["living_area_m2"] * df["state_of_the_building"]
    df["area_x_bathrooms"] = df["living_area_m2"] * df["bathrooms"]
    df["area_x_facade"] = df["living_area_m2"] * df["facades"]
    df["living_area_m2_sq"] = df["living_area_m2"] ** 2
    df["building_age"] = 2026 - df["building_year"]
    df["building_age_sq"] = df["building_age"] ** 2

    # --- ratios ---
    df["area_per_bedroom"] = df["living_area_m2"] / df["bedrooms"].replace(0, np.nan)
    df["area_per_bathroom"] = df["living_area_m2"] / df["bathrooms"].replace(0, np.nan)
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    df["garden_to_living_ratio"] = df["garden_area_m2"] / df["living_area_m2"].replace(0, np.nan)
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]

    # --- state / epc combinations ---
    df["quality_score"] = df["state_of_the_building"] + df["epc_score"]
    df["state_x_epc"] = df["state_of_the_building"] * df["epc_score"]
    df["quality_x_area"] = df["quality_score"] * df["living_area_m2"]
    df["state_x_epc_x_area"] = df["state_x_epc"] * df["living_area_m2"]

    # --- non-linear transforms, useful for linear / log-target models ---
    df["log_living_area"] = np.log1p(df["living_area_m2"])
    df["sqrt_living_area"] = np.sqrt(df["living_area_m2"])
    df["log_quality_x_area"] = np.log1p(df["quality_x_area"])
    df["log_state_x_epc_x_area"] = np.log1p(df["state_x_epc_x_area"])

    # --- building age categories (in addition to the continuous version) ---
    df["is_new_building"] = (df["building_age"] <= 5).astype(int)
    df["is_old_building"] = (df["building_age"] >= 80).astype(int)

    return df


def filter_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str = "price",
    lower_q: float = 0.01,
    upper_q: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute quantile bounds on train_df ONLY, then filter both train_df
    and test_df using the same bounds (avoids leakage from test
    distribution into the filtering decision).
    """
    lower_bound = train_df[column].quantile(lower_q)
    upper_bound = train_df[column].quantile(upper_q)

    train_df = train_df[(train_df[column] >= lower_bound) & (train_df[column] <= upper_bound)]
    test_df = test_df[(test_df[column] >= lower_bound) & (test_df[column] <= upper_bound)]

    return train_df, test_df