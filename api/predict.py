"""
Prediction logic: loads the trained model and preprocessing artifacts,
and exposes a predict() function usable by the FastAPI app.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from preprocessing.cleaning import apply_cleaning
from preprocessing.ordinal_encoding import apply_ordinal_encoding
from preprocessing.feature_engineering import engeneering_feature
from preprocessing.onehot_encoding import apply_onehot

ARTIFACTS_DIR = Path(__file__).parent

# Load artifacts once, at import time (not on every request)
model = joblib.load(ARTIFACTS_DIR / "XGBoost_production.pkl")
stats = joblib.load(ARTIFACTS_DIR / "artifacts" / "cleaning_stats.pkl")
ordinal_medians = joblib.load(ARTIFACTS_DIR / "artifacts" / "ordinal_medians.pkl")
onehot_categories = joblib.load(ARTIFACTS_DIR / "artifacts" / "onehot_categories.pkl")
feature_columns = joblib.load(ARTIFACTS_DIR / "artifacts" / "feature_columns.pkl")


def preprocess(data: dict) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline (fitted on train) to a single
    new property, provided as a dict of raw feature values.
    """
    df = pd.DataFrame([data])
    df = apply_cleaning(df, stats)
    df = apply_ordinal_encoding(df, ordinal_medians)
    df = engeneering_feature(df)
    df = apply_onehot(df, onehot_categories)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict(data: dict) -> float:
    """
    Predict the price (in €) for a single property given as a dict of
    raw feature values (see preprocess() for the expected keys).
    """
    processed = preprocess(data)
    X = processed.values
    y_pred_log = model.predict(X)
    y_pred_real = np.expm1(y_pred_log)
    return float(y_pred_real[0])

