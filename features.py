"""Shared, deterministic feature preparation for training and inference."""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = RAW_FEATURES[1:] + [
    "Temperature gap [K]",
    "Estimated shaft power [kW]",
    "Wear x torque",
]
CATEGORICAL_FEATURES = ["Type"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features available before a failure is known.

    IDs and failure-type flags are deliberately ignored. No fitted statistics are
    used here; imputation and encoding are learned inside the training pipeline.
    """
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing operational columns: {missing}")
    result = frame.loc[:, RAW_FEATURES].copy()
    result["Type"] = result["Type"].map(
        lambda value: str(value).strip().upper()
        if pd.notna(value) and str(value).strip() else np.nan
    )
    for column in RAW_FEATURES[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["Temperature gap [K]"] = (
        result["Process temperature [K]"] - result["Air temperature [K]"]
    )
    result["Estimated shaft power [kW]"] = (
        result["Torque [Nm]"] * result["Rotational speed [rpm]"] * 2 * np.pi / 60000
    )
    result["Wear x torque"] = result["Tool wear [min]"] * result["Torque [Nm]"]
    return result.loc[:, MODEL_FEATURES]
