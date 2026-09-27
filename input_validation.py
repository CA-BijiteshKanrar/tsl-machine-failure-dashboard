"""Check inference inputs against values observed in the training data."""

import numpy as np
import pandas as pd

from features import RAW_FEATURES


# Observed minima and maxima in data/train.csv (the data used to fit this model).
TRAINING_RANGES = {
    "Air temperature [K]": (295.3, 304.4),
    "Process temperature [K]": (305.8, 313.8),
    "Rotational speed [rpm]": (1181, 2886),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0, 253),
}
TRAINING_TYPES = {"L", "M", "H"}


def out_of_range_reasons(frame: pd.DataFrame) -> pd.Series:
    """Return one explanation per unsupported row; blank means eligible to score.

    Missing feature values are left to the trained pipeline's imputer. A present
    numeric value beyond the training range or an unknown product type is excluded.
    """
    missing_columns = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    reasons = [""] * len(frame)

    def add_reason(mask: pd.Series, message: str) -> None:
        for position in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            reasons[position] = f"{reasons[position]}; {message}" if reasons[position] else message

    product_type = frame["Type"].astype("string").str.strip().str.upper()
    add_reason(
        product_type.notna() & product_type.ne("") & ~product_type.isin(TRAINING_TYPES),
        "Type is not L, M, or H",
    )
    for column, (low, high) in TRAINING_RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        add_reason(
            values.notna() & ~values.between(low, high),
            f"{column} outside {low:g} to {high:g}",
        )
    return pd.Series(reasons, index=frame.index, name="validation_reason", dtype="string")
