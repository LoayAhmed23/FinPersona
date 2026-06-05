"""
Preprocessin: encoding, imputation, and feature preparation for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import config


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts: dict = None):
    """Prepare features for modeling.

    If fit is True, fit encoders and return them in artifacts.
    If fit is False, reuse encoders from artifacts.
    
    artifacts : dict
        Previously fitted {encoders, feature_order}.

    Returns
    -------
    X : pd.DataFrame 
    y : pd.Series — target vector
    artifacts : dict
    """
    df = df.copy()

    # Separate target
    y = df[config.TARGET_COL].copy()

    # Drop columns that shouldn't be features
    cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")

    if fit:
        # --- Label-encode categorical columns ---
        encoders = {}
        cat_cols = [c for c in config.CATEGORICAL_COLS if c in X.columns]

        for col in cat_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("UNKNOWN")
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # --- Ensure all remaining columns are numeric ---
        for col in X.columns:
            if col not in cat_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        artifacts = {
            "encoders": encoders,
            "cat_cols": cat_cols,
            "feature_order": X.columns.tolist(),
        }
    else:
        if artifacts is None:
            raise ValueError("artifacts must be provided when fit=False")

        cat_cols = artifacts["cat_cols"]
        for col in cat_cols:
            if col in X.columns:
                le = artifacts["encoders"][col]
                X[col] = X[col].astype(str).fillna("UNKNOWN")
                X[col] = X[col].map(
                    lambda v, _le=le: (
                        _le.transform([v])[0] if v in _le.classes_ else -1
                    )
                )

        for col in X.columns:
            if col not in cat_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Align to training feature order
        for c in artifacts["feature_order"]:
            if c not in X.columns:
                X[c] = 0
        X = X[artifacts["feature_order"]]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[preprocess] Output shape: {X.shape}  |  fit={fit}")
    return X, y, artifacts
