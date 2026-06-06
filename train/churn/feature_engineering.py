"""
Feature engineering for the Churn Prediction System.

Creates transaction-level aggregations and prime-level snapshot features and,
merges the data to create a single dataset.
"""

import config
import pandas as pd


# Transaction features
def engineer_transaction_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data into per-customer features.

    Features created:
        - total_spend, avg_spend, transaction_count
        - recency_days (days since last transaction)
        - reversal_ratio
    """
    df = txn_df.copy()
    cid = config.CUSTOMER_ID
    amt_col = config.TXN_AMOUNT_COL
    date_col = config.TXN_DATE_COL
    rev_col = config.TXN_REVERSAL_COL

    # Ensure amount column is numeric
    if amt_col in df.columns:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)

    # --- Basic aggregations ---
    agg = (
        df.groupby(cid)
        .agg(
            total_spend=(amt_col, "sum"),
            avg_spend=(amt_col, "mean"),
            transaction_count=(amt_col, "count"),
        )
        .reset_index()
    )

    # --- Recency (days since last transaction) ---
    if date_col in df.columns:
        reference_date = df[date_col].max()
        last_txn = df.groupby(cid)[date_col].max().reset_index()
        last_txn["recency_days"] = (reference_date - last_txn[date_col]).dt.days
        agg = agg.merge(last_txn[[cid, "recency_days"]], on=cid, how="left")
    else:
        agg["recency_days"] = 0

    # --- Reversal ratio ---
    if rev_col in df.columns:
        reversal = (
            df.groupby(cid)
            .agg(
                reversal_count=(rev_col, "sum"),
                total_txn=(rev_col, "count"),
            )
            .reset_index()
        )
        reversal["reversal_ratio"] = reversal["reversal_count"] / reversal["total_txn"]
        agg = agg.merge(reversal[[cid, "reversal_ratio"]], on=cid, how="left")
    else:
        agg["reversal_ratio"] = 0

    agg = agg.fillna(0)

    print(f"[feature_eng] Transaction features -> {agg.shape}")
    return agg


# Prime features
def engineer_prime_features(prime_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and engineer features from the prime (customer snapshot) data.

    Keeps the latest snapshot per customer and computes TENURE_DAYS.
    """
    df = prime_df.copy()
    cid = config.CUSTOMER_ID

    # Parse creation date
    if "CREATION_DATE" in df.columns:
        df["CREATION_DATE"] = pd.to_datetime(df["CREATION_DATE"], errors="coerce")

    # Keep latest record per customer
    df = df.drop_duplicates(subset=[cid], keep="last")

    # Select feature columns (only those present in the data)
    keep_cols = [c for c in config.PRIME_FEATURE_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # Compute tenure
    reference_date = pd.to_datetime(config.REFERENCE_DATE)
    if "CREATION_DATE" in df.columns:
        df["TENURE_DAYS"] = (reference_date - df["CREATION_DATE"]).dt.days

    df = df.fillna(0)

    print(f"[feature_eng] Prime features -> {df.shape}")
    return df


# Merge all data sources
def merge_all(
    txn_features: pd.DataFrame,
    prime_features: pd.DataFrame,
    churn_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transaction features, prime features, and churn labels."""
    cid = config.CUSTOMER_ID

    # Filter prime to customers with transactions
    prime_features = prime_features[prime_features[cid].isin(txn_features[cid])]

    # Merge txn + prime
    merged = txn_features.merge(prime_features, on=cid, how="left")

    # Merge churn labels
    merged = merged.merge(
        churn_labels[[cid, config.TARGET_COL]],
        on=cid,
        how="left",
    )

    # Customers not in churn labels default to non-churned
    merged[config.TARGET_COL] = merged[config.TARGET_COL].fillna(0).astype(int)

    print(f"[feature_eng] Final merged dataset -> {merged.shape}")
    print(merged[config.TARGET_COL].value_counts(dropna=False).to_string())
    return merged
