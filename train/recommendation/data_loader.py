"""
Data loading utilities for the Recommendation module.

Assumes that prime and transaction data have already been cleaned by the
centralized cleaning pipeline and are available as CSVs under the paths
configured in config.PRIME_DATA_DIR and config.TRANSACTION_DATA_DIR.

This mirrors the approach used in the credit and churn modules.
"""

import glob
import os

import config
import pandas as pd


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _parse_float_col(series: pd.Series) -> pd.Series:
    """Clean commas and cast to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def _parse_int_col(series: pd.Series) -> pd.Series:
    """Clean commas and cast to nullable Int64."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).astype("Int64")


# --------------------------------------------------------------------------
# Prime data
# --------------------------------------------------------------------------
def load_prime_data(data_dir: str = None, logs: list = None) -> pd.DataFrame:
    """Load and concatenate all *active* prime CSV files.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing cleaned prime CSVs.  Defaults to
        ``config.PRIME_DATA_DIR``.
    logs : list, optional
        Mutable list for log messages.

    Returns
    -------
    pd.DataFrame
    """
    if logs is None:
        logs = []

    data_dir = data_dir or config.PRIME_DATA_DIR

    logs.append("=" * 60)
    logs.append(" PHASE 1: Loading Cleaned Prime Data")
    logs.append("=" * 60)

    files = sorted(glob.glob(os.path.join(data_dir, "*active.csv")))
    if not files:
        # Fallback: try all CSVs in the directory
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'.")

    logs.append(f"Found {len(files)} file(s) in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin", dtype=str)
        logs.append(f"  -> {os.path.basename(f)}  ({len(df)} rows)")
        frames.append(df)

    prime_df = pd.concat(frames, ignore_index=True)

    # ---------- Cast columns to proper types ----------
    for col in config.PRIME_STRING_COLS:
        if col in prime_df.columns:
            prime_df[col] = prime_df[col].astype("string")

    for col in config.PRIME_INT_COLS:
        if col in prime_df.columns:
            prime_df[col] = _parse_int_col(prime_df[col])

    for col in config.PRIME_FLOAT_COLS:
        if col in prime_df.columns:
            prime_df[col] = _parse_float_col(prime_df[col])

    for col in config.PRIME_DATE_COLS:
        if col in prime_df.columns:
            prime_df[col] = pd.to_datetime(prime_df[col], errors="coerce")

    logs.append(f"Total prime rows loaded: {len(prime_df)}")
    return prime_df


# --------------------------------------------------------------------------
# Transaction data
# --------------------------------------------------------------------------
def load_transaction_data(data_dir: str = None, logs: list = None) -> pd.DataFrame:
    """Load and concatenate all cleaned transaction CSV files.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing cleaned transaction CSVs.  Defaults to
        ``config.TRANSACTION_DATA_DIR``.
    logs : list, optional
        Mutable list for log messages.

    Returns
    -------
    pd.DataFrame or None
        ``None`` if no transaction files are found (non-fatal).
    """
    if logs is None:
        logs = []

    data_dir = data_dir or config.TRANSACTION_DATA_DIR

    logs.append("")
    logs.append("=" * 60)
    logs.append(" PHASE 2: Loading Cleaned Transaction Data")
    logs.append("=" * 60)

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # Skip any leftover "_missing_id" files
    files = [f for f in all_files if not f.endswith("_missing_id.csv")]

    if not files:
        logs.append(f"WARNING: No transaction CSVs found in '{data_dir}'.")
        return None

    logs.append(f"Found {len(files)} file(s) in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin", dtype=str)
        logs.append(f"  -> {os.path.basename(f)}  ({len(df)} rows)")
        frames.append(df)

    txn_df = pd.concat(frames, ignore_index=True)

    # ---------- Cast columns to proper types ----------
    for col in config.TXN_STRING_COLS:
        if col in txn_df.columns:
            txn_df[col] = txn_df[col].astype("string")

    for col in config.TXN_INT_COLS:
        if col in txn_df.columns:
            txn_df[col] = _parse_int_col(txn_df[col])

    for col in config.TXN_FLOAT_COLS:
        if col in txn_df.columns:
            txn_df[col] = _parse_float_col(txn_df[col])

    for col in config.TXN_DATE_COLS:
        if col in txn_df.columns:
            txn_df[col] = pd.to_datetime(txn_df[col], errors="coerce")

    logs.append(f"Total transaction rows loaded: {len(txn_df)}")
    return txn_df
