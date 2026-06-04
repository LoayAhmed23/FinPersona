"""
Data loading utilities for the Churn Prediction System.

Loads monthly prime CSVs, transaction CSVs, creates churn labels,
and provides a merge helper.
"""

import glob
import os

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Churn labeling
# ---------------------------------------------------------------------------

def create_churn_labels() -> pd.DataFrame:
    """Label customers as churned based on presence across two months.

    Logic: If a CUSTOMER_ID exists in the reference month but NOT in the
    target month (or has a WROF status) → churn = 1, else churn = 0.

    Returns
    -------
    pd.DataFrame with columns: [CUSTOMER_ID, Card account status, churn]
    """
    ref_key = config.CHURN_REFERENCE_MONTH
    tgt_key = config.CHURN_TARGET_MONTH

    ref_path = os.path.join(config.PRIME_DATA_DIR, config.CHURN_LABEL_FILES[ref_key])
    tgt_path = os.path.join(config.PRIME_DATA_DIR, config.CHURN_LABEL_FILES[tgt_key])

    cid = config.CUSTOMER_ID
    status = config.STATUS_COL

    print("=" * 55)
    print(f"  Churn Labeling — {ref_key} → {tgt_key}")
    print("=" * 55)

    ref_df = _load_label_file(ref_path, ref_key, [cid, status])
    tgt_df = _load_label_file(tgt_path, tgt_key, [cid, status])

    # Build set of customer IDs present in target month
    tgt_ids = set(tgt_df[cid].unique())
    print(f"\n{tgt_key} unique customers (active set): {len(tgt_ids):,}")

    # Deduplicate reference month — keep last occurrence per customer
    ref_deduped = ref_df.drop_duplicates(subset=[cid], keep="last").copy()

    # Label churn
    ref_deduped[config.TARGET_COL] = ref_deduped.apply(
        lambda row: 1 if (
            row[cid] not in tgt_ids or
            row[status] in config.CHURN_DEFAULT_STATUSES
        ) else 0,
        axis=1,
    )

    total = len(ref_deduped)
    churned = ref_deduped[config.TARGET_COL].sum()
    rate = churned / total * 100 if total > 0 else 0

    print(f"\n  Total customers : {total:>10,}")
    print(f"  Churned (1)     : {churned:>10,}  ({rate:.2f}%)")
    print(f"  Retained (0)    : {total - churned:>10,}  ({100 - rate:.2f}%)")

    return ref_deduped[[cid, status, config.TARGET_COL]].reset_index(drop=True)


def _load_label_file(filepath: str, label: str, usecols: list) -> pd.DataFrame:
    """Load a single CSV for churn labeling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[{label}] File not found: {filepath}")

    # Read only the columns that exist in the file
    header_cols = pd.read_csv(filepath, nrows=0, encoding="latin").columns.tolist()
    cols_to_read = [c for c in usecols if c in header_cols]
    if not cols_to_read:
        raise ValueError(f"[{label}] None of {usecols} found in {filepath}")

    df = pd.read_csv(filepath, usecols=cols_to_read, dtype=str, encoding="latin")
    for col in cols_to_read:
        df[col] = df[col].str.strip()
    df.dropna(subset=[cols_to_read[0]], inplace=True)

    print(f"  [{label}] Loaded {len(df):,} rows | unique IDs: {df[cols_to_read[0]].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Transaction data
# ---------------------------------------------------------------------------

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all transaction CSV files."""
    data_dir = data_dir or config.TRANSACTION_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, config.TXN_FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin")
        df[config.TXN_DATE_COL] = pd.to_datetime(df[config.TXN_DATE_COL], errors="coerce")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Map reversal flag to numeric
    if config.TXN_REVERSAL_COL in combined.columns:
        combined[config.TXN_REVERSAL_COL] = combined[config.TXN_REVERSAL_COL].map(
            config.TXN_REVERSAL_MAP
        )

    # Clean customer ID
    combined = combined.dropna(subset=[config.CUSTOMER_ID])
    combined[config.CUSTOMER_ID] = combined[config.CUSTOMER_ID].astype(str).str.strip()

    print(f"[data_loader] Loaded {len(files)} transaction file(s) -> {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Prime data
# ---------------------------------------------------------------------------

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all prime (customer snapshot) CSV files."""
    data_dir = data_dir or config.PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, config.PRIME_FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        print(f"  Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, encoding="latin")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined[config.CUSTOMER_ID] = combined[config.CUSTOMER_ID].astype(str).str.strip()

    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    return combined
