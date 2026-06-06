"""
Data loading for the Churn Module.

Loads monthly prime files, transaction files, creates the churn label
"""

import glob
import os

import config
import pandas as pd


# Churn labeling
def create_churn_labels(data_dir=None) -> pd.DataFrame:
    """Label customers as churned based on presence across the start and end months.

    If a CUSTOMER_ID exists in the reference month but NOT in the
    target month (or has a WROF status) → churn = 1, else churn = 0.

    Returns
    -------
    pd.DataFrame with columns: [CUSTOMER_ID, Card account status, churn]
    """
    ref_key = config.CHURN_REFERENCE_MONTH
    tgt_key = config.CHURN_TARGET_MONTH

    data_dir = data_dir or config.RAW_PRIME_DATA_DIR

    ref_path = os.path.join(data_dir, config.CHURN_LABEL_FILES[ref_key])
    tgt_path = os.path.join(data_dir, config.CHURN_LABEL_FILES[tgt_key])

    cid = config.CUSTOMER_ID
    status = config.STATUS_COL

    print("=" * 55)
    print(f"  Churn Labeling -- {ref_key} -> {tgt_key}")
    print("=" * 55)

    ref_df = _load_label_file(ref_path, ref_key, [cid, status])
    tgt_df = _load_label_file(tgt_path, tgt_key, [cid, status])

    # Set of customer IDs present in target month
    tgt_ids = set(tgt_df[cid].unique())
    print(f"\n{tgt_key} unique customers (active set): {len(tgt_ids):,}")

    # Deduplicate reference month - keep last occurrence per customer
    ref_deduped = ref_df.drop_duplicates(subset=[cid], keep="last").copy()

    # Generate churn label
    ref_deduped[config.TARGET_COL] = ref_deduped.apply(
        lambda row: (
            1
            if (row[cid] not in tgt_ids or row[status] in config.CHURN_DEFAULT_STATUSES)
            else 0
        ),
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

    # Map requested cols to file cols (handle RIM_NO → CUSTOMER_ID)
    raw_id = config.PRIME_CUSTOMER_ID_RAW
    canonical_id = config.CUSTOMER_ID
    rename_map = {}
    resolved = []
    for c in usecols:
        if c in header_cols:
            resolved.append(c)
        elif c == canonical_id and raw_id in header_cols:
            resolved.append(raw_id)
            rename_map[raw_id] = canonical_id

    if not resolved:
        raise ValueError(f"[{label}] None of {usecols} found in {filepath}")

    df = pd.read_csv(filepath, usecols=resolved, dtype=str, encoding="latin")
    if rename_map:
        df = df.rename(columns=rename_map)

    cols_present = [c for c in usecols if c in df.columns]
    for col in cols_present:
        df[col] = df[col].str.strip()
    df.dropna(subset=[cols_present[0]], inplace=True)

    print(
        f"  [{label}] Loaded {len(df):,} rows | unique IDs: {df[cols_present[0]].nunique():,}"
    )
    return df


# Transaction data
def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all transaction CSV files."""
    data_dir = data_dir or config.CLEANED_TRANSACTION_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin")
        df[config.TXN_DATE_COL] = pd.to_datetime(
            df[config.TXN_DATE_COL], errors="coerce"
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Replace fake sequential CUSTOMER_ID with real RIMNO
    raw_id = config.TXN_CUSTOMER_ID_RAW
    canonical_id = config.CUSTOMER_ID
    if raw_id in combined.columns:
        if canonical_id in combined.columns:
            combined = combined.drop(columns=[canonical_id])
        combined = combined.rename(columns={raw_id: canonical_id})

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


# Prime data
def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all prime (customer snapshot) CSV files."""
    data_dir = data_dir or config.CLEANED_PRIME_DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        print(f"  Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, encoding="latin")
        # Rename raw customer ID column to canonical name
        raw_id = config.PRIME_CUSTOMER_ID_RAW
        if raw_id in df.columns and config.CUSTOMER_ID not in df.columns:
            df = df.rename(columns={raw_id: config.CUSTOMER_ID})
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined[config.CUSTOMER_ID] = combined[config.CUSTOMER_ID].astype(str).str.strip()

    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {combined.shape}")
    return combined
