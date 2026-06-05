"""
Data loading for the Churn Module.

Reads **raw** prime and transaction files, applies the data-cleaning
pipeline (from ``data_cleaning.clean``), then creates churn labels.
"""

import glob
import os
import sys

import pandas as pd

import config

# Make the project-level ``data_cleaning`` package importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_cleaning.clean import clean_prime, clean_transactions


# ═══════════════════════════════════════════════════════════════
#  Churn labeling
# ═══════════════════════════════════════════════════════════════

def create_churn_labels(data_dir: str = None) -> pd.DataFrame:
    """Label customers as churned based on presence across start/end months.

    If a CUSTOMER_ID exists in the reference month but NOT in the
    target month (or has a WROF status) → churn = 1, else churn = 0.

    The function reads the **raw** prime CSV files specified in
    ``config.CHURN_LABEL_FILES`` and applies basic cleaning (rename
    columns, strip whitespace) before labeling.

    Returns
    -------
    pd.DataFrame with columns: [CUSTOMER_ID, Card account status, CHURN]
    """
    data_dir = data_dir or config.PRIME_DATA_DIR

    ref_key = config.CHURN_REFERENCE_MONTH
    tgt_key = config.CHURN_TARGET_MONTH

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
    """Load a single raw CSV for churn labeling.

    Handles the RIM_NO → CUSTOMER_ID rename for raw files.
    """
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

    print(f"  [{label}] Loaded {len(df):,} rows | unique IDs: {df[cols_present[0]].nunique():,}")
    return df


# ═══════════════════════════════════════════════════════════════
#  Transaction data
# ═══════════════════════════════════════════════════════════════

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and clean all raw transaction files.

    Calls ``clean_transactions()`` from ``data_cleaning.clean``
    to cast columns, map CUSTOMER_ID, and return cleaned rows.
    """
    data_dir = data_dir or config.TRANSACTION_DATA_DIR

    # Get the CUSTOMER_ID mapping from the previous prime load
    cid_mapping = getattr(load_prime_data, "_cid_mapping", None)
    if cid_mapping is None:
        raise RuntimeError(
            "load_prime_data() must be called before load_transaction_data() "
            "to build the CUSTOMER_ID mapping."
        )

    # Run cleaning pipeline
    combined = clean_transactions(data_dir, cid_mapping)

    # Parse transaction dates
    if config.TXN_DATE_COL in combined.columns:
        combined[config.TXN_DATE_COL] = pd.to_datetime(
            combined[config.TXN_DATE_COL], errors="coerce"
        )

    # Map reversal flag to numeric
    if config.TXN_REVERSAL_COL in combined.columns:
        combined[config.TXN_REVERSAL_COL] = combined[config.TXN_REVERSAL_COL].map(
            config.TXN_REVERSAL_MAP
        )

    # Clean customer ID
    combined = combined.dropna(subset=[config.CUSTOMER_ID])
    combined[config.CUSTOMER_ID] = combined[config.CUSTOMER_ID].astype(str).str.strip()

    print(f"[data_loader] Transaction data shape: {combined.shape}")
    return combined


# ═══════════════════════════════════════════════════════════════
#  Prime data
# ═══════════════════════════════════════════════════════════════

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and clean all raw prime (customer snapshot) CSV files.

    Calls ``clean_prime()`` from ``data_cleaning.clean`` to cast, fill,
    assign CUSTOMER_ID, and split active records.
    """
    data_dir = data_dir or config.PRIME_DATA_DIR

    # Run cleaning pipeline
    active_df, cid_mapping = clean_prime(data_dir)

    # Store mapping for later use by transaction loader
    load_prime_data._cid_mapping = cid_mapping

    # Ensure canonical customer ID column
    combined = active_df
    combined[config.CUSTOMER_ID] = combined[config.CUSTOMER_ID].astype(str).str.strip()

    print(f"[data_loader] Loaded prime data -> {combined.shape}")
    return combined
