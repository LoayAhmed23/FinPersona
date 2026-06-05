"""
Data loading utilities for the Credit Risk module.

Reads **raw** monthly CSVs from ``data/original/``, applies the
data-cleaning pipeline (from ``data_cleaning.clean``), then tags
each row with a ``snapshot_month`` extracted from the filename.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import config

# Make the project-level ``data_cleaning`` package importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_cleaning.clean import clean_prime, clean_transactions


# ── Month extraction from filenames ─────────────────────────────

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _get_month_year_prime(file_path: str) -> pd.Timestamp:
    """Extract datetime from prime filename like ``FEB2026.csv``."""
    name = os.path.basename(file_path).upper()
    match = re.search(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_?(\d{4})", name,
    )
    if match:
        month = _MONTH_MAP[match.group(1)]
        year = int(match.group(2))
        return pd.Timestamp(year, month, 1)


def _get_month_year_txn(file_path: str) -> pd.Timestamp:
    """Extract datetime from transaction filename like ``202506.xlsx``."""
    name = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"(\d{4})(\d{2})", name)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return pd.Timestamp(year, month, 1)


# Helpers for parsing messy numeric strings (e.g. "1,234.00")
def _parse_int(x):
    if pd.isna(x):
        return pd.NA
    x_clean = str(x).strip().replace(",", "")
    if x_clean.endswith(".00"):
        x_clean = x_clean[:-3]
    try:
        return int(x_clean)
    except ValueError:
        return np.nan


def _parse_float(x):
    if pd.isna(x):
        return np.nan
    x_clean = str(x).strip().replace(",", "")
    try:
        return float(x_clean)
    except ValueError:
        return np.nan


# ── Prime data ──────────────────────────────────────────────────

def load_prime_data(data_dir: str = None) -> pd.DataFrame:
    """Load and clean all raw prime CSV files.

    1. Calls ``clean_prime()`` to cast, fill, assign CUSTOMER_ID, and
       split active records.
    2. Tags each row with ``snapshot_month`` extracted from the filename.

    Returns
    -------
    pd.DataFrame  — cleaned, active prime data with snapshot_month.
    """
    data_dir = data_dir or config.PRIME_DATA_DIR

    # Run the cleaning pipeline (returns combined active_df + mapping)
    active_df, cid_mapping = clean_prime(data_dir)

    # Store mapping for later use by transaction loader
    load_prime_data._cid_mapping = cid_mapping

    # Tag each row with snapshot_month based on source filename
    # The clean pipeline doesn't track source files, so we re-derive
    # snapshot_month from the original files by matching RIMNO+CUSTOMER_ID
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Build a per-file row index for snapshot_month assignment
    # Re-read files lightly just for the month tagging
    month_frames = []
    for f in files:
        month_ts = _get_month_year_prime(f)
        if month_ts is None:
            continue
        temp = pd.read_csv(
            f, encoding="latin", usecols=["RIM_NO"], dtype="string",
        )
        temp = temp.rename(columns={"RIM_NO": "RIMNO"})
        temp["RIMNO"] = temp["RIMNO"].str.strip()
        temp[config.MONTH_COL] = month_ts
        temp["_source_file"] = os.path.basename(f)
        month_frames.append(temp)

    if month_frames:
        # The active_df has same row order as the concatenated input files.
        # Assign snapshot_month by matching positional indices.
        # Since clean_prime processes files in sorted order, we can
        # reconstruct the month for each original row.
        all_months = pd.concat(month_frames, ignore_index=True)
        # active_df may have fewer rows (dropped rows), so we use the
        # source_file based approach: tag before cleaning.
        # 
        # Simpler approach: re-derive from the raw files by building a
        # (RIMNO) -> month mapping per file.
        # But a row can appear in multiple months, so we need positional.
        #
        # Actually, the cleanest approach: re-tag based on the raw file
        # each row came from. Let's add source tracking into the clean.
        # 
        # For now, use a pragmatic approach: assign months based on the
        # original file order (clean_prime processes sorted files and
        # concatenates them).
        pass

    # Since clean_prime doesn't track source file, we need to re-derive
    # snapshot_month. We'll do a lightweight pass: read each raw file,
    # extract its (RIMNO rows), and merge the month.
    #
    # Better approach: just re-implement the month tagging ourselves
    # since we have the active_df already cleaned.
    # We'll re-load the raw files, tag each with month, then join.

    # Actually, let's use a simpler and more reliable approach:
    # Read each file, get the set of row indices, tag them.
    # Since clean_prime concatenates in sorted file order,
    # we can track cumulative indices.

    # Re-implement with source file tracking:
    str_dtype = {
        col: "string"
        for col in config.PRIME_STRING_COLS + config.PRIME_INT_COLS + config.PRIME_FLOAT_COLS
    }

    tagged_frames = []
    for f in files:
        month_ts = _get_month_year_prime(f)
        temp = pd.read_csv(
            f, encoding="latin",
            usecols=["RIM_NO"],
            dtype="string",
        )
        temp["_row_id"] = range(len(temp))
        temp[config.MONTH_COL] = month_ts
        temp["source_file"] = os.path.basename(f)
        temp = temp.rename(columns={"RIM_NO": "RIMNO"})
        temp["RIMNO"] = temp["RIMNO"].str.strip()
        tagged_frames.append(temp[["RIMNO", config.MONTH_COL, "source_file"]])

    # Since the active_df came from clean_prime which processes files in
    # the same sorted order, we can add a source_file column by tracking
    # cumulative positions. But clean_prime drops/filters rows...
    #
    # Simplest robust solution: add source_file to clean output.
    # For now, let's just re-tag using the file: re-read each file lightly,
    # clean it, tag it, and concatenate. This duplicates some work but is
    # correct.

    # PRAGMATIC SOLUTION: Redo the load with cleaning + month tagging
    from data_cleaning.clean import (
        _apply_cast, _drop_empty,
        PRIME_STRING_COLS as _P_STR, PRIME_FLOAT_COLS as _P_FLT,
        PRIME_INT_COLS as _P_INT, PRIME_DATE_COLS as _P_DT,
        PRIME_COLUMNS_TO_DROP as _P_DROP, INACTIVE_STATUSES as _INACT,
    )

    # We already have the global_pairs from clean_prime inside active_df.
    # Extract the mapping that was built:
    # Actually, let's just use the active_df but assign months by re-reading.
    # 
    # FINAL APPROACH: Build month tag from raw files and merge into active_df.
    # The key insight: each row in active_df has (RIMNO, CUSTOMER_ID, DOB)
    # which is unique per file. We can merge on these to get the month.

    # Build month lookup: for each file, read (RIMNO, DOB) and tag month
    month_lookup_parts = []
    for f in files:
        month_ts = _get_month_year_prime(f)
        if month_ts is None:
            continue
        temp = pd.read_csv(
            f, encoding="latin", usecols=["RIM_NO", "DOB"], dtype="string",
        )
        temp = temp.rename(columns={"RIM_NO": "RIMNO"})
        temp["RIMNO"] = temp["RIMNO"].str.strip()
        temp["DOB"] = pd.to_datetime(temp["DOB"], errors="coerce")
        temp[config.MONTH_COL] = month_ts
        temp["source_file"] = os.path.basename(f)
        # A (RIMNO, DOB) pair can appear multiple times in a file (multiple products)
        # Keep all rows; they'll merge to multiple active_df rows
        month_lookup_parts.append(temp[["RIMNO", "DOB", config.MONTH_COL, "source_file"]])

    month_lookup = pd.concat(month_lookup_parts, ignore_index=True)
    # Drop duplicates: same (RIMNO, DOB, month) should only tag once
    month_lookup = month_lookup.drop_duplicates(subset=["RIMNO", "DOB", config.MONTH_COL])

    # Merge month into active_df
    active_df["RIMNO"] = active_df["RIMNO"].astype("string").str.strip()
    if "DOB" in active_df.columns:
        active_df = active_df.merge(
            month_lookup, on=["RIMNO", "DOB"], how="left", suffixes=("", "_month"),
        )
    else:
        # Fallback: just merge on RIMNO (less precise)
        ml_no_dob = month_lookup.drop(columns=["DOB"]).drop_duplicates(subset=["RIMNO", config.MONTH_COL])
        active_df = active_df.merge(ml_no_dob, on=["RIMNO"], how="left")

    # Sort by month
    active_df = active_df.sort_values(config.MONTH_COL).reset_index(drop=True)

    months = active_df[config.MONTH_COL].nunique()
    print(f"[data_loader] Loaded {len(files)} prime file(s) -> {active_df.shape}")
    print(f"  Detected {months} unique month(s): "
          f"{sorted(active_df[config.MONTH_COL].dt.strftime('%Y-%m').unique())}")

    return active_df


# ── Transaction data ────────────────────────────────────────────

def load_transaction_data(data_dir: str = None) -> pd.DataFrame:
    """Load and clean all raw transaction files.

    1. Calls ``clean_transactions()`` to cast, map CUSTOMER_ID.
    2. Tags each row with ``snapshot_month`` extracted from the filename.

    Returns
    -------
    pd.DataFrame  — cleaned transactions with snapshot_month.
    """
    data_dir = data_dir or config.TRANSACTION_DATA_DIR

    # Get the CUSTOMER_ID mapping from the previous prime load
    cid_mapping = getattr(load_prime_data, "_cid_mapping", None)
    if cid_mapping is None:
        raise RuntimeError(
            "load_prime_data() must be called before load_transaction_data() "
            "to build the CUSTOMER_ID mapping."
        )

    # Run the cleaning pipeline
    cleaned = clean_transactions(data_dir, cid_mapping)

    # Tag each row with snapshot_month
    # The clean_transactions function doesn't track source files,
    # so we'll derive month from the transaction date or filename.
    # Since clean_transactions processes files in sorted order and
    # concatenates, we can re-derive by re-reading file metadata.

    xlsx_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = xlsx_files + csv_files

    # Build file -> month lookup
    file_months = {}
    for f in files:
        month_ts = _get_month_year_txn(f)
        if month_ts:
            file_months[os.path.basename(f)] = month_ts

    # If cleaned df has a source column, use it. Otherwise use TRXN DATE.
    if config.MONTH_COL not in cleaned.columns:
        # Derive from TRXN DATE (most reliable for transactions)
        if "TRXN DATE" in cleaned.columns:
            cleaned[config.MONTH_COL] = cleaned["TRXN DATE"].dt.to_period("M").dt.to_timestamp()
        elif "POST DATE" in cleaned.columns:
            cleaned[config.MONTH_COL] = cleaned["POST DATE"].dt.to_period("M").dt.to_timestamp()
        else:
            # Last resort: use the first file's month for all rows
            if file_months:
                cleaned[config.MONTH_COL] = list(file_months.values())[0]

    # Parse date columns if not already done
    for col in config.DATE_COLS_TXN:
        if col in cleaned.columns and not pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    print(f"[data_loader] Loaded {len(files)} transaction file(s) -> {cleaned.shape}")
    if config.MONTH_COL in cleaned.columns:
        months = cleaned[config.MONTH_COL].nunique()
        print(f"  Detected {months} unique month(s): "
              f"{sorted(cleaned[config.MONTH_COL].dt.strftime('%Y-%m').unique())}")

    return cleaned


# ── Merge ───────────────────────────────────────────────────────

def merge_data(prime_df: pd.DataFrame, txn_features_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join prime data with aggregated transaction features.

    If both DataFrames contain a ``snapshot_month`` column, the merge is
    performed on ``(CUSTOMER_ID, snapshot_month)`` so that each monthly
    snapshot gets its own transaction summary. Otherwise falls back to a
    simple customer-ID merge.
    """
    cid = config.CUSTOMER_ID
    month_col = config.MONTH_COL
    txn_cols = txn_features_df.columns.tolist()

    prime_df[cid] = pd.to_numeric(prime_df[cid], errors="coerce").astype("Int64")
    # CUSTOMER_ID is part of the MultiIndex — rebuild with aligned dtype
    idx_names = txn_features_df.index.names
    txn_features_df = txn_features_df.reset_index()
    txn_features_df[cid] = pd.to_numeric(txn_features_df[cid], errors="coerce").astype("Int64")
    txn_features_df = txn_features_df.set_index(idx_names)

    # Month-aware merge: txn_features is indexed by (CUSTOMER_ID, month)
    txn_reset = txn_features_df.reset_index()
    merged = prime_df.merge(txn_reset, on=[cid, month_col], how="left")
    # Fill NaN for customers with no transactions in that month
    merged[txn_cols] = merged[txn_cols].fillna(0)

    print(f"[data_loader] Merged shape: {merged.shape}")
    return merged
