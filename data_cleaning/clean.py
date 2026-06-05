"""
Importable data-cleaning functions.

Refactored from the standalone scripts ``prime_id_creation.py`` and
``transaction_id_mapping.py`` so that both the Credit Risk and Churn
training pipelines can clean raw data in-memory without needing
pre-processed files on disk.
"""

import glob
import os

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  Column definitions (same as the standalone scripts)
# ═══════════════════════════════════════════════════════════════════

PRIME_STRING_COLS = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status ",
]
PRIME_INT_COLS = ["BRANCH_ID", "RIMNO"]
PRIME_FLOAT_COLS = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD", "SETTLEMENT AMT",
]
PRIME_DATE_COLS = [
    "CREATION_DATE", "LAST_STATEMENT_DATE",
    "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE",
]
PRIME_COLUMNS_TO_DROP = [
    "MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD",
    "ORGANIZATION", "JOINING_FEE", "ANNUAL_FEE", "LAST_PAYMENT_AMOUNT",
    "LAST_PAYMENT_DATE", "SETTLEMENT AMT",
]

INACTIVE_STATUSES = [
    "CLSB", "CLSC", "LOST", "WROF", "CNCD", "SUSP", "FRAD",
    "CLSD", "BLOK", "NOAU", "EXMU", "PICK", "BLCK", "STLC",
    "OFBL", "ONBL", "FREZ", "EXPD", "EXPC", "CLSS",
]

TRANSACTION_STRING_COLS = [
    "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH",
    "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME",
]
TRANSACTION_INT_COLS = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY"]
TRANSACTION_FLOAT_COLS = [
    "ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT",
]
TRANSACTION_DATE_COLS = ["TRXN DATE", "POST DATE"]


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════

def _apply_cast(df: pd.DataFrame, columns: list, cast_type: str) -> None:
    """Cast *columns* in place. Silently skips missing columns."""
    for col in columns:
        if col not in df.columns:
            continue
        if cast_type == "string":
            df[col] = df[col].astype("string")
        elif cast_type == "float":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif cast_type == "int":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif cast_type == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _drop_empty(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop rows where any of *columns* is null."""
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    before = len(df)
    df = df.dropna(subset=valid, how="any")
    dropped = before - len(df)
    if dropped:
        print(f"  [clean] Dropped {dropped:,} rows missing {valid}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  Prime cleaning
# ═══════════════════════════════════════════════════════════════════

def clean_prime(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean all raw prime CSV files in *input_dir*.

    Reproduces the logic of ``prime_id_creation.py``:
      1. Cast columns to proper types
      2. Drop unnecessary columns
      3. Drop rows missing RIMNO
      4. Fill missing values (GENDER, BRANCH_NAME, DOB, etc.)
      5. Build a global CUSTOMER_ID mapping (RIMNO × DOB)
      6. Split into active / historical and remove relatives

    Returns
    -------
    (active_df, customer_id_mapping)
        *active_df* is the concatenation of all active records across
        every monthly file, with a ``CUSTOMER_ID`` column.
        *customer_id_mapping* is a DataFrame with columns
        ``[RIMNO, PRODUCT_NAME, CUSTOMER_ID]`` needed by
        ``clean_transactions``.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    print(f"[clean] Found {len(files)} prime file(s) in {input_dir}")

    # ── Pass 1: build global CUSTOMER_ID mapping ──────────────────
    all_pairs = []
    for f in files:
        temp = pd.read_csv(
            f, encoding="latin", usecols=["RIM_NO", "DOB"], dtype="string",
        )
        temp = temp.rename(columns={"RIM_NO": "RIMNO"})
        temp["RIMNO"] = temp["RIMNO"].str.strip()
        temp["DOB"] = pd.to_datetime(temp["DOB"], errors="coerce")
        all_pairs.append(temp[["RIMNO", "DOB"]])

    global_pairs = (
        pd.concat(all_pairs, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    global_pairs["CUSTOMER_ID"] = range(1, len(global_pairs) + 1)
    print(f"  [clean] Global unique (RIMNO, DOB) pairs: {len(global_pairs):,}")

    # ── Pass 2: process each file ─────────────────────────────────
    str_dtype = {
        col: "string"
        for col in PRIME_STRING_COLS + PRIME_INT_COLS + PRIME_FLOAT_COLS
    }

    active_frames = []

    for f in files:
        print(f"  [clean] Processing: {os.path.basename(f)}")
        df = pd.read_csv(
            f, encoding="latin", dtype=str_dtype, parse_dates=PRIME_DATE_COLS,
        ).rename(columns={"RIM_NO": "RIMNO", "NAME": "PRODUCT_NAME"})

        # Cast columns
        _apply_cast(df, PRIME_STRING_COLS, "string")
        _apply_cast(df, PRIME_FLOAT_COLS, "float")
        _apply_cast(df, PRIME_INT_COLS, "int")
        _apply_cast(df, PRIME_DATE_COLS, "date")

        # Drop unnecessary columns
        to_drop = [c for c in PRIME_COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=to_drop)

        # Drop rows missing RIMNO
        df = _drop_empty(df, ["RIMNO"])

        # Fill missing values
        if "GENDER" in df.columns:
            df["GENDER"] = df["GENDER"].fillna("Unknown")
        if "BRANCH_NAME" in df.columns:
            df["BRANCH_NAME"] = df["BRANCH_NAME"].fillna("Unknown")
        if "BRANCH_ID" in df.columns:
            df["BRANCH_ID"] = df["BRANCH_ID"].fillna(-1)

        df["DOB_WAS_MISSING"] = df["DOB"].isna().astype(int)
        if not df["DOB"].dropna().empty:
            median_dob_int = df["DOB"].dropna().astype("int64").median()
            median_dob = pd.to_datetime(median_dob_int)
            df["DOB"] = df["DOB"].fillna(median_dob)

        # Normalize STATUS columns
        if "STATUS" in df.columns:
            df["STATUS"] = df["STATUS"].astype("string").str.strip().str.upper()
        if "Card account status " in df.columns:
            df["Card account status "] = (
                df["Card account status "].astype("string").str.strip().str.upper()
            )

        # Assign CUSTOMER_ID
        df["RIMNO"] = df["RIMNO"].astype("string").str.strip()
        df = df.merge(global_pairs, on=["RIMNO", "DOB"], how="left")

        # Split active / historical
        is_inactive_status = df["STATUS"].isin(INACTIVE_STATUSES) if "STATUS" in df.columns else pd.Series(False, index=df.index)
        is_inactive_card = (
            df["Card account status "].isin(INACTIVE_STATUSES)
            if "Card account status " in df.columns
            else pd.Series(False, index=df.index)
        )
        is_historical = is_inactive_status & is_inactive_card
        active_df = df[~is_historical].copy()

        # Remove relatives (same RIMNO+PRODUCT_NAME, different CUSTOMER_ID)
        if "PRODUCT_NAME" in active_df.columns and len(active_df) > 0:
            dup_check = active_df.groupby(["RIMNO", "PRODUCT_NAME"])["CUSTOMER_ID"].nunique()
            dup_groups = dup_check[dup_check > 1].index
            if len(dup_groups) > 0:
                oldest_dob = (
                    active_df[active_df.set_index(["RIMNO", "PRODUCT_NAME"]).index.isin(dup_groups)]
                    .groupby(["RIMNO", "PRODUCT_NAME"])["DOB"]
                    .min()
                    .reset_index()
                    .rename(columns={"DOB": "OLDEST_DOB"})
                )
                active_df = active_df.merge(oldest_dob, on=["RIMNO", "PRODUCT_NAME"], how="left")
                is_dup = active_df.set_index(["RIMNO", "PRODUCT_NAME"]).index.isin(dup_groups)
                is_not_oldest = active_df["DOB"] != active_df["OLDEST_DOB"]
                active_df = active_df[~(is_dup & is_not_oldest)].copy()
                if "OLDEST_DOB" in active_df.columns:
                    active_df = active_df.drop(columns=["OLDEST_DOB"])

        print(f"    Active rows: {len(active_df):,}")
        active_frames.append(active_df)

    combined_active = pd.concat(active_frames, ignore_index=True)
    print(f"  [clean] Total active rows after cleaning: {len(combined_active):,}")

    # Build (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID lookup for transactions
    if "PRODUCT_NAME" in combined_active.columns:
        cid_mapping = (
            combined_active[["RIMNO", "PRODUCT_NAME", "CUSTOMER_ID"]]
            .drop_duplicates()
            .copy()
        )
    else:
        cid_mapping = (
            combined_active[["RIMNO", "CUSTOMER_ID"]]
            .drop_duplicates()
            .copy()
        )

    return combined_active, cid_mapping


# ═══════════════════════════════════════════════════════════════════
#  Transaction cleaning
# ═══════════════════════════════════════════════════════════════════

def clean_transactions(
    input_dir: str,
    customer_id_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Clean all raw transaction files in *input_dir*.

    Handles both ``.xlsx`` and ``.csv`` files transparently.

    Reproduces the logic of ``transaction_id_mapping.py``:
      1. Cast columns
      2. Map ``CUSTOMER_ID`` using the (RIMNO, PRODUCT_NAME) lookup
         produced by :func:`clean_prime`
      3. Return only the matched rows

    Parameters
    ----------
    input_dir
        Directory containing raw transaction files.
    customer_id_mapping
        DataFrame with ``[RIMNO, PRODUCT_NAME, CUSTOMER_ID]`` from
        :func:`clean_prime`.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction rows with ``CUSTOMER_ID``.
    """
    xlsx_files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    files = xlsx_files + csv_files
    if not files:
        raise FileNotFoundError(f"No xlsx/csv files found in {input_dir}")
    print(f"[clean] Found {len(files)} transaction file(s) in {input_dir}")

    str_dtype = {
        col: "string"
        for col in TRANSACTION_STRING_COLS + TRANSACTION_INT_COLS + TRANSACTION_FLOAT_COLS
    }

    frames = []
    for f in files:
        print(f"  [clean] Processing: {os.path.basename(f)}")
        ext = os.path.splitext(f)[1].lower()
        if ext == ".xlsx":
            df = pd.read_excel(
                f,
                dtype=str_dtype,
                parse_dates=TRANSACTION_DATE_COLS,
            )
        else:
            df = pd.read_csv(f, encoding="latin", dtype=str_dtype)
            # Parse dates for csv
            for col in TRANSACTION_DATE_COLS:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Rename DESCRIPTION -> PRODUCT_NAME (matches prime naming)
        if "DESCRIPTION" in df.columns:
            df = df.rename(columns={"DESCRIPTION": "PRODUCT_NAME"})

        # Cast columns
        _apply_cast(df, TRANSACTION_STRING_COLS, "string")
        _apply_cast(df, TRANSACTION_FLOAT_COLS, "float")
        _apply_cast(df, TRANSACTION_INT_COLS, "int")
        _apply_cast(df, TRANSACTION_DATE_COLS, "date")

        # Prepare merge keys
        df["RIMNO"] = df["RIMNO"].astype("string").str.strip()
        if "PRODUCT_NAME" in df.columns:
            df["PRODUCT_NAME"] = df["PRODUCT_NAME"].astype("string").str.strip()

        # Map CUSTOMER_ID
        if "PRODUCT_NAME" in customer_id_mapping.columns and "PRODUCT_NAME" in df.columns:
            merge_keys = ["RIMNO", "PRODUCT_NAME"]
        else:
            merge_keys = ["RIMNO"]

        # Ensure mapping types match
        mapping = customer_id_mapping.copy()
        for key in merge_keys:
            mapping[key] = mapping[key].astype("string").str.strip()

        df = df.merge(mapping, on=merge_keys, how="left")

        matched = df["CUSTOMER_ID"].notna().sum()
        unmatched = df["CUSTOMER_ID"].isna().sum()
        print(f"    Matched: {matched:,} | Unmatched: {unmatched:,}")

        # Keep only matched rows
        df = df[df["CUSTOMER_ID"].notna()].copy()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  [clean] Total cleaned transaction rows: {len(combined):,}")
    return combined
