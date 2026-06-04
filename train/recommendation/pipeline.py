"""
pipeline.py
===========
Refactored analytics & XGBoost pipeline exposed as callable functions
for the Flask GUI.
"""

import pandas as pd
import numpy as np
import glob
import os
import io
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from xgboost import XGBClassifier
import pickle

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Helper Functions (from analytics.py)
# ---------------------------------------------------------------------------
def apply_cast_and_report(df, columns, cast_type, logs):
    """Casts columns to specific types and collects a missing data report."""
    total_rows = len(df)
    for col in columns:
        if col not in df.columns:
            logs.append(f"  [Skip] Column '{col}' not found in dataframe.")
            continue
        nulls_before = df[col].isna().sum()
        if cast_type == 'string':
            df[col] = df[col].astype("string")
        elif cast_type == 'float':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif cast_type == 'int':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif cast_type == 'date':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        nulls_after = df[col].isna().sum()
        pct_missing = (nulls_after / total_rows) * 100
        logs.append(f"  [{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}%)")


# ---------------------------------------------------------------------------
# PHASE 0 — run_prestep  (prime_id + transaction_id cleaning)
# ---------------------------------------------------------------------------

# Status codes that indicate an inactive / closed / blocked card
INACTIVE_STATUSES = [
    'CLSB', 'CLSC', 'LOST', 'WROF', 'CNCD', 'SUSP', 'FRAD', 'CLSD',
    'BLOK', 'NOAU', 'EXMU', 'PICK', 'BLCK', 'STLC', 'OFBL', 'ONBL',
    'FREZ', 'EXPD', 'EXPC', 'CLSS',
]


def run_prestep(raw_prime_dir, raw_transaction_dir,
                prime_output_dir="prime_cleaned",
                transaction_output_dir="transaction_cleaned"):
    """
    Runs the data-cleaning prestep that was previously in prime_id.py and
    transaction_id.py.

    1. Reads raw prime CSVs  → builds global CUSTOMER_ID mapping, casts
       columns, splits active / historical / relatives → saves to
       *prime_output_dir*.
    2. Reads the active prime output → builds (RIMNO, PRODUCT_NAME) →
       CUSTOMER_ID lookup → processes raw transaction XLSX files → saves
       matched / unmatched to *transaction_output_dir*.

    Returns
    -------
    logs : list[str]
        Human-readable processing log.
    """
    logs = []
    os.makedirs(prime_output_dir, exist_ok=True)
    os.makedirs(transaction_output_dir, exist_ok=True)

    # ── Column definitions (prime) ──
    prime_string_cols = [
        "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
        "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "
    ]
    prime_int_cols = ["BRANCH_ID", "RIMNO"]
    prime_float_cols = [
        "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
        "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
        "THIRD_REPLACED_CARD"
    ]
    prime_date_cols = ["CREATION_DATE", "LAST_STATEMENT_DATE", "DOB", "CLOSURE_DATE"]

    # ── Column definitions (transaction) ──
    txn_string_cols = [
        "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH",
        "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME"
    ]
    txn_int_cols = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY"]
    txn_float_cols = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
    txn_date_cols = ["TRXN DATE", "POST DATE"]

    # ====================================================================
    #  PRESTEP PART 1 — Prime Cleaning  (from prime_id.py)
    # ====================================================================
    logs.append("=" * 60)
    logs.append(" PRESTEP 1: Cleaning Raw Prime Files")
    logs.append("=" * 60)

    prime_files = glob.glob(os.path.join(raw_prime_dir, "*.csv"))
    if not prime_files:
        raise FileNotFoundError(
            f"No CSV files found in '{raw_prime_dir}'. "
            "Make sure the raw prime folder exists and contains CSVs."
        )
    logs.append(f"Found {len(prime_files)} raw prime file(s).")

    # ── PASS 1: Build global CUSTOMER_ID mapping ──
    logs.append("")
    logs.append("PASS 1: Scanning all files to build global CUSTOMER_ID mapping...")

    all_pairs = []
    for file in prime_files:
        logs.append(f"  Scanning: {file}")
        temp_df = pd.read_csv(
            file, encoding='latin',
            dtype='string'
        )
        temp_df = temp_df.rename(columns={'RIM_NO': 'RIMNO', 'ï»¿BRANCH_ID': 'BRANCH_ID'})
        temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
        temp_df['DOB'] = pd.to_datetime(temp_df['DOB'], errors='coerce')
        all_pairs.append(temp_df[['RIMNO', 'DOB']])

    global_pairs = (
        pd.concat(all_pairs, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    global_pairs['CUSTOMER_ID'] = range(1, len(global_pairs) + 1)

    logs.append(f"  Global unique (RIMNO, DOB) pairs: {len(global_pairs)}")
    logs.append(f"  CUSTOMER_ID range: 1 – {len(global_pairs)}")

    # ── PASS 2: Process each prime file ──
    logs.append("")
    logs.append("PASS 2: Processing each file with the global CUSTOMER_ID mapping...")

    for file in prime_files:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        logs.append("")
        logs.append(f"{'=' * 50}")
        logs.append(f"  Processing: {file}")
        logs.append(f"{'=' * 50}")

        df = pd.read_csv(
            file, encoding='latin',
            dtype={c: "string" for c in prime_string_cols + prime_int_cols + prime_float_cols},
            parse_dates=prime_date_cols
        ).rename(columns={'NAME': 'PRODUCT_NAME', 'RIM_NO': 'RIMNO', 'ï»¿BRANCH_ID': 'BRANCH_ID'})

        logs.append(f"  Rows loaded: {len(df)}")

        # Cast
        apply_cast_and_report(df, prime_string_cols, 'string', logs)
        apply_cast_and_report(df, prime_float_cols, 'float', logs)
        apply_cast_and_report(df, prime_int_cols, 'int', logs)
        apply_cast_and_report(df, prime_date_cols, 'date', logs)

        # Clean STATUS columns
        df['STATUS'] = df['STATUS'].astype("string").str.strip().str.upper()
        df['Card account status '] = df['Card account status '].astype("string").str.strip().str.upper()

        # Merge CUSTOMER_ID
        df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
        df = df.merge(global_pairs, on=['RIMNO', 'DOB'], how='left')

        assigned = df['CUSTOMER_ID'].notna().sum()
        unmatched = df['CUSTOMER_ID'].isna().sum()
        logs.append(f"  CUSTOMER_ID assigned: {assigned} rows")
        if unmatched > 0:
            logs.append(f"  WARNING: {unmatched} rows could not be assigned a CUSTOMER_ID.")

        # Split active / historical
        is_inactive_status = df['STATUS'].isin(INACTIVE_STATUSES)
        is_inactive_card = df['Card account status '].isin(INACTIVE_STATUSES)
        is_historical = is_inactive_status & is_inactive_card

        historical_df = df[is_historical].copy()
        active_df = df[~is_historical].copy()

        logs.append(f"  Active rows:     {len(active_df)}")
        logs.append(f"  Historical rows: {len(historical_df)}")

        # Detect relatives: same (RIMNO, PRODUCT_NAME) but different CUSTOMER_ID
        dup_check = active_df.groupby(['RIMNO', 'PRODUCT_NAME'])['CUSTOMER_ID'].nunique()
        dup_groups = dup_check[dup_check > 1].index

        if len(dup_groups) > 0:
            logs.append(f"  Found {len(dup_groups)} (RIMNO, PRODUCT_NAME) pair(s) with multiple CUSTOMER_IDs.")

            oldest_dob = (
                active_df[active_df.set_index(['RIMNO', 'PRODUCT_NAME']).index.isin(dup_groups)]
                .groupby(['RIMNO', 'PRODUCT_NAME'])['DOB']
                .min()
                .reset_index()
                .rename(columns={'DOB': 'OLDEST_DOB'})
            )
            active_df = active_df.merge(oldest_dob, on=['RIMNO', 'PRODUCT_NAME'], how='left')

            is_dup_group = active_df.set_index(['RIMNO', 'PRODUCT_NAME']).index.isin(dup_groups)
            is_not_oldest = active_df['DOB'] != active_df['OLDEST_DOB']

            relatives_df = active_df[is_dup_group & is_not_oldest].copy()
            active_df = active_df[~(is_dup_group & is_not_oldest)].copy()

            active_df = active_df.drop(columns=['OLDEST_DOB'])
            relatives_df = relatives_df.drop(columns=['OLDEST_DOB'])

            logs.append(f"  Moved {len(relatives_df)} row(s) to active_relatives.")
            logs.append(f"  Active after dedup: {len(active_df)} rows")
        else:
            relatives_df = pd.DataFrame()
            logs.append("  No (RIMNO, PRODUCT_NAME) duplicates found.")

        # Save
        active_path = os.path.join(prime_output_dir, f"{file_basename}_active.csv")
        historical_path = os.path.join(prime_output_dir, f"{file_basename}_historical.csv")

        active_df.to_csv(active_path, index=False)
        historical_df.to_csv(historical_path, index=False)
        logs.append(f"  Saved -> {active_path}")
        logs.append(f"  Saved -> {historical_path}")

        if len(relatives_df) > 0:
            relatives_path = os.path.join(prime_output_dir, f"{file_basename}_active_relatives.csv")
            relatives_df.to_csv(relatives_path, index=False)
            logs.append(f"  Saved -> {relatives_path}")

    logs.append("")
    logs.append("Prime prestep complete.")

    # ====================================================================
    #  PRESTEP PART 2 — Transaction Cleaning  (from transaction_id.py)
    # ====================================================================
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PRESTEP 2: Cleaning Raw Transaction Files")
    logs.append("=" * 60)

    # Build CUSTOMER_ID lookup from the active prime files we just created
    active_files = glob.glob(os.path.join(prime_output_dir, "*_active.csv"))
    if not active_files:
        raise FileNotFoundError(
            f"No active prime files found in '{prime_output_dir}/'. "
            "Prime prestep may have failed."
        )

    logs.append(f"Building (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID lookup from {len(active_files)} active file(s)...")

    mapping_list = []
    for file in active_files:
        logs.append(f"  -> {file}")
        temp_df = pd.read_csv(
            file, encoding='latin',
            usecols=['RIMNO', 'PRODUCT_NAME', 'CUSTOMER_ID'],
            dtype='string'
        )
        temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
        temp_df['PRODUCT_NAME'] = temp_df['PRODUCT_NAME'].str.strip()
        mapping_list.append(temp_df)

    customer_lookup = pd.concat(mapping_list, ignore_index=True).drop_duplicates()
    customer_lookup['CUSTOMER_ID'] = pd.to_numeric(
        customer_lookup['CUSTOMER_ID'], errors='coerce'
    ).astype('Int64')

    logs.append(f"  Unique mappings: {len(customer_lookup)}")

    # Process transaction files
    transaction_files = glob.glob(os.path.join(raw_transaction_dir, "*.xlsx"))
    if not transaction_files:
        logs.append(f"WARNING: No .xlsx files found in '{raw_transaction_dir}'. Skipping transaction prestep.")
        return logs

    logs.append(f"Found {len(transaction_files)} transaction file(s).")

    for file in transaction_files:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        logs.append("")
        logs.append(f"{'=' * 50}")
        logs.append(f"  Processing: {file}")
        logs.append(f"{'=' * 50}")

        df = pd.read_excel(
            file,
            dtype={c: "string" for c in txn_string_cols + txn_int_cols + txn_float_cols},
            parse_dates=txn_date_cols
        ).rename(columns={'DESCRIPTION': 'PRODUCT_NAME'})

        logs.append(f"  Rows loaded: {len(df)}")

        apply_cast_and_report(df, txn_string_cols, 'string', logs)
        apply_cast_and_report(df, txn_float_cols, 'float', logs)
        apply_cast_and_report(df, txn_int_cols, 'int', logs)
        apply_cast_and_report(df, txn_date_cols, 'date', logs)

        # Map CUSTOMER_ID
        df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
        df['PRODUCT_NAME'] = df['PRODUCT_NAME'].astype("string").str.strip()
        df = df.merge(customer_lookup, on=['RIMNO', 'PRODUCT_NAME'], how='left')

        matched_count = df['CUSTOMER_ID'].notna().sum()
        unmatched_count = df['CUSTOMER_ID'].isna().sum()
        logs.append(f"  CUSTOMER_ID mapped: {matched_count} rows")
        if unmatched_count > 0:
            logs.append(f"  Missing CUSTOMER_ID: {unmatched_count} rows")

        # Split matched / unmatched
        matched_df = df[df['CUSTOMER_ID'].notna()].copy()
        missing_df = df[df['CUSTOMER_ID'].isna()].copy()

        output_path = os.path.join(transaction_output_dir, f"{file_basename}.csv")
        matched_df.to_csv(output_path, index=False)
        logs.append(f"  Saved -> {output_path}")

        if len(missing_df) > 0:
            missing_path = os.path.join(transaction_output_dir, f"{file_basename}_missing_id.csv")
            missing_df.to_csv(missing_path, index=False)
            logs.append(f"  Saved -> {missing_path}")

    logs.append("")
    logs.append("Transaction prestep complete.")
    logs.append("=" * 60)

    return logs


# ---------------------------------------------------------------------------
# PHASE 1 — run_analytics_pipeline
# ---------------------------------------------------------------------------
def run_analytics_pipeline(prime_dir, transaction_dir=None,
                           raw_prime_dir=None, raw_transaction_dir=None):
    """
    Runs the full analytics pipeline from the given directory.

    If *raw_prime_dir* and *raw_transaction_dir* are supplied the prestep
    (CUSTOMER_ID creation, active/historical splitting, transaction mapping)
    is executed first, writing cleaned files into *prime_dir* and
    *transaction_dir* respectively.

    Returns (final_customer_profile DataFrame, logs list, product_cols list, feature_cols list).
    """
    logs = []

    if transaction_dir is None:
        transaction_dir = "transaction_cleaned"

    # ── Optional prestep ──
    if raw_prime_dir and raw_transaction_dir:
        logs.append("Running data-cleaning prestep...")
        logs.append("")
        prestep_logs = run_prestep(
            raw_prime_dir=raw_prime_dir,
            raw_transaction_dir=raw_transaction_dir,
            prime_output_dir=prime_dir,
            transaction_output_dir=transaction_dir,
        )
        logs.extend(prestep_logs)
        logs.append("")


    # ── Column definitions ──
    prime_string_cols = ["BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME", "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "]
    prime_int_cols = ["BRANCH_ID", "RIMNO", "CUSTOMER_ID"]
    prime_float_cols = ["CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT", "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD", "THIRD_REPLACED_CARD", "SETTLEMENT AMT"]
    prime_date_cols = ["CREATION_DATE", "LAST_STATEMENT_DATE", "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE"]

    transaction_string_cols = ["DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME"]
    transaction_int_cols = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY", "CUSTOMER_ID"]
    transaction_float_cols = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
    transaction_date_cols = ["TRXN DATE", "POST DATE"]

    # ── Phase 1: Consolidate Active Prime ──
    logs.append("=" * 60)
    logs.append(" PHASE 1: Consolidating Active Prime Files")
    logs.append("=" * 60)

    prime_files = glob.glob(os.path.join(prime_dir, "*_active.csv"))
    if not prime_files:
        raise FileNotFoundError(f"No active prime files found in '{prime_dir}'.")

    prime_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in prime_files]
    prime_df = pd.concat(prime_dfs, ignore_index=True)
    logs.append(f"Loaded {len(prime_df)} total rows from {len(prime_files)} files.")

    apply_cast_and_report(prime_df, prime_string_cols, 'string', logs)
    apply_cast_and_report(prime_df, prime_int_cols, 'int', logs)
    apply_cast_and_report(prime_df, prime_float_cols, 'float', logs)
    apply_cast_and_report(prime_df, prime_date_cols, 'date', logs)

    # ── Phase 2: Consolidate Transactions ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PHASE 2: Consolidating Matched Transaction Files")
    logs.append("=" * 60)

    all_txn_files = glob.glob(os.path.join(transaction_dir, "*.csv"))
    txn_files = [f for f in all_txn_files if not f.endswith("_missing_id.csv")]
    if not txn_files:
        raise FileNotFoundError(f"No matched transaction files found in '{transaction_dir}'.")

    txn_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in txn_files]
    transaction_df = pd.concat(txn_dfs, ignore_index=True)
    logs.append(f"Loaded {len(transaction_df)} total rows from {len(txn_files)} files.")

    apply_cast_and_report(transaction_df, transaction_string_cols, 'string', logs)
    apply_cast_and_report(transaction_df, transaction_int_cols, 'int', logs)
    apply_cast_and_report(transaction_df, transaction_float_cols, 'float', logs)
    apply_cast_and_report(transaction_df, transaction_date_cols, 'date', logs)

    # ── Phase 3: Cleanup ──
    prime_columns_to_drop = [
        "MAPPING_ACCNO", "STATUS", "CREATION_DATE", "MIN_PAYMENT", "OVER_LIMIT",
        "ACTIVATED", "DELIQUENCY", "STATUS_NAME", "OVER_LIMIT", "TOTAL_HOLD",
        "ORGANIZATION", "JOINING_FEE", "ANNUAL_FEE", "LAST_PAYMENT_AMOUNT",
        "LAST_PAYMENT_DATE", "SETTLEMENT AMT", 'FIRST_REPLACED_CARD',
        'SECOND_REPLACED_CARD', 'THIRD_REPLACED_CARD', 'LAST_STATEMENT_DATE',
        "LEDGER_BALANCE", "AVAILABLE_LIMIT", "CLOSURE_DATE",
        "Card account status ", "CUSTOMER_TYPE", "OVERDUEAMOUNT"
    ]
    existing_cols_to_drop = [col for col in prime_columns_to_drop if col in prime_df.columns]
    prime_df = prime_df.drop(columns=existing_cols_to_drop)

    transaction_columns_to_drop = ["POST DATE", "ORIG AMOUNT", "EMBEDDED _FEE", "SETTLEMENT AMT", "SETTLEMENT CCY", "SOURCES"]
    existing_cols_to_drop = [col for col in transaction_columns_to_drop if col in transaction_df.columns]
    transaction_df = transaction_df.drop(columns=existing_cols_to_drop)

    prime_df["GENDER"] = prime_df["GENDER"].fillna("Unknown")

    # ── User-item matrix ──
    user_item_matrix = pd.crosstab(prime_df['CUSTOMER_ID'], prime_df['PRODUCT_NAME'])
    user_item_matrix.columns = [f"HAS_PROD_{col.strip().replace(' ', '_')}" for col in user_item_matrix.columns]
    user_item_matrix = (user_item_matrix > 0).astype(int)
    prime_df = prime_df.drop_duplicates(subset=['CUSTOMER_ID']).merge(user_item_matrix, on='CUSTOMER_ID', how='inner')

    # ── RFM Features ──
    rfm_features = transaction_df.groupby('CUSTOMER_ID').agg(
        TOTAL_SPEND_AMT=('BILLING AMT', 'sum'),
        AVG_TRXN_AMT=('BILLING AMT', 'mean'),
        TRXN_COUNT=('BILLING AMT', 'count'),
        DAYS_SINCE_LAST_TRXN=('TRXN DATE', lambda x: (pd.to_datetime('today') - x.max()).days)
    ).reset_index()

    # ── MCC Spend ──
    mcc_spend = pd.pivot_table(
        transaction_df, values='BILLING AMT', index='CUSTOMER_ID',
        columns='MCC', aggfunc='sum', fill_value=0
    )
    mcc_spend.columns = [f"MCC_{str(col)}_SPEND" for col in mcc_spend.columns]
    mcc_spend = mcc_spend.reset_index()

    transaction_df['IS_FOREIGN_TRXN'] = (transaction_df['TRXN COUNTRY'] != 'EGYPT').fillna(False).astype(int)

    extraction_date = pd.to_datetime("today")
    prime_df['AGE'] = (extraction_date - prime_df['DOB']).dt.days // 365
    bins = [18, 25, 35, 50, 65, 100]
    labels = ['18-25', '26-35', '36-50', '51-65', '65+']
    prime_df['AGE_GROUP'] = pd.cut(prime_df['AGE'], bins=bins, labels=labels, right=True)

    foreign_agg = transaction_df.groupby('CUSTOMER_ID').agg(
        FOREIGN_TRXN_COUNT=('IS_FOREIGN_TRXN', 'sum')
    ).reset_index()

    # ── Final Customer 360 Merge ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PHASE 3: Building Final Customer Profile (1 Row per Customer)")
    logs.append("=" * 60)

    final_customer_profile = prime_df.copy()
    final_customer_profile = final_customer_profile.merge(rfm_features, on='CUSTOMER_ID', how='inner')
    final_customer_profile = final_customer_profile.merge(mcc_spend, on='CUSTOMER_ID', how='inner')
    final_customer_profile = final_customer_profile.merge(foreign_agg, on='CUSTOMER_ID', how='left')

    mcc_cols = [col for col in mcc_spend.columns if col != 'CUSTOMER_ID']
    fill_zero_cols = ['TOTAL_SPEND_AMT', 'AVG_TRXN_AMT', 'TRXN_COUNT', 'FOREIGN_TRXN_COUNT'] + mcc_cols
    final_customer_profile[fill_zero_cols] = final_customer_profile[fill_zero_cols].fillna(0)
    final_customer_profile['DAYS_SINCE_LAST_TRXN'] = final_customer_profile['DAYS_SINCE_LAST_TRXN'].fillna(9999)

    # One-hot encode AGE_GROUP
    age_group_dummies = pd.get_dummies(final_customer_profile['AGE_GROUP'], prefix='AGE_GROUP', drop_first=False)
    age_group_dummies.columns = [col.replace('-', '_') for col in age_group_dummies.columns]
    final_customer_profile = pd.concat([final_customer_profile, age_group_dummies], axis=1)
    final_customer_profile = final_customer_profile.drop(columns=['AGE_GROUP'])

    # One-hot encode GENDER
    gender_dummies = pd.get_dummies(final_customer_profile['GENDER'], prefix='GENDER', drop_first=False)
    final_customer_profile = pd.concat([final_customer_profile, gender_dummies], axis=1)
    final_customer_profile = final_customer_profile.drop(columns=['GENDER'])

    # One-hot encode BRANCH_ID
    branch_id_dummies = pd.get_dummies(final_customer_profile['BRANCH_ID'], prefix='BRANCH_ID', drop_first=False)
    final_customer_profile = pd.concat([final_customer_profile, branch_id_dummies], axis=1)
    final_customer_profile = final_customer_profile.drop(columns=['BRANCH_ID'])

    final_columns_to_drop = ['AGE', "BRANCH_NAME", "PRODUCT_NAME", "DOB", "RIMNO", "DOB_WAS_MISSING", "GENDER_Unknown"]
    existing_cols_to_drop = [col for col in final_columns_to_drop if col in final_customer_profile.columns]
    final_customer_profile = final_customer_profile.drop(columns=existing_cols_to_drop)

    # ── Correlation filtering ──
    final_customer_profile = final_customer_profile.drop(columns=["BRANCH_ID"], errors='ignore')

    CORR_THRESHOLD = 0.1
    product_cols = [col for col in final_customer_profile.columns if col.startswith('HAS_PROD_')]
    exclude_cols_corr = ['CUSTOMER_ID'] + product_cols
    feature_cols = [col for col in final_customer_profile.columns if col not in exclude_cols_corr]

    logs.append(f"Analyzing {len(feature_cols)} features against {len(product_cols)} products...")

    corr_matrix = final_customer_profile[feature_cols + product_cols].corr()
    feature_product_corr = corr_matrix.loc[feature_cols, product_cols]
    max_corr_per_feature = feature_product_corr.abs().max(axis=1)

    features_to_keep = max_corr_per_feature[max_corr_per_feature >= CORR_THRESHOLD].index.tolist()
    features_to_drop = max_corr_per_feature[max_corr_per_feature < CORR_THRESHOLD].index.tolist()

    logs.append(f"Keeping {len(features_to_keep)} strongly correlated features.")
    logs.append(f"Dropping {len(features_to_drop)} weak/uncorrelated features.")

    final_customer_profile = final_customer_profile.drop(columns=features_to_drop)

    # Drop low-volume products
    current_product_cols = [col for col in final_customer_profile.columns if col.startswith('HAS_PROD_')]
    product_counts = final_customer_profile[current_product_cols].sum()
    products_to_drop = product_counts[product_counts < 100].index.tolist()
    logs.append(f"Dropping {len(products_to_drop)} products with fewer than 100 holders.")
    if products_to_drop:
        final_customer_profile = final_customer_profile.drop(columns=products_to_drop)

    # Final summary
    final_products = [col for col in final_customer_profile.columns if col.startswith('HAS_PROD_')]
    final_features = [col for col in final_customer_profile.columns if col not in final_products and col != 'CUSTOMER_ID']

    logs.append("")
    logs.append("=" * 60)
    logs.append(" FINAL DATASET SUMMARY")
    logs.append("=" * 60)
    logs.append(f"Total Customers:         {len(final_customer_profile)}")
    logs.append(f"Total Features Retained: {len(final_features)}")
    logs.append(f"Total Products Retained: {len(final_products)}")

    logs.append("")
    logs.append("Extracted Features:")
    for i, feat in enumerate(final_features, 1):
        logs.append(f"  {i:3d}. {feat}")

    logs.append("")
    logs.append("Target Products:")
    for i, prod in enumerate(final_products, 1):
        logs.append(f"  {i:3d}. {prod.replace('HAS_PROD_', '')}")

    return final_customer_profile, logs, final_products, final_features


# ---------------------------------------------------------------------------
# PHASE 2 — train_xgboost
# ---------------------------------------------------------------------------
def train_xgboost(df):
    """
    Trains one XGBoost model per product with threshold tuning.
    Returns (models_dict, optimal_thresholds, feature_cols, valid_targets, metrics_dict, logs).
    """
    logs = []
    logs.append("=" * 60)
    logs.append(" XGBoost + Per-Product Threshold Tuning")
    logs.append("=" * 60)

    target_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]
    exclude_cols = ['CUSTOMER_ID', 'BRANCH_ID'] + target_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0)

    MIN_SAMPLES_REQUIRED = 50
    valid_targets = [col for col in target_cols if df[col].sum() >= MIN_SAMPLES_REQUIRED]

    logs.append(f"Total Features:  {len(feature_cols)}")
    logs.append(f"Valid Products:  {len(valid_targets)}")

    Y = df[valid_targets]

    # Three-way split
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=42)

    logs.append(f"Train size:      {len(X_train)}")
    logs.append(f"Validation size: {len(X_val)}")
    logs.append(f"Test size:       {len(X_test)}")

    # Save test set customer IDs to a text file
    test_customer_ids = df.loc[X_test.index, 'CUSTOMER_ID'].astype(int).tolist()
    with open("test_customer_ids.txt", "w") as f:
        for cid in test_customer_ids:
            f.write(f"{cid}\n")
    logs.append(f"Saved {len(test_customer_ids)} test customer IDs to test_customer_ids.txt")

    # Train one XGBoost per product
    val_proba_dict = {}
    test_proba_dict = {}
    models_dict = {}

    logs.append("")
    logs.append("Training XGBoost models (one per product)...")

    for i, product in enumerate(valid_targets):
        y_train_col = Y_train[product].values
        positives = y_train_col.sum()
        negatives = len(y_train_col) - positives
        spw = negatives / (positives + 1e-5)

        xgb_model = XGBClassifier(
            scale_pos_weight=spw,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train_col)
        models_dict[product] = xgb_model

        val_proba_dict[product] = xgb_model.predict_proba(X_val)[:, 1]
        test_proba_dict[product] = xgb_model.predict_proba(X_test)[:, 1]

        if (i + 1) % 10 == 0 or (i + 1) == len(valid_targets):
            logs.append(f"  Trained {i+1}/{len(valid_targets)} models "
                        f"(last: {product.replace('HAS_PROD_', '')}  spw={spw:.1f})")

    logs.append("All models trained.")

    # Threshold optimization
    logs.append("")
    logs.append("Optimizing decision thresholds on validation set...")
    optimal_thresholds = {}

    for product in valid_targets:
        val_proba = val_proba_dict[product]
        y_val_col = Y_val[product].values
        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (val_proba >= t).astype(int)
            score = f1_score(y_val_col, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = round(float(t), 2)
        optimal_thresholds[product] = best_threshold

    # Apply on test set
    predictions_dict = {}
    for product in valid_targets:
        t = optimal_thresholds[product]
        predictions_dict[product] = (test_proba_dict[product] >= t).astype(int)

    Y_pred = pd.DataFrame(predictions_dict, index=Y_test.index)

    # Metrics
    exact_acc = accuracy_score(Y_test, Y_pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='weighted', zero_division=0)
    samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='samples', zero_division=0)

    metrics = {
        "exact_match_accuracy": round(exact_acc * 100, 2),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
        "samples": {"precision": round(samples_p, 4), "recall": round(samples_r, 4), "f1": round(samples_f1, 4)},
    }

    # Per-product threshold table
    threshold_table = []
    for p in valid_targets:
        threshold_table.append({
            "product": p.replace('HAS_PROD_', ''),
            "positives_in_train": int(Y_train[p].sum()),
            "optimal_threshold": optimal_thresholds[p]
        })

    metrics["threshold_table"] = threshold_table

    logs.append("")
    logs.append(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
    logs.append(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}")

    return models_dict, optimal_thresholds, feature_cols, valid_targets, metrics, logs


# ---------------------------------------------------------------------------
# PHASE 3 — predict_for_customer
# ---------------------------------------------------------------------------
def predict_for_customer(customer_id, df, models_dict, optimal_thresholds, feature_cols, valid_targets):
    """
    Given a customer ID, uses trained XGBoost models to predict which products
    the customer is likely to hold.
    Returns (model_predictions list, already_holding list, new_recommendations list, customer_found bool).
    """
    customer_row = df[df['CUSTOMER_ID'] == customer_id]
    if customer_row.empty:
        return [], [], [], False

    X_customer = customer_row[feature_cols].fillna(0)

    # --- 1. Ground truth: what the customer ACTUALLY holds in the data ---
    already_holding = []
    for product in valid_targets:
        if product in customer_row.columns:
            if int(customer_row[product].values[0]) == 1:
                already_holding.append(product.replace('HAS_PROD_', ''))

    # --- 2. Model predictions: run every model and collect results ---
    model_predictions = []       # all products the model predicts (above threshold)

    for product in valid_targets:
        model = models_dict[product]
        proba = model.predict_proba(X_customer)[:, 1][0]
        threshold = optimal_thresholds[product]
        product_name = product.replace('HAS_PROD_', '')
        currently_holds = product_name in already_holding

        if proba >= threshold:
            model_predictions.append({
                "product": product_name,
                "probability": round(float(proba) * 100, 2),
                "threshold": threshold,
                "currently_holds": currently_holds
            })
    return model_predictions, already_holding, True


# ---------------------------------------------------------------------------
# PHASE 4 — Content-Based Filtering (CBF)
# ---------------------------------------------------------------------------
def train_cbf(df):
    """
    Builds an item-item cosine similarity matrix, evaluates with a
    masking-based train/test split, and finds optimal similarity thresholds.
    Returns (sim_matrix, product_cols, cbf_thresholds, metrics, logs).
    """
    logs = []
    logs.append("=" * 60)
    logs.append(" Content-Based Filtering (Item-Item Similarity)")
    logs.append("=" * 60)

    product_cols = [col for col in df.columns if col.startswith('HAS_PROD_')]
    user_item = df.set_index('CUSTOMER_ID')[product_cols]
    user_item = (user_item > 0).astype(int)

    logs.append(f"Users:    {user_item.shape[0]}")
    logs.append(f"Products: {user_item.shape[1]}")

    # ── Masking-based train/test split ──
    np.random.seed(42)
    train_matrix = user_item.copy()
    test_matrix = pd.DataFrame(0, index=user_item.index, columns=user_item.columns)

    users_with_test = 0
    for idx in range(user_item.shape[0]):
        held = np.where(user_item.iloc[idx].values > 0)[0]
        if len(held) > 1:
            n_mask = max(1, int(len(held) * 0.2))
            masked = np.random.choice(held, size=n_mask, replace=False)
            test_matrix.iloc[idx, masked] = 1
            train_matrix.iloc[idx, masked] = 0
            users_with_test += 1

    logs.append(f"Train/Test masking: {users_with_test} users have masked items.")

    # ── Build similarity on TRAIN data only ──
    sim_array = cosine_similarity(train_matrix.T)
    sim_matrix = pd.DataFrame(sim_array, index=product_cols, columns=product_cols)
    # ── Precompute score matrix (users × products) via matrix multiply ──
    # score[u, p] = sum of similarities between product p and all products user u holds
    train_np = train_matrix.values        # (n_users, n_products)
    sim_np   = sim_matrix.values          # (n_products, n_products)
    score_matrix = train_np @ sim_np      # (n_users, n_products)

    # Mask: set score to -inf for products the user already holds (so they won't be predicted)
    held_mask = train_np > 0
    score_matrix_masked = score_matrix.copy()
    score_matrix_masked[held_mask] = -np.inf

    # Identify users who have test items
    test_np = test_matrix.values
    test_user_mask = test_np.sum(axis=1) > 0  # (n_users,) bool

    logs.append(f"Evaluable users: {test_user_mask.sum()}")

    # ── Vectorized threshold sweep per product ──
    logs.append("")
    logs.append("Optimizing similarity thresholds...")

    from sklearn.metrics import f1_score as f1_fn

    cbf_thresholds = {}
    thresholds_to_try = np.arange(0.1, 2.1, 0.1)

    for p_idx, product in enumerate(product_cols):
        scores_col = score_matrix_masked[test_user_mask, p_idx]   # scores for this product, evaluable users only
        y_true_col = test_np[test_user_mask, p_idx]

        best_t = 0.5
        best_f1 = 0.0
        for t in thresholds_to_try:
            y_pred_col = (scores_col >= t).astype(int)
            f1_val = f1_fn(y_true_col, y_pred_col, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_t = round(float(t), 2)
        cbf_thresholds[product] = best_t

    logs.append("Thresholds optimized.")

    # ── Vectorized evaluation on test set ──
    logs.append("")
    logs.append("Evaluating on test set...")

    threshold_arr = np.array([cbf_thresholds[p] for p in product_cols])   # (n_products,)
    Y_pred = (score_matrix_masked[test_user_mask] >= threshold_arr).astype(int)
    Y_true = test_np[test_user_mask]

    exact_acc = accuracy_score(Y_true, Y_pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='micro', zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='macro', zero_division=0
    )

    metrics = {
        "exact_match_accuracy": round(exact_acc * 100, 2),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
    }

    logs.append(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
    logs.append(f"Micro  — P: {micro_p:.4f}  R: {micro_r:.4f}  F1: {micro_f1:.4f}")
    logs.append(f"Macro  — P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}")

    # ── Rebuild similarity on FULL data for inference ──
    full_sim_array = cosine_similarity(user_item.T)
    sim_matrix_full = pd.DataFrame(full_sim_array, index=product_cols, columns=product_cols)

    return sim_matrix_full, product_cols, cbf_thresholds, metrics, logs


def predict_cbf_for_customer(customer_id, df, sim_matrix, product_cols, cbf_thresholds):
    """
    Recommends products based on item-item similarity using per-product
    thresholds. Returns (recommendations list, already_holding list, found bool).
    """
    customer_row = df[df['CUSTOMER_ID'] == customer_id]
    if customer_row.empty:
        return [], [], False

    held_cols = []
    for col in product_cols:
        if col in customer_row.columns:
            val = customer_row[col].values[0]
            if pd.notna(val) and float(val) == 1:
                held_cols.append(col)
    already_holding = [col.replace('HAS_PROD_', '') for col in held_cols]

    if not held_cols:
        return [], already_holding, True

    recommendations = []
    for product in product_cols:
        if product in held_cols:
            continue
        score = sim_matrix.loc[product, held_cols].sum()
        threshold = cbf_thresholds.get(product, 0.5)
        if score >= threshold:
            recommendations.append({
                "product": product.replace('HAS_PROD_', ''),
                "similarity_score": round(float(score), 4),
                "threshold": threshold,
            })

    # Sort by score descending
    recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)

    return recommendations, already_holding, True
