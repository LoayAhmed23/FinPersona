import os
import glob
import pandas as pd
import config

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


def run_prestep(raw_prime_dir, raw_transaction_dir,
                prime_output_dir="prime_cleaned",
                transaction_output_dir="transaction_cleaned",
                logs=None):
    """
    Runs the data-cleaning prestep that was previously in prime_id.py and
    transaction_id.py.

    Returns
    -------
    logs : list[str]
        Human-readable processing log.
    """
    if logs is None:
        logs = []
    os.makedirs(prime_output_dir, exist_ok=True)
    os.makedirs(transaction_output_dir, exist_ok=True)

    # ====================================================================
    #  PRESTEP PART 1 — Prime Cleaning
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
        if 'RIMNO' in temp_df.columns:
            temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
        if 'DOB' in temp_df.columns:
            temp_df['DOB'] = pd.to_datetime(temp_df['DOB'], errors='coerce')
        
        cols_to_keep = []
        if 'RIMNO' in temp_df.columns: cols_to_keep.append('RIMNO')
        if 'DOB' in temp_df.columns: cols_to_keep.append('DOB')
        if cols_to_keep:
            all_pairs.append(temp_df[cols_to_keep])

    if all_pairs:
        global_pairs = (
            pd.concat(all_pairs, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        global_pairs['CUSTOMER_ID'] = range(1, len(global_pairs) + 1)

        logs.append(f"  Global unique (RIMNO, DOB) pairs: {len(global_pairs)}")
        logs.append(f"  CUSTOMER_ID range: 1 – {len(global_pairs)}")
    else:
        global_pairs = pd.DataFrame()
        logs.append("  Warning: No RIMNO/DOB columns found for mapping.")

    # ── PASS 2: Process each prime file ──
    logs.append("")
    logs.append("PASS 2: Processing each file with the global CUSTOMER_ID mapping...")

    for file in prime_files:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        logs.append("")
        logs.append(f"{'=' * 50}")
        logs.append(f"  Processing: {file}")
        logs.append(f"{'=' * 50}")

        # Need to handle potential missing columns in CSV
        df = pd.read_csv(
            file, encoding='latin',
            dtype=str
        ).rename(columns={'NAME': 'PRODUCT_NAME', 'RIM_NO': 'RIMNO', 'ï»¿BRANCH_ID': 'BRANCH_ID'})
        
        # Apply datetime parsing manually since parse_dates in read_csv fails if column is missing
        for c in config.PRIME_DATE_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')

        logs.append(f"  Rows loaded: {len(df)}")

        # Cast
        apply_cast_and_report(df, config.PRIME_STRING_COLS, 'string', logs)
        apply_cast_and_report(df, config.PRIME_FLOAT_COLS, 'float', logs)
        apply_cast_and_report(df, config.PRIME_INT_COLS, 'int', logs)
        apply_cast_and_report(df, config.PRIME_DATE_COLS, 'date', logs)

        # Clean STATUS columns
        if 'STATUS' in df.columns:
            df['STATUS'] = df['STATUS'].astype("string").str.strip().str.upper()
        if 'Card account status ' in df.columns:
            df['Card account status '] = df['Card account status '].astype("string").str.strip().str.upper()

        # Merge CUSTOMER_ID
        if 'RIMNO' in df.columns:
            df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
            
        if not global_pairs.empty and 'RIMNO' in df.columns and 'DOB' in df.columns:
            df = df.merge(global_pairs, on=['RIMNO', 'DOB'], how='left')
        elif 'CUSTOMER_ID' not in df.columns:
             # Fallback if no mapping possible
             df['CUSTOMER_ID'] = pd.NA

        assigned = df['CUSTOMER_ID'].notna().sum()
        unmatched = df['CUSTOMER_ID'].isna().sum()
        logs.append(f"  CUSTOMER_ID assigned: {assigned} rows")
        if unmatched > 0:
            logs.append(f"  WARNING: {unmatched} rows could not be assigned a CUSTOMER_ID.")

        # Split active / historical
        is_inactive_status = df['STATUS'].isin(config.INACTIVE_STATUSES) if 'STATUS' in df.columns else False
        is_inactive_card = df['Card account status '].isin(config.INACTIVE_STATUSES) if 'Card account status ' in df.columns else False
        is_historical = is_inactive_status & is_inactive_card

        historical_df = df[is_historical].copy()
        active_df = df[~is_historical].copy()

        logs.append(f"  Active rows:     {len(active_df)}")
        logs.append(f"  Historical rows: {len(historical_df)}")

        # Detect relatives: same (RIMNO, PRODUCT_NAME) but different CUSTOMER_ID
        if 'RIMNO' in active_df.columns and 'PRODUCT_NAME' in active_df.columns:
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
        else:
            relatives_df = pd.DataFrame()
            
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
    #  PRESTEP PART 2 — Transaction Cleaning
    # ====================================================================
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PRESTEP 2: Cleaning Raw Transaction Files")
    logs.append("=" * 60)

    # Build CUSTOMER_ID lookup from the active prime files we just created
    active_files = glob.glob(os.path.join(prime_output_dir, "*_active.csv"))
    if not active_files:
        logs.append(f"WARNING: No active prime files found in '{prime_output_dir}/'. Transaction prestep may fail to map IDs.")
        mapping_list = []
    else:
        logs.append(f"Building (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID lookup from {len(active_files)} active file(s)...")
        mapping_list = []
        for file in active_files:
            logs.append(f"  -> {file}")
            # we need to be careful if columns are missing
            headers = pd.read_csv(file, nrows=0).columns
            usecols = [c for c in ['RIMNO', 'PRODUCT_NAME', 'CUSTOMER_ID'] if c in headers]
            temp_df = pd.read_csv(
                file, encoding='latin',
                usecols=usecols,
                dtype='string'
            )
            if 'RIMNO' in temp_df.columns:
                temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
            if 'PRODUCT_NAME' in temp_df.columns:
                temp_df['PRODUCT_NAME'] = temp_df['PRODUCT_NAME'].str.strip()
            mapping_list.append(temp_df)

    if mapping_list:
        customer_lookup = pd.concat(mapping_list, ignore_index=True).drop_duplicates()
        if 'CUSTOMER_ID' in customer_lookup.columns:
            customer_lookup['CUSTOMER_ID'] = pd.to_numeric(
                customer_lookup['CUSTOMER_ID'], errors='coerce'
            ).astype('Int64')
        logs.append(f"  Unique mappings: {len(customer_lookup)}")
    else:
        customer_lookup = pd.DataFrame()

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
            dtype=str
        ).rename(columns={'DESCRIPTION': 'PRODUCT_NAME'})
        
        for c in config.TXN_DATE_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')

        logs.append(f"  Rows loaded: {len(df)}")

        apply_cast_and_report(df, config.TXN_STRING_COLS, 'string', logs)
        apply_cast_and_report(df, config.TXN_FLOAT_COLS, 'float', logs)
        apply_cast_and_report(df, config.TXN_INT_COLS, 'int', logs)
        apply_cast_and_report(df, config.TXN_DATE_COLS, 'date', logs)

        # Map CUSTOMER_ID
        if 'RIMNO' in df.columns:
            df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
        if 'PRODUCT_NAME' in df.columns:
            df['PRODUCT_NAME'] = df['PRODUCT_NAME'].astype("string").str.strip()
            
        if not customer_lookup.empty and 'RIMNO' in df.columns and 'PRODUCT_NAME' in df.columns:
            df = df.merge(customer_lookup, on=['RIMNO', 'PRODUCT_NAME'], how='left')
        elif 'CUSTOMER_ID' not in df.columns:
            df['CUSTOMER_ID'] = pd.NA

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


def load_active_prime_data(prime_dir, logs):
    """Loads consolidated active prime files."""
    logs.append("=" * 60)
    logs.append(" PHASE 1: Consolidating Active Prime Files")
    logs.append("=" * 60)

    prime_files = glob.glob(os.path.join(prime_dir, "*_active.csv"))
    if not prime_files:
        raise FileNotFoundError(f"No active prime files found in '{prime_dir}'.")

    prime_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in prime_files]
    prime_df = pd.concat(prime_dfs, ignore_index=True)
    logs.append(f"Loaded {len(prime_df)} total rows from {len(prime_files)} files.")

    apply_cast_and_report(prime_df, config.PRIME_STRING_COLS, 'string', logs)
    apply_cast_and_report(prime_df, config.PRIME_INT_COLS, 'int', logs)
    apply_cast_and_report(prime_df, config.PRIME_FLOAT_COLS, 'float', logs)
    apply_cast_and_report(prime_df, config.PRIME_DATE_COLS, 'date', logs)

    return prime_df


def load_matched_transaction_data(transaction_dir, logs):
    """Loads consolidated matched transaction files."""
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PHASE 2: Consolidating Matched Transaction Files")
    logs.append("=" * 60)

    all_txn_files = glob.glob(os.path.join(transaction_dir, "*.csv"))
    txn_files = [f for f in all_txn_files if not f.endswith("_missing_id.csv")]
    if not txn_files:
        logs.append(f"WARNING: No matched transaction files found in '{transaction_dir}'.")
        return None

    txn_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in txn_files]
    transaction_df = pd.concat(txn_dfs, ignore_index=True)
    logs.append(f"Loaded {len(transaction_df)} total rows from {len(txn_files)} files.")

    apply_cast_and_report(transaction_df, config.TXN_STRING_COLS, 'string', logs)
    apply_cast_and_report(transaction_df, config.TXN_INT_COLS, 'int', logs)
    apply_cast_and_report(transaction_df, config.TXN_FLOAT_COLS, 'float', logs)
    apply_cast_and_report(transaction_df, config.TXN_DATE_COLS, 'date', logs)

    return transaction_df
