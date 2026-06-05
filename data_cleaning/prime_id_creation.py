import pandas as pd
import numpy as np
import glob
import os

# ========================= Default Project Paths =========================
# Resolve project root from this script's location (data_cleaning/ -> project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "prime")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "prime_cleaned")


# ========================= Helper Functions =========================
def apply_cast_and_report(df, columns, cast_type):
    """Cast columns to the specified type and report null changes."""
    total_rows = len(df)

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue

        # Count nulls before
        nulls_before = df[col].isna().sum()

        # Apply the specific vectorized casting logic
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

        # Count nulls after
        nulls_after = df[col].isna().sum()

        # Calculate percentage of missing data
        pct_missing = (nulls_after / total_rows) * 100

        print(f"[{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)")


def drop_empty_records(df, columns):
    """
    Drops rows from a dataframe if they are missing data in the specified columns.
    Prints a summary of how many rows were removed.
    """
    if isinstance(columns, str):
        columns = [columns]

    valid_cols = [col for col in columns if col in df.columns]

    if not valid_cols:
        print(f"\nWarning: None of the specified columns {columns} exist in the dataframe. No rows dropped.")
        return df

    rows_before = len(df)
    cleaned_df = df.dropna(subset=valid_cols, how='any')
    rows_after = len(cleaned_df)
    rows_dropped = rows_before - rows_after

    print(f"\n--- Dropping Empty Records ---")
    print(f"Checked columns: {valid_cols}")
    print(f"Dropped {rows_dropped} rows.")
    print(f"Total rows remaining: {rows_after}")

    return cleaned_df


def check_missing_values(df):
    """
    Analyzes a DataFrame for missing values and returns a summary table
    containing only the columns that have missing data.
    """
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100

    summary_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Percentage (%)': missing_percentages.round(2)
    })
    summary_df = summary_df.sort_values(by='Missing Count', ascending=False)

    if summary_df.empty:
        print("Great news! There are no missing values in this DataFrame.")
        return None

    return summary_df


# ========================= Column Definitions (union of both files) =========================
prime_string_cols = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "
]
prime_int_cols = ["BRANCH_ID", "RIMNO"]
prime_float_cols = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD", "SETTLEMENT AMT"
]
prime_date_cols = [
    "CREATION_DATE", "LAST_STATEMENT_DATE", "LAST_STATEMENT_DATE",
    "LAST_PAYMENT_DATE", "DOB", "CLOSURE_DATE"
]

# Columns to drop after cleaning
prime_columns_to_drop = [
    "MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD",
    "ORGANIZATION", "JOINING_FEE", "ANNUAL_FEE", "LAST_PAYMENT_AMOUNT",
    "LAST_PAYMENT_DATE", "SETTLEMENT AMT"
]

# Status codes that indicate an inactive / closed / blocked card
inactive_statuses = [
    'CLSB',   # closed by bank
    'CLSC',   # closed by customer
    'LOST',   # lost card
    'WROF',   # write off status
    'CNCD',   # card not collected by DSU
    'SUSP',   # profit suspended
    'FRAD',   # pick up card special fraud
    'CLSD',   # closed
    'BLOK',   # blok card
    'NOAU',   # no authorization
    'EXMU',   # expired murabha
    'PICK',   # pick up card
    'BLCK',   # blocked status
    'STLC',   # stolen card
    'OFBL',   # offline pin block
    'ONBL',   # online pin block
    'FREZ',   # freeze card
    'EXPD',   # expired
    'EXPC',   # expired card
    'CLSS'
]


# ========================= Main Processing Function =========================
def run(input_dir=None, output_dir=None):
    """
    Run the prime ID creation pipeline.

    Parameters
    ----------
    input_dir : str or None
        Directory containing raw prime CSV files.
        Defaults to ``<PROJECT_ROOT>/data/prime``.
    output_dir : str or None
        Directory where cleaned CSV files will be saved.
        Defaults to ``<PROJECT_ROOT>/data/prime_cleaned``.
    """
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # ========================= Find All Prime Files =========================
    prime_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not prime_files:
        print(f"ERROR: No CSV files found in '{input_dir}'. Make sure the folder exists.")
        return

    print(f"Found {len(prime_files)} prime file(s).\n")

    # ========================= PASS 1: Build Global CUSTOMER_ID Mapping =========================
    print("PASS 1: Scanning all files to build a global CUSTOMER_ID mapping...\n")

    all_pairs = []

    for file in prime_files:
        print(f"  Scanning: {file}")
        temp_df = pd.read_csv(
            file,
            encoding='latin',
            usecols=['RIM_NO', 'DOB'],
            dtype='string'
        )

        temp_df = temp_df.rename(columns={'RIM_NO': 'RIMNO'})

        temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
        temp_df['DOB'] = pd.to_datetime(temp_df['DOB'], errors='coerce')

        all_pairs.append(temp_df[['RIMNO', 'DOB']])

    # Combine and deduplicate across ALL files
    global_pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    global_pairs['CUSTOMER_ID'] = range(1, len(global_pairs) + 1)

    print(f"\n  Global unique (RIMNO, DOB) pairs: {len(global_pairs)}")
    print(f"  CUSTOMER_ID range: 1 - {len(global_pairs)}\n")

    # ========================= PASS 2: Process Each File =========================
    print("PASS 2: Processing each file — cast, clean, assign IDs, split, save...\n")

    for file in prime_files:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        print(f"{'='*60}")
        print(f"Processing: {file}")
        print(f"{'='*60}")

        # --- Load ---
        df = pd.read_csv(
            file,
            encoding='latin',
            dtype={col: "string" for col in prime_string_cols + prime_int_cols + prime_float_cols},
            parse_dates=prime_date_cols
        ).rename(columns={'RIM_NO': 'RIMNO', 'NAME': 'PRODUCT_NAME'})

        print(f"  Rows loaded: {len(df)}")

        # --- Cast all columns ---
        print("\n  -> String Columns:")
        apply_cast_and_report(df, prime_string_cols, 'string')

        print("\n  -> Float Columns:")
        apply_cast_and_report(df, prime_float_cols, 'float')

        print("\n  -> Integer Columns:")
        apply_cast_and_report(df, prime_int_cols, 'int')

        print("\n  -> Date Columns:")
        apply_cast_and_report(df, prime_date_cols, 'date')

        # --- Drop unnecessary columns ---
        existing_cols_to_drop = [col for col in prime_columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        print("\n--- Final Dataframe Info ---")
        print(df.info())

        # --- Drop rows missing critical columns ---
        critical_columns = ["RIMNO"]
        df = drop_empty_records(df, critical_columns)

        # --- Handle missing values ---
        df["GENDER"] = df["GENDER"].fillna("Unknown")
        df["BRANCH_NAME"] = df["BRANCH_NAME"].fillna("Unknown")
        df["BRANCH_ID"] = df["BRANCH_ID"].fillna(-1)

        df['DOB_WAS_MISSING'] = df['DOB'].isna().astype(int)

        if not df["DOB"].dropna().empty:
            median_dob_int = df['DOB'].dropna().astype('int64').median()
            median_dob = pd.to_datetime(median_dob_int)
            df['DOB'] = df['DOB'].fillna(median_dob)

        # --- Missing values summary ---
        print("\n--- Missing Values Summary ---")
        missing_summary = check_missing_values(df)
        if missing_summary is not None:
            print(missing_summary)

        # --- Clean STATUS columns for splitting ---
        df['STATUS'] = df['STATUS'].astype("string").str.strip().str.upper()
        df['Card account status '] = df['Card account status '].astype("string").str.strip().str.upper()

        # --- Assign CUSTOMER_ID from the global mapping ---
        df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
        df = df.merge(global_pairs, on=['RIMNO', 'DOB'], how='left')

        assigned = df['CUSTOMER_ID'].notna().sum()
        unmatched = df['CUSTOMER_ID'].isna().sum()
        print(f"\n  CUSTOMER_ID assigned: {assigned} rows")
        if unmatched > 0:
            print(f"  WARNING: {unmatched} rows could not be assigned a CUSTOMER_ID (missing RIMNO or DOB).")

        # --- Split into Active & Historical ---
        is_inactive_status = df['STATUS'].isin(inactive_statuses)
        is_inactive_card = df['Card account status '].isin(inactive_statuses)

        is_historical = is_inactive_status & is_inactive_card

        historical_df = df[is_historical].copy()
        active_df = df[~is_historical].copy()

        print(f"  Active   (active STATUS):     {len(active_df)} rows")
        print(f"  Historical (inactive STATUS): {len(historical_df)} rows")

        # --- Detect relatives: same (RIMNO, PRODUCT_NAME) but different CUSTOMER_ID ---
        dup_check = active_df.groupby(['RIMNO', 'PRODUCT_NAME'])['CUSTOMER_ID'].nunique()
        dup_groups = dup_check[dup_check > 1].index

        if len(dup_groups) > 0:
            print(f"\n  Found {len(dup_groups)} (RIMNO, PRODUCT_NAME) pair(s) with multiple CUSTOMER_IDs.")

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

            print(f"  Moved {len(relatives_df)} row(s) to active_relatives.")
            print(f"  Active after dedup: {len(active_df)} rows")
        else:
            relatives_df = pd.DataFrame()
            print(f"\n  No (RIMNO, PRODUCT_NAME) duplicates found — all CUSTOMER_IDs are unique per combo.")

        # --- Save ---
        active_path = os.path.join(output_dir, f"{file_basename}_active.csv")
        historical_path = os.path.join(output_dir, f"{file_basename}_historical.csv")

        active_df.to_csv(active_path, index=False)
        historical_df.to_csv(historical_path, index=False)

        print(f"  Saved -> {active_path}")
        print(f"  Saved -> {historical_path}")

        if len(relatives_df) > 0:
            relatives_path = os.path.join(output_dir, f"{file_basename}_active_relatives.csv")
            relatives_df.to_csv(relatives_path, index=False)
            print(f"  Saved -> {relatives_path}")

        print()

    print("All files processed!")


# ========================= Run When Executed Directly =========================
if __name__ == "__main__":
    run()
