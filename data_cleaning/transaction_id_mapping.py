import glob
import os

import pandas as pd

# ========================= Default Project Paths =========================
# Resolve project root from this script's location (data_cleaning/ -> project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_PRIME_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "prime_cleaned")
DEFAULT_TRANSACTION_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "transaction")
DEFAULT_TRANSACTION_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "transaction_cleaned"
)


# ========================= Helper Functions (from main.py) =========================
def apply_cast_and_report(df, columns, cast_type):
    # Get the total number of rows for our percentage math
    total_rows = len(df)

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue

        # Count nulls before
        nulls_before = df[col].isna().sum()

        # Apply the specific vectorized casting logic
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

        # Count nulls after
        nulls_after = df[col].isna().sum()

        # Calculate percentage of missing data
        pct_missing = (nulls_after / total_rows) * 100

        # Print the results, adding the percentage formatted to 2 decimal places
        print(
            f"[{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)"
        )


# ========================= Transaction Column Definitions (from main.py) =========================
transaction_string_cols = [
    "MERCHNAME",
    "MERCH ID",
    "SOURCES",
    "BANKBRANCH",
    "TRXN COUNTRY",
    "REVERSAL FLAG",
    "PRODUCT_NAME",
]
transaction_int_cols = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY"]
transaction_float_cols = [
    "ORIG AMOUNT",
    "EMBEDDED _FEE",
    "BILLING AMT",
    "SETTLEMENT AMT",
]
transaction_date_cols = ["TRXN DATE", "POST DATE"]


# ========================= Main Processing Function =========================
def run(
    prime_cleaned_dir=None, transaction_input_dir=None, transaction_output_dir=None
):
    """
    Run the transaction ID mapping pipeline.

    Parameters
    ----------
    prime_cleaned_dir : str or None
        Directory containing the cleaned active prime CSV files
        (output of ``prime_id_creation.run``).
        Defaults to ``<PROJECT_ROOT>/data/prime_cleaned``.
    transaction_input_dir : str or None
        Directory containing raw transaction Excel files.
        Defaults to ``<PROJECT_ROOT>/data/transaction``.
    transaction_output_dir : str or None
        Directory where cleaned transaction CSV files will be saved.
        Defaults to ``<PROJECT_ROOT>/data/transaction_cleaned``.
    """
    prime_cleaned_dir = prime_cleaned_dir or DEFAULT_PRIME_CLEANED_DIR
    transaction_input_dir = transaction_input_dir or DEFAULT_TRANSACTION_INPUT_DIR
    transaction_output_dir = transaction_output_dir or DEFAULT_TRANSACTION_OUTPUT_DIR
    transaction_missing_dir = os.path.join(transaction_output_dir, "missing")

    # ========================= Build CUSTOMER_ID Lookup from Active Prime Files =========================
    print(
        "Building (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID lookup from active prime files...\n"
    )

    active_files = glob.glob(os.path.join(prime_cleaned_dir, "*_active.csv"))

    if not active_files:
        print(
            f"ERROR: No active files found in '{prime_cleaned_dir}/'. Run prime_id_creation.run() first."
        )
        return

    print(f"Found {len(active_files)} active file(s):")

    mapping_list = []
    for file in active_files:
        print(f"  -> {file}")
        temp_df = pd.read_csv(
            file,
            encoding="latin",
            usecols=["RIMNO", "PRODUCT_NAME", "CUSTOMER_ID"],
            dtype="string",
        )
        temp_df["RIMNO"] = temp_df["RIMNO"].str.strip()
        temp_df["PRODUCT_NAME"] = temp_df["PRODUCT_NAME"].str.strip()
        mapping_list.append(temp_df)

    customer_lookup = pd.concat(mapping_list, ignore_index=True).drop_duplicates()

    # Cast CUSTOMER_ID back to numeric
    customer_lookup["CUSTOMER_ID"] = pd.to_numeric(
        customer_lookup["CUSTOMER_ID"], errors="coerce"
    ).astype("Int64")

    print(
        f"\n  Unique (RIMNO, PRODUCT_NAME) -> CUSTOMER_ID mappings: {len(customer_lookup)}"
    )

    # ========================= Find All Transaction Files =========================
    os.makedirs(transaction_output_dir, exist_ok=True)
    os.makedirs(transaction_missing_dir, exist_ok=True)

    transaction_files = glob.glob(os.path.join(transaction_input_dir, "*.xlsx"))

    if not transaction_files:
        print(f"\nERROR: No Excel files found in '{transaction_input_dir}'.")
        return

    print(f"\nFound {len(transaction_files)} transaction file(s).\n")

    # ========================= Process Each Transaction File =========================
    for file in transaction_files:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        print(f"{'='*60}")
        print(f"Processing: {file}")
        print(f"{'='*60}")

        # --- Load ---
        df = pd.read_excel(
            file,
            dtype={
                col: "string"
                for col in transaction_string_cols
                + transaction_int_cols
                + transaction_float_cols
            },
            parse_dates=transaction_date_cols,
        ).rename(columns={"DESCRIPTION": "PRODUCT_NAME"})

        print(f"  Rows loaded: {len(df)}")

        # --- Cast all columns using helper function ---
        print("\n  -> String Columns:")
        apply_cast_and_report(df, transaction_string_cols, "string")

        print("\n  -> Float Columns:")
        apply_cast_and_report(df, transaction_float_cols, "float")

        print("\n  -> Integer Columns:")
        apply_cast_and_report(df, transaction_int_cols, "int")

        print("\n  -> Date Columns:")
        apply_cast_and_report(df, transaction_date_cols, "date")

        # --- Map CUSTOMER_ID using (RIMNO, PRODUCT_NAME) ---
        df["RIMNO"] = df["RIMNO"].astype("string").str.strip()
        df["PRODUCT_NAME"] = df["PRODUCT_NAME"].astype("string").str.strip()

        df = df.merge(customer_lookup, on=["RIMNO", "PRODUCT_NAME"], how="left")

        matched_count = df["CUSTOMER_ID"].notna().sum()
        unmatched_count = df["CUSTOMER_ID"].isna().sum()
        print(f"\n  CUSTOMER_ID mapped: {matched_count} rows")
        if unmatched_count > 0:
            print(f"  Missing CUSTOMER_ID: {unmatched_count} rows")

        # --- Split into matched & unmatched ---
        matched_df = df[df["CUSTOMER_ID"].notna()].copy()
        missing_df = df[df["CUSTOMER_ID"].isna()].copy()

        # --- Save matched ---
        output_path = os.path.join(transaction_output_dir, f"{file_basename}.csv")
        matched_df.to_csv(output_path, index=False)
        print(f"  Saved -> {output_path}")

        # --- Save unmatched ---
        if len(missing_df) > 0:
            missing_path = os.path.join(
                transaction_missing_dir, f"{file_basename}_missing_id.csv"
            )
            missing_df.to_csv(missing_path, index=False)
            print(f"  Saved -> {missing_path}")

        print()

    print("All transaction files processed!")


# ========================= Run When Executed Directly =========================
if __name__ == "__main__":
    run()
