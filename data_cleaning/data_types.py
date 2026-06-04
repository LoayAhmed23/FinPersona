import pandas as pd 
import numpy as np
from datetime import datetime
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
import glob
import pandas as pd
    
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
        
        # Print the results, adding the percentage formatted to 2 decimal places
        print(f"[{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)")


def drop_empty_records(df, columns):
    """
    Drops rows from a dataframe if they are missing data in the specified columns.
    Prints a summary of how many rows were removed.
    """
    # Safeguard: if a single string is passed instead of a list, convert it
    if isinstance(columns, str):
        columns = [columns]
        
    # Filter the list to only include columns that actually exist in the dataframe
    valid_cols = [col for col in columns if col in df.columns]
    
    if not valid_cols:
        print(f"\nWarning: None of the specified columns {columns} exist in the dataframe. No rows dropped.")
        return df
        
    # Count rows before dropping
    rows_before = len(df)
    
    # Drop the empty records
    cleaned_df = df.dropna(subset=valid_cols, how='any')
    
    # Report the results
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
    # 1. Count missing values per column
    missing_counts = df.isna().sum()
    
    # 2. Filter out columns that have 0 missing values
    missing_counts = missing_counts[missing_counts > 0]
    
    # 3. Calculate the percentage of missing data
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100
    
    # 4. Create a clean summary DataFrame
    summary_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Percentage (%)': missing_percentages.round(2)
    })
    
    # 5. Sort from highest number of missing values to lowest
    summary_df = summary_df.sort_values(by='Missing Count', ascending=False)
    
    # Check if the dataframe had no missing values at all
    if summary_df.empty:
        print("Great news! There are no missing values in this DataFrame.")
        return None
        
    return summary_df


prime_string_cols = ["BRANCH_NAME","ACTIVATED","STATUS","STATUS_NAME","PRODUCT_NAME","GENDER","CUSTOMER_TYPE","Card account status "]
prime_int_cols = ["BRANCH_ID","RIMNO"]
prime_float_cols = ["CREDIT_LIMIT","DELIQUENCY","LEDGER_BALANCE","AVAILABLE_LIMIT","OVERDUEAMOUNT","FIRST_REPLACED_CARD","SECOND_REPLACED_CARD","THIRD_REPLACED_CARD","SETTLEMENT AMT"]
prime_date_cols = ["CREATION_DATE","LAST_STAEMENT_DATE","LAST_PAYMENT_DATE","DOB","CLOSURE_DATE"]

# ========================= 1. Load Data ==========================
prime_files = glob.glob("prime/*.csv")

print("Loading CSV files...")
for file in prime_files:
    temp_df = pd.read_csv(
        file, 
        encoding='latin', 
    ).rename(columns={'RIM_NO': 'RIMNO', "NAME": "PRODUCT_NAME"})
    
    
    # ========================= 2. Casting & Reporting ==========================
    print("\n--- Casting Columns and Checking Nulls ---")


    print("\n-> String Columns:")
    apply_cast_and_report(prime_df, prime_string_cols, 'string')

    print("\n-> Float Columns:")
    apply_cast_and_report(prime_df, prime_float_cols, 'float')

    print("\n-> Integer Columns:")
    apply_cast_and_report(prime_df, prime_int_cols, 'int')

    print("\n-> Date Columns:")
    apply_cast_and_report(prime_df, prime_date_cols, 'date')

    # ========================= 3. Cleanup & Final Info ==========================
    prime_columns_to_drop = ["MAPPING_ACCNO", "MIN_PAYMENT", "OVER_LIMIT", "TOTAL_HOLD" ,"ORGANIZATION","JOINING_FEE","ANNUAL_FEE","LAST_PAYMENT_AMOUNT", "LAST_PAYMENT_DATE", "SETTLEMENT AMT"]
    existing_cols_to_drop = [col for col in prime_columns_to_drop if col in prime_df.columns]
    prime_df = prime_df.drop(columns=existing_cols_to_drop)

    print("\n--- Final Dataframe Info ---")
    print(prime_df.info())
    
    critical_columns = ["RIMNO"]
    prime_df = drop_empty_records(prime_df, critical_columns)

    # ==handling none values
    prime_df["GENDER"] = prime_df["GENDER"].fillna("Unkown")
    prime_df["BRANCH_NAME"] = prime_df["BRANCH_NAME"].fillna("Unkown")
    prime_df["BRANCH_ID"] = prime_df["BRANCH_ID"].fillna(-1)

    prime_df['DOB_WAS_MISSING'] = prime_df['DOB'].isna().astype(int)

    if not prime_df["DOB"].dropna().empty:
        median_dob_int = prime_df['DOB'].dropna().astype('int64').median()
        median_dob = pd.to_datetime(median_dob_int)
        prime_df['DOB'] = prime_df['DOB'].fillna(median_dob)

    prime_df["LAST_PAYMENT_DATE"] = prime_df["LAST_PAYMENT_DATE"].fillna(prime_df['CREATION_DATE'])

    # Save cleaned file
    prime_df.to_csv(f"new_prime_data_{file[-12:-4]}.csv", index=False)


    