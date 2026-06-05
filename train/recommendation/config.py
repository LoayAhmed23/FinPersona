"""
config.py
===========
Central configuration for the Recommendation Pipeline.
Stores column definitions, constants, and hyperparameters.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# Output files
PREPROCESSED_CSV = os.path.join(SAVE_DIR, "final_customer_profile.csv")
XGB_MODELS_PKL = os.path.join(SAVE_DIR, "xgb_models.pkl")
XGB_META_JSON = os.path.join(SAVE_DIR, "xgb_meta.json")
CBF_SIM_PKL = os.path.join(SAVE_DIR, "cbf_sim_matrix.pkl")
CBF_META_JSON = os.path.join(SAVE_DIR, "cbf_meta.json")
BATCH_OUTPUT_PATH = os.path.join(SAVE_DIR, "batch_predictions.csv")

# ---------------------------------------------------------------------------
# Column Definitions (Prime)
# ---------------------------------------------------------------------------
PRIME_STRING_COLS = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "
]
PRIME_INT_COLS = ["BRANCH_ID", "RIMNO", "CUSTOMER_ID"]
PRIME_FLOAT_COLS = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD"
]
PRIME_DATE_COLS = ["CREATION_DATE", "LAST_STATEMENT_DATE", "DOB", "CLOSURE_DATE", "LAST_PAYMENT_DATE"]

# ---------------------------------------------------------------------------
# Column Definitions (Transaction)
# ---------------------------------------------------------------------------
TXN_STRING_COLS = [
    "MERCHNAME", "MERCH ID", "SOURCES", "BANKBRANCH",
    "TRXN COUNTRY", "REVERSAL FLAG", "PRODUCT_NAME", "DESCRIPTION"
]
TXN_INT_COLS = ["RIMNO", "CCY", "MCC", "SETTLEMENT CCY", "CUSTOMER_ID"]
TXN_FLOAT_COLS = ["ORIG AMOUNT", "EMBEDDED _FEE", "BILLING AMT", "SETTLEMENT AMT"]
TXN_DATE_COLS = ["TRXN DATE", "POST DATE"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INACTIVE_STATUSES = [
    'CLSB', 'CLSC', 'LOST', 'WROF', 'CNCD', 'SUSP', 'FRAD', 'CLSD',
    'BLOK', 'NOAU', 'EXMU', 'PICK', 'BLCK', 'STLC', 'OFBL', 'ONBL',
    'FREZ', 'EXPD', 'EXPC', 'CLSS',
]

PRIME_COLUMNS_TO_DROP = [
    "MAPPING_ACCNO", "STATUS", "CREATION_DATE", "MIN_PAYMENT", "OVER_LIMIT",
    "ACTIVATED", "DELIQUENCY", "STATUS_NAME", "OVER_LIMIT", "TOTAL_HOLD",
    "ORGANIZATION", "JOINING_FEE", "ANNUAL_FEE", "LAST_PAYMENT_AMOUNT",
    "LAST_PAYMENT_DATE", "SETTLEMENT AMT", 'FIRST_REPLACED_CARD',
    'SECOND_REPLACED_CARD', 'THIRD_REPLACED_CARD', 'LAST_STATEMENT_DATE',
    "LEDGER_BALANCE", "AVAILABLE_LIMIT", "CLOSURE_DATE",
    "Card account status ", "CUSTOMER_TYPE", "OVERDUEAMOUNT"
]

TRANSACTION_COLUMNS_TO_DROP = [
    "POST DATE", "ORIG AMOUNT", "EMBEDDED _FEE", "SETTLEMENT AMT", 
    "SETTLEMENT CCY", "SOURCES"
]

CORR_THRESHOLD = 0.1
MIN_SAMPLES_REQUIRED = 50
