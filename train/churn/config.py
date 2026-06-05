"""
All Parameters used in Churn Module 
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRIME_DATA_DIR = os.path.join(DATA_DIR, "original", "prime")
TRANSACTION_DATA_DIR = os.path.join(DATA_DIR, "cleaned", "transaction")

# Outputs
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")
SCORES_PATH = os.path.join(OUTPUT_DIR, "churn_scores.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "churn_evaluation_report.txt")

# Columns
CUSTOMER_ID = "CUSTOMER_ID"         
PRIME_CUSTOMER_ID_RAW = "RIM_NO"    
TXN_CUSTOMER_ID_RAW = "RIMNO"       
TARGET_COL = "CHURN"
STATUS_COL = "Card account status "

# Churn labeling
CHURN_REFERENCE_MONTH = "202602"   # FEB 2026 — base file
CHURN_TARGET_MONTH = "202605"      # MAY 2026 — target file
CHURN_DEFAULT_STATUSES = ["WROF"]  # statuses that count as churn regardless

CHURN_LABEL_FILES = {
    "202602": "FEB2026.csv",
    "202605": "MAY2026.csv",
}


# Transaction columns
TXN_AMOUNT_COL = "BILLING AMT"
TXN_DATE_COL = "TRXN DATE"
TXN_REVERSAL_COL = "REVERSAL FLAG"
TXN_REVERSAL_MAP = {"N": 1, "R": 0}

# Prime feature columns 
PRIME_FEATURE_COLS = [
    "CUSTOMER_ID", "CREDIT_LIMIT", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "STATUS", "STATUES_NAME", "ACTIVATED",
    "Card account status ", "CREATION_DATE",
]

# Categorical columns to label-encode
CATEGORICAL_COLS = ["STATUS", "STATUES_NAME", "ACTIVATED", "Card account status "]

# Columns to drop before modeling 
DROP_COLS = ["CUSTOMER_ID", "CREATION_DATE", "churn"]


# R
REFERENCE_DATE = "2026-05-31"


# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.55
CV_FOLDS = 5
CV_SCORING = "f1"

# GridSearch parameter grids
RF_GRID_PARAMS = {
    "n_estimators": [200, 300],
    "max_depth": [8, 10, 15],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],
}

XGB_GRID_PARAMS = {
    "n_estimators": [200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
}
