"""
Central configuration for the Churn Prediction System.
All paths, column names, hyperparameters, and constants live here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRIME_DATA_DIR = os.path.join(DATA_DIR, "cleaned", "prime")
TRANSACTION_DATA_DIR = os.path.join(DATA_DIR, "cleaned", "transaction")

# Output / model artefacts
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")
SCORES_PATH = os.path.join(OUTPUT_DIR, "churn_scores.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "churn_evaluation_report.txt")

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
CUSTOMER_ID = "CUSTOMER_ID"
TARGET_COL = "churn"
STATUS_COL = "Card account status "

# ---------------------------------------------------------------------------
# Churn labeling
# ---------------------------------------------------------------------------
# A customer is labeled churn=1 if they exist in the reference month
# but are absent in the target month, or have a WROF status.
CHURN_REFERENCE_MONTH = "202507"   # JUL 2025 — base cohort
CHURN_TARGET_MONTH = "202602"      # FEB 2026 — check presence
CHURN_DEFAULT_STATUSES = ["WROF"]  # statuses that count as churn regardless

CHURN_LABEL_FILES = {
    "202507": "202507.csv",
    "202602": "202602.csv",
}

# ---------------------------------------------------------------------------
# File patterns
# ---------------------------------------------------------------------------
PRIME_FILE_PATTERN = "*.csv"
TXN_FILE_PATTERN = "*.csv"

# ---------------------------------------------------------------------------
# Transaction column names (preserved from raw data, including typos)
# ---------------------------------------------------------------------------
TXN_AMOUNT_COL = "BILLING AMT"
TXN_DATE_COL = "TRXN_DATE"
TXN_REVERSAL_COL = "REVERRSAL FLAG"
TXN_REVERSAL_MAP = {"Norm": 1, "R": 0}

# ---------------------------------------------------------------------------
# Prime feature columns to keep
# ---------------------------------------------------------------------------
PRIME_FEATURE_COLS = [
    "CUSTOMER_ID", "CREDIT_LIMIT", "LEDGER_BALANCE", "AVILABLE_LIMIT",
    "OVERDUEAMOUNT", "STATUS", "STATUS_NAME", "ACTIVATED",
    "Card account status ", "CREATION_DATE",
]

# Categorical columns to label-encode
CATEGORICAL_COLS = ["STATUS", "STATUS_NAME", "ACTIVATED", "Card account status "]

# Columns to drop before modeling (identifiers / dates)
DROP_COLS = ["CUSTOMER_ID", "CREATION_DATE", "churn"]

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
REFERENCE_DATE = "2026-02-28"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
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
