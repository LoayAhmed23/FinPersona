"""
Central configuration for the Credit Risk Module.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRIME_DATA_DIR = os.path.join(DATA_DIR, "cleaned", "prime")
TRANSACTION_DATA_DIR = os.path.join(DATA_DIR, "cleaned", "transaction")

# Output 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(OUTPUT_DIR, "credit_default_model.joblib")
SCORES_PATH = os.path.join(OUTPUT_DIR, "credit_risk_scores.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "credit_evaluation_report.txt")

# Column mappings
CUSTOMER_ID = "CUSTOMER_ID"
TARGET_COL = "target"
STATUS_COL = "Card account status "
MONTH_COL = "snapshot_month"

TARGET_MODE = "weighted_binary"

SOFT_DEFAULT_STATUSES = ["30DD", "SUSP"]           # early default
HARD_DEFAULT_STATUSES = ["60DA", "90DA", "WROF"]   # severe default

DEFAULT_STATUSES = SOFT_DEFAULT_STATUSES + HARD_DEFAULT_STATUSES

# Everything else is non-default (target = 0)
NON_DEFAULT_STATUSES = [
    "NORM", "NEW", "CLSB", "CLSC", "CLSD",
    "LOST", "CNCD", "FRAD", "BLOK", "NOAU", "EXMU",
    "PICK", "BLCK", "STLC", "OFBL", "ONBL", "WARN",
    "NENC", "FREZ", "EXPD", "ACCA", "EXPC",
]

# Prime CSV dtype
PRIME_STRING_COLS = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME", "NAME",
    "GENDER", "CUSTOMER_TYPE", "Card account status ", "ORGANIZATION",
]
PRIME_INT_COLS = ["RIMNO", "CUSTOMER_ID", "BRANCH_ID"]
PRIME_FLOAT_COLS = [
    "AVAILABLE_LIMIT", "LEDGER_BALANCE", "LAST_PAYMENT_AMOUNT",
    "OVERDUEAMOUNT", "CREDIT_LIMIT",
]

# Transaction string cols
TXN_STRING_COLS = [
    "DESCRIPTION", "MERCHNAME", "MERCH ID", "SOURCES",
    "BANKBRANCH", "TRXN COUNTRY", "REVERSAL FLAG",
]

# Three leakage categories:
#   A) Direct label encodings
#   B) Temporal leakage
#   C) Raw amounts
DROP_COLS = [
    # --- Identifiers ---
    "RIMNO",
    "CUSTOMER_ID",
    "snapshot_month",
    "BRANCH_ID",
    "BRANCH_NAME",
    "ACTIVATED",

    # --- (A) Direct label encodings ---
    "STATUS",
    "STATUS_NAME",
    "DELINQUENCY",
    "Card account status ",

    # --- (B) Temporal leakage ---
    "CLOSURE_DATE",
    "CREATION_DATE",
    "LAST_STATEMENT_DATE",
    "DOB",

    "PRODUCT_NAME",
    "MERCHNAME",
    "source_file",

    "FIRST_REPLACED_CARD",
    "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD",

    "OVERDUEAMOUNT",
    "LEDGER_BALANCE",

    "sample_weight",
    "target",
    "CUSTOMER_TYPE",
]

# Leakage detection — single-feature AUC above this stops the training.
LEAKAGE_AUC_THRESHOLD = 0.95

# Correlation-based feature filtering — drop features whose 
# correlation with the target falls below this threshold.
CORR_THRESHOLD = 0.02

# Date columns (for parsing)
DATE_COLS_PRIME = [
    "CREATION_DATE", "LAST_STATEMENT_DATE", "CLOSURE_DATE", "DOB",
]
DATE_COLS_TXN = ["POST DATE", "TRXN DATE"]

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Default XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 5,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "seed": RANDOM_STATE,
    "max_bin": 64,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# SMOTE
SMOTE_ENABLED = True
# ratio of minority to majority after resampling
# 0.5 -> minority becomes 50% of majority count
SMOTE_SAMPLING_STRATEGY = 0.5   

# Hyperparameter tuning
TUNE_N_ITER = 10         
TUNE_CV_FOLDS = 5         
N_GPUS = 2                

# F-beta value for threshold tuning 
# Recall is more important than precision for default prediction
FBETA_VALUE = 1.5

# XGBoost hyperparameters for tuning — expanded search space
from scipy.stats import uniform, randint, loguniform

TUNE_PARAM_GRID = {
    # ---- Tree structure ----
    "max_depth":          randint(3, 12),            
    "min_child_weight":   randint(1, 30),            
    "gamma":              uniform(0, 10),             
    "max_leaves":         [0, 15, 31, 63, 127],      
    "grow_policy":        ["depthwise", "lossguide"], 

    # ---- Learning rate & boosting rounds ----
    "learning_rate":      loguniform(0.003, 0.3),    
    "n_estimators":       randint(100, 2000),         

    # ---- Sampling / regularization ----
    "colsample_bytree":   uniform(0.2, 0.8),         
    "colsample_bylevel":  uniform(0.3, 0.7),         
    "colsample_bynode":   uniform(0.5, 0.5),         
    "subsample":          uniform(0.4, 0.6),         

    # ---- L1 / L2 regularization ----
    "reg_alpha":          loguniform(1e-4, 10),       
    "reg_lambda":         loguniform(1e-4, 10),       

    # ---- Histogram bins ----
    "max_bin":            [32, 64, 128, 256, 512],   
}

# Temporal feature engineering
TREND_COLS = [
    "utilization_ratio", "overdue_ratio",
    "financial_stress_score", "available_credit_ratio", "over_limit_ratio",
]
TXN_TREND_COLS = [
    "txn_count", "txn_total_amount", "txn_mean_amount",
]
