"""
Churn Labeling Script
=====================
Processes monthly CSV files from JUL2025 to FEB2026.
Logic: If a RIMNO exists in JUL2025 but NOT in FEB2026 → churn = 1, else churn = 0

Output columns: RIMNO, Card_account_Status, churn
"""

import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# CONFIGURATION — update these to match your setup
# ─────────────────────────────────────────────

# Folder containing all monthly CSV files
DATA_FOLDER = "D:\G14\G14_Code\Credit_Risk\prm"  # Change to your folder path if needed

# Column names in your CSV files (update if different)
RIMNO_COL = "RIMNO"
STATUS_COL = "Card account status "

# File naming pattern — update to match your actual file names
# Examples: "data_JUL2025.csv", "JUL2025_data.csv", "2025-07.csv"
FILE_PATTERN = "*JUL2025*.csv"   # used only for auto-detection

# Explicit file paths (recommended — edit these to your actual filenames)
MONTHLY_FILES = {
    "202507": "202507.csv",
    "202602": "202602.csv",
}

OUTPUT_FILE = "churn_output.csv"

# ─────────────────────────────────────────────
# STEP 1: Load the two key months
# ─────────────────────────────────────────────

def load_month(filepath, label):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[{label}] File not found: {filepath}")
    df = pd.read_csv(filepath, usecols=[RIMNO_COL, STATUS_COL], dtype=str)
    df[RIMNO_COL] = df[RIMNO_COL].str.strip()
    df[STATUS_COL] = df[STATUS_COL].str.strip()
    df.dropna(subset=[RIMNO_COL], inplace=True)
    print(f"  [{label}] Loaded {len(df):,} rows | unique RIMNOs: {df[RIMNO_COL].nunique():,}")
    return df

print("=" * 55)
print("  Churn Labeling — JUL2025 → FEB2026")
print("=" * 55)

jul_path = os.path.join(DATA_FOLDER, MONTHLY_FILES["202507"])
feb_path = os.path.join(DATA_FOLDER, MONTHLY_FILES["202602"])

print("\nLoading monthly files...")
jul_df = load_month(jul_path, "JUL2025")
feb_df = load_month(feb_path, "FEB2026")

# ─────────────────────────────────────────────
# STEP 2: Build reference set from FEB2026
# ─────────────────────────────────────────────

feb_rimnos = set(feb_df[RIMNO_COL].unique())
print(f"\nFEB2026 unique RIMNOs (active set): {len(feb_rimnos):,}")

# ─────────────────────────────────────────────
# STEP 3: Label churn on JUL2025 base
# ─────────────────────────────────────────────

# Keep latest status per RIMNO in JUL2025 (in case of duplicates)
jul_deduped = jul_df.drop_duplicates(subset=[RIMNO_COL], keep="last").copy()

jul_deduped["churn"] = jul_deduped.apply(
    lambda row: 1 
    if (
        (row[RIMNO_COL] not in feb_rimnos) or (row[STATUS_COL] == "WROF")
    )
    else 0,
    axis=1
)

# ─────────────────────────────────────────────
# STEP 4: Summary
# ─────────────────────────────────────────────

total     = len(jul_deduped)
churned   = jul_deduped["churn"].sum()
retained  = total - churned
rate      = churned / total * 100 if total > 0 else 0

print("\n──────────────────────────────────────────────")
print("  CHURN SUMMARY")
print("──────────────────────────────────────────────")
print(f"  Total customers (JUL2025) : {total:>10,}")
print(f"  Churned (churn = 1)       : {churned:>10,}  ({rate:.2f}%)")
print(f"  Retained (churn = 0)      : {retained:>10,}  ({100-rate:.2f}%)")
print("──────────────────────────────────────────────")

# ─────────────────────────────────────────────
# STEP 5: Export output
# ─────────────────────────────────────────────

output = jul_deduped[[RIMNO_COL, STATUS_COL, "churn"]].reset_index(drop=True)
output.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Output saved → {OUTPUT_FILE}")
print(f"   Columns: {list(output.columns)}")
print(f"   Rows   : {len(output):,}")
print("\nPreview (first 10 rows):")
print(output.head(10).to_string(index=False))


files = sorted(glob.glob("trx/202*.csv")) 
dfs = []

for f in files:
    df = pd.read_csv(f)
    df["TRXN_DATE"] = pd.to_datetime(df["TRXN_DATE"], errors="coerce")
    dfs.append(df)

trans = pd.concat(dfs, ignore_index=True)
trans["REVERRSAL FLAG"] = trans["REVERRSAL FLAG"].map({
    "Norm": 1,
    "R": 0
    })


trans = trans.dropna(subset=["CUSTOMER_ID"])
trans["CUSTOMER_ID"] = trans["CUSTOMER_ID"].astype(str).str.strip()


reference_date = trans["TRXN_DATE"].max()
txn_features = trans.groupby("CUSTOMER_ID").agg(
    total_spend=("BILING AMT", "sum"),
    avg_spend=("BILLING AMT", "mean"),
    transaction_count=("BILLING AMT", "count")
).reset_index()


last_txn = trans.groupby("CUSTOMER_ID")["TRXN_DATE"].max().reset_index()

last_txn["recency_days"] = (reference_date - last_txn["TRXN_DATE"]).dt.days

txn_features = txn_features.merge(
    last_txn[["CUSTOMER_ID", "recency_days"]],
    on="CUSTOMER_ID",
    how="left")

reversal = trans.groupby("CUSTOMER_ID").agg(
    reversal_count=("REVERRSAL FLAG", "sum"),
    total_txn=("REVERRSAL FLAG", "count")
).reset_index()

reversal["reversal_ratio"] = reversal["reversal_count"] / reversal["total_txn"]

txn_features = txn_features.merge(
    reversal[["CUSTOMER_ID", "reversal_ratio"]],
    on="CUSTOMER_ID",
    how="left"
)

txn_features = txn_features.fillna(0)


print(txn_features.head())

files = sorted(glob.glob("prm/202*.csv"))

if not files:
    raise FileNotFoundError("No files found matching pattern: prm/202*.csv")

dfs = []

for f in files:
    print(f"Loading {f}...")
    df = pd.read_csv(f)

    dfs.append(df)

prime = pd.concat(dfs, ignore_index=True)

prime["CUSTOMER_ID"] = prime["CUSTOMER_ID"].astype(str)

prime["CREATION_DATE"] = pd.to_datetime(prime["CREATION_DATE"], errors="coerce")

prime_latest = prime.drop_duplicates(subset=["CUSTOMER_ID"], keep="last")

prime_features = prime_latest[[
    "CUSTOMER_ID",
    "CREDIT_LIMIT",
    "LEDGER_BALANCE",
    "AVILABLE_LIMIT",
    "OVERDUEAMOUNT",
    "STATUS",
    "STATUS_NAME",
    "ACTIVATED",
    "Card account status ",
    "CREATION_DATE",
    ]].copy()
reference_date = pd.to_datetime("2026-02-28")

prime_features["TENURE_DAYS"] = (
    reference_date - prime_features["CREATION_DATE"]
    ).dt.days

prime_features = prime_features.fillna(0)

prime_features = prime_features[
    prime_features["CUSTOMER_ID"].isin(txn_features["CUSTOMER_ID"])
    ]
print(prime_features.head())

final_df = txn_features.merge(prime_features, on="CUSTOMER_ID", how="left")

final_df = final_df.merge(
    output[["CUSTOMER_ID", "churn"]],
    on="CUSTOMER_ID",
    how="left"
)

final_df.to_csv("final_churn_dataset.csv", index=False)
print(final_df["churn"].value_counts(dropna=False))
final_df["churn"] = final_df["churn"].fillna(0)
print("----------------------------")


print(final_df["churn"].value_counts(dropna=False))

X = final_df.drop(columns=["CUSTOMER_ID", "churn"])
y = final_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)





models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
        ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
        ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
        ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        eval_metric="logloss",
        random_state=42
        ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )    
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}
THRESHOLD = 0.55

results = {}

rf_params = {
    "n_estimators": [200, 300],
    "max_depth": [8, 10, 15],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=rf_params,
    scoring="f1",
    cv=skf,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

xgb_params = {
    "n_estimators": [200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
}
xgb_grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    ),
    param_grid=xgb_params,
    scoring="f1",
    cv=skf,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

best_models = {
    "Random Forest": rf_grid.best_estimator_,
    "XGBoost": xgb_grid.best_estimator_
}

for name, model in best_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

results_df = pd.DataFrame(results).T
print(results_df.sort_values(by="roc_auc", ascending=False))
