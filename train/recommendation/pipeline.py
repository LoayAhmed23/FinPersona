import os
import glob
import numpy as np
import pandas as pd
import config
from data_loader import run_prestep, load_active_prime_data, load_matched_transaction_data, apply_cast_and_report
from feature_engineering import (
    build_user_item_matrix, build_rfm_features, build_mcc_spend,
    build_foreign_trxn_features, build_demographics_features, merge_all_features
)
from preprocessing import preprocess_pipeline
from model import train_xgboost, predict_for_customer, train_cbf, predict_cbf_for_customer

def run_PREPROCESSING_pipeline(prime_dir, transaction_dir=None,
                           raw_prime_dir=None, raw_transaction_dir=None, logs=None):
    """
    Runs the full Preprocessing pipeline from the given directory.

    If *raw_prime_dir* and *raw_transaction_dir* are supplied the prestep
    (CUSTOMER_ID creation, active/historical splitting, transaction mapping)
    is executed first, writing cleaned files into *prime_dir* and
    *transaction_dir* respectively.

    Returns (final_customer_profile DataFrame, logs list, product_cols list, feature_cols list).
    """
    if logs is None:
        logs = []

    if transaction_dir is None:
        transaction_dir = "transaction_cleaned"

    # ── Optional prestep ──
    if raw_prime_dir and raw_transaction_dir:
        logs.append("Running data-cleaning prestep...")
        logs.append("")
        run_prestep(
            raw_prime_dir=raw_prime_dir,
            raw_transaction_dir=raw_transaction_dir,
            prime_output_dir=prime_dir,
            transaction_output_dir=transaction_dir,
            logs=logs
        )
        logs.append("")

    # Phase 1: Consolidate Active Prime
    prime_df = load_active_prime_data(prime_dir, logs)

    # Phase 2: Consolidate Transactions
    transaction_df = load_matched_transaction_data(transaction_dir, logs)

    # User-item matrix
    prime_df, _ = build_user_item_matrix(prime_df, logs)

    # RFM Features
    rfm_features = build_rfm_features(transaction_df)

    # MCC Spend
    mcc_spend = build_mcc_spend(transaction_df)

    # Foreign Transactions
    foreign_agg = build_foreign_trxn_features(transaction_df)

    # Demographics
    prime_df = build_demographics_features(prime_df)

    # Final Merge
    profile = merge_all_features(prime_df, rfm_features, mcc_spend, foreign_agg, logs)

    # Preprocessing Pipeline (clean, drop, encode, filter)
    final_customer_profile, final_products, final_features = preprocess_pipeline(profile, logs)

    return final_customer_profile, logs, final_products, final_features


def predict_new_data(raw_prime_dir, raw_transaction_dir,
                     models_dict, optimal_thresholds,
                     trained_feature_cols, valid_targets,
                     sim_matrix=None, cbf_product_cols=None, cbf_thresholds=None,
                     logs=None, progress_callback=None):
    """
    Processes new raw prime/transaction data, engineers the same features used
    during training, and predicts products for every customer using the
    already-trained XGBoost (and optionally CBF) models.
    """
    if logs is None:
        logs = []
    if progress_callback is None:
        progress_callback = lambda pct: None

    logs.append("=" * 60)
    logs.append(" BATCH PREDICTION: Processing New Customer Data")
    logs.append("=" * 60)

    # ── Temp output dirs for the prestep ──
    prime_output = os.path.join(config.BASE_DIR, "new_prime_cleaned")
    txn_output = os.path.join(config.BASE_DIR, "new_transaction_cleaned")

    # ── 1. Run prestep ──
    logs.append("")
    logs.append("Step 1/4: Running data-cleaning prestep on new files...")
    progress_callback(15)
    run_prestep(
        raw_prime_dir=raw_prime_dir,
        raw_transaction_dir=raw_transaction_dir,
        prime_output_dir=prime_output,
        transaction_output_dir=txn_output,
        logs=logs
    )
    progress_callback(35)

    # ── 2. Load & cast cleaned data ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" Step 2/4: Feature Engineering")
    logs.append("=" * 60)

    # Load prime
    prime_files = glob.glob(os.path.join(prime_output, "*_active.csv"))
    if not prime_files:
        raise FileNotFoundError(f"No active prime files in '{prime_output}'.")

    prime_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in prime_files]
    prime_df = pd.concat(prime_dfs, ignore_index=True)
    logs.append(f"Loaded {len(prime_df)} prime rows from {len(prime_files)} files.")

    apply_cast_and_report(prime_df, config.PRIME_STRING_COLS, 'string', logs)
    apply_cast_and_report(prime_df, config.PRIME_INT_COLS, 'int', logs)
    apply_cast_and_report(prime_df, config.PRIME_FLOAT_COLS, 'float', logs)
    apply_cast_and_report(prime_df, config.PRIME_DATE_COLS, 'date', logs)
    progress_callback(45)

    # Load transactions
    all_txn_files = glob.glob(os.path.join(txn_output, "*.csv"))
    txn_files = [f for f in all_txn_files if not f.endswith("_missing_id.csv")]

    has_transactions = len(txn_files) > 0
    if has_transactions:
        txn_dfs = [pd.read_csv(f, encoding='latin', dtype=str) for f in txn_files]
        transaction_df = pd.concat(txn_dfs, ignore_index=True)
        logs.append(f"Loaded {len(transaction_df)} transaction rows from {len(txn_files)} files.")

        apply_cast_and_report(transaction_df, config.TXN_STRING_COLS, 'string', logs)
        apply_cast_and_report(transaction_df, config.TXN_INT_COLS, 'int', logs)
        apply_cast_and_report(transaction_df, config.TXN_FLOAT_COLS, 'float', logs)
        apply_cast_and_report(transaction_df, config.TXN_DATE_COLS, 'date', logs)
    else:
        logs.append("WARNING: No transaction files found. Transaction-based features will be zero.")
        transaction_df = None

    # ── Drop unneeded columns ──
    from preprocessing import drop_unneeded_columns
    prime_df = drop_unneeded_columns(prime_df, prime_only=True)
    if transaction_df is not None:
        transaction_df = drop_unneeded_columns(transaction_df, txn_only=True)

    prime_df["GENDER"] = prime_df["GENDER"].fillna("Unknown")

    # ── Preserve RIMNO → CUSTOMER_ID mapping before dedup ──
    rimno_map = prime_df[['CUSTOMER_ID', 'RIMNO']].drop_duplicates(subset=['CUSTOMER_ID'])

    # ── Build features ──
    prime_df, user_item_df = build_user_item_matrix(prime_df, logs)
    rfm_features = build_rfm_features(transaction_df)
    mcc_spend = build_mcc_spend(transaction_df)
    foreign_agg = build_foreign_trxn_features(transaction_df)
    prime_df = build_demographics_features(prime_df)

    profile = merge_all_features(prime_df, rfm_features, mcc_spend, foreign_agg, logs)

    # Fill NaN features & One hot encoding & drop features
    from preprocessing import fill_missing_values, one_hot_encode_categoricals
    profile = fill_missing_values(profile)
    profile = one_hot_encode_categoricals(profile)

    cols_to_drop = [
        'AGE', 'BRANCH_NAME', 'PRODUCT_NAME', 'DOB', 'RIMNO',
        'DOB_WAS_MISSING', 'GENDER_Unknown'
    ]
    existing_drop = [c for c in cols_to_drop if c in profile.columns]
    profile = profile.drop(columns=existing_drop)
    profile = profile.drop(columns=['BRANCH_ID'], errors='ignore')

    # Drop any HAS_PROD_ columns that might exist in the new data
    prod_cols_new = [c for c in profile.columns if c.startswith('HAS_PROD_')]
    if prod_cols_new:
        profile = profile.drop(columns=prod_cols_new)

    # Drop rows where CUSTOMER_ID is missing (unmapped during prestep)
    na_count = profile['CUSTOMER_ID'].isna().sum()
    if na_count > 0:
        logs.append(f"  Dropping {na_count} rows with missing CUSTOMER_ID.")
        profile = profile.dropna(subset=['CUSTOMER_ID']).reset_index(drop=True)

    profile['CUSTOMER_ID'] = profile['CUSTOMER_ID'].astype(int)
    customer_ids = profile['CUSTOMER_ID'].values
    logs.append(f"Built feature profile for {len(profile)} customers.")
    progress_callback(60)

    # ── 3. Align features with trained model ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" Step 3/4: Aligning Features with Trained Model")
    logs.append("=" * 60)

    for col in trained_feature_cols:
        if col not in profile.columns:
            profile[col] = 0

    X = profile[trained_feature_cols].fillna(0)

    present = sum(1 for c in trained_feature_cols if c in profile.columns)
    logs.append(f"Trained model expects {len(trained_feature_cols)} features.")
    logs.append(f"  Matched from new data: {present}")
    logs.append(f"  Filled with zeros:     {len(trained_feature_cols) - present}")
    progress_callback(70)

    # ── 4. Predict ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" Step 4/4: Running Predictions")
    logs.append("=" * 60)

    results = []

    # XGBoost predictions
    logs.append(f"Running XGBoost predictions for {len(valid_targets)} products...")
    for i, product in enumerate(valid_targets):
        model = models_dict[product]
        proba = model.predict_proba(X)[:, 1]
        threshold = optimal_thresholds[product]
        product_name = product.replace('HAS_PROD_', '')

        mask = proba >= threshold
        if mask.any():
            for cid, prob in zip(customer_ids[mask], proba[mask]):
                results.append({
                    'CUSTOMER_ID': int(cid),
                    'PRODUCT_NAME': product_name,
                    'PROBABILITY': round(float(prob) * 100, 2),
                    'THRESHOLD': round(float(threshold) * 100, 2),
                    'MODEL': 'XGBoost',
                })

        if (i + 1) % 5 == 0 or (i + 1) == len(valid_targets):
            pct = 70 + int(20 * (i + 1) / len(valid_targets))
            progress_callback(pct)
            logs.append(f"  Predicted {i+1}/{len(valid_targets)} products...")

    logs.append(f"XGBoost: {len(results)} recommended predictions.")

    # CBF predictions
    if sim_matrix is not None and cbf_product_cols is not None and cbf_thresholds is not None:
        logs.append("Running CBF predictions...")
        cbf_rec_count = 0

        if user_item_df is None:
            logs.append("  No product ownership data in new files — skipping CBF.")
        else:
            common_prods = [c for c in cbf_product_cols if c in user_item_df.columns and c in sim_matrix.index]
            if not common_prods:
                logs.append("  No matching product columns between CBF model and new data — skipping CBF.")
            else:
                logs.append(f"  {len(common_prods)} CBF product columns matched.")
                valid_cids = [int(cid) for cid in customer_ids if cid in user_item_df.index]
                if not valid_cids:
                    logs.append("  No customers found in user-item matrix — skipping CBF.")
                else:
                    user_held = user_item_df.loc[valid_cids, common_prods].values
                    sim_aligned = sim_matrix.loc[common_prods, common_prods].values
                    scores = user_held @ sim_aligned
                    threshold_arr = np.array([cbf_thresholds.get(p, 0.5) for p in common_prods])

                    above_threshold = scores >= threshold_arr
                    not_held = user_held == 0
                    recommend = above_threshold & not_held

                    cust_indices, prod_indices = np.where(recommend)
                    for ci, pi in zip(cust_indices, prod_indices):
                        results.append({
                            'CUSTOMER_ID': valid_cids[ci],
                            'PRODUCT_NAME': common_prods[pi].replace('HAS_PROD_', ''),
                            'PROBABILITY': round(float(scores[ci, pi]) * 100, 2),
                            'THRESHOLD': round(float(threshold_arr[pi]) * 100, 2),
                            'MODEL': 'CBF',
                        })
                        cbf_rec_count += 1

                    logs.append(f"  CBF: {cbf_rec_count} recommended predictions.")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        logs.append("WARNING: No products predicted for any customer.")
        output_path = config.BATCH_OUTPUT_PATH
        pd.DataFrame(columns=['RIMNO', 'CUSTOMER_ID', 'PRODUCT_NAME', 'PROBABILITY', 'THRESHOLD', 'MODEL']).to_csv(output_path, index=False)
        return pd.DataFrame(), output_path, logs

    results_df = results_df.merge(rimno_map, on='CUSTOMER_ID', how='left')

    col_order = ['RIMNO', 'CUSTOMER_ID', 'PRODUCT_NAME', 'PROBABILITY', 'THRESHOLD', 'MODEL']
    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    results_df = results_df.sort_values(
        ['CUSTOMER_ID', 'MODEL', 'PROBABILITY'],
        ascending=[True, True, False]
    )

    output_path = config.BATCH_OUTPUT_PATH
    results_df.to_csv(output_path, index=False)

    unique_customers = results_df['CUSTOMER_ID'].nunique()
    logs.append("")
    logs.append("=" * 60)
    logs.append(" BATCH PREDICTION SUMMARY")
    logs.append("=" * 60)
    logs.append(f"Total customers processed: {len(profile)}")
    logs.append(f"Customers with recommendations: {unique_customers}")
    logs.append(f"Total recommendations: {len(results_df)}")
    logs.append(f"Saved to: {output_path}")

    return results_df, output_path, logs