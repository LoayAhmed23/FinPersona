import argparse
import json
import pickle
import sys

import config
import numpy as np
import pandas as pd
from data_loader import load_prime_data, load_transaction_data
from feature_engineering import (
    build_demographics_features,
    build_foreign_trxn_features,
    build_mcc_spend,
    build_rfm_features,
    build_user_item_matrix,
    merge_all_features,
)
from model import train_cbf, train_xgboost
from preprocessing import preprocess_pipeline


def run_PREPROCESSING_pipeline(prime_dir=None, transaction_dir=None, logs=None):
    """
    Runs the full Preprocessing pipeline.

    Reads already-cleaned prime and transaction CSVs from the given
    directories (defaulting to the centralized paths in config).

    Returns (final_customer_profile DataFrame, logs list, product_cols list, feature_cols list).
    """
    if logs is None:
        logs = []

    # Phase 1: Load cleaned prime data
    prime_df = load_prime_data(prime_dir, logs)

    # Phase 2: Load cleaned transaction data
    transaction_df = load_transaction_data(transaction_dir, logs)

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
    final_customer_profile, final_products, final_features = preprocess_pipeline(
        profile, logs
    )

    return final_customer_profile, logs, final_products, final_features


def predict_new_data(
    prime_dir,
    transaction_dir,
    models_dict,
    optimal_thresholds,
    trained_feature_cols,
    valid_targets,
    sim_matrix=None,
    cbf_product_cols=None,
    cbf_thresholds=None,
    logs=None,
    progress_callback=None,
):
    """
    Processes new cleaned prime/transaction data, engineers the same features
    used during training, and predicts products for every customer using the
    already-trained XGBoost (and optionally CBF) models.

    Unlike the old version, this assumes the data is already cleaned —
    no prestep is run.
    """
    if logs is None:
        logs = []
    if progress_callback is None:

        def progress_callback(_pct):
            return None

    logs.append("=" * 60)
    logs.append(" BATCH PREDICTION: Processing New Customer Data")
    logs.append("=" * 60)

    # ── 1. Load cleaned data ──
    logs.append("")
    logs.append("Step 1/3: Loading cleaned data...")
    progress_callback(15)

    prime_df = load_prime_data(prime_dir, logs)
    progress_callback(25)

    transaction_df = load_transaction_data(transaction_dir, logs)
    has_transactions = transaction_df is not None and len(transaction_df) > 0
    progress_callback(35)

    # ── Drop unneeded columns ──
    from preprocessing import drop_unneeded_columns

    prime_df = drop_unneeded_columns(prime_df, prime_only=True)
    if has_transactions:
        transaction_df = drop_unneeded_columns(transaction_df, txn_only=True)

    prime_df["GENDER"] = prime_df["GENDER"].fillna("Unknown")

    # ── Preserve RIMNO → CUSTOMER_ID mapping before dedup ──
    rimno_map = prime_df[["CUSTOMER_ID", "RIMNO"]].drop_duplicates(
        subset=["CUSTOMER_ID"]
    )

    # ── 2. Feature Engineering ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" Step 2/3: Feature Engineering")
    logs.append("=" * 60)

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
        "AGE",
        "BRANCH_NAME",
        "PRODUCT_NAME",
        "DOB",
        "RIMNO",
        "DOB_WAS_MISSING",
        "GENDER_Unknown",
    ]
    existing_drop = [c for c in cols_to_drop if c in profile.columns]
    profile = profile.drop(columns=existing_drop)
    profile = profile.drop(columns=["BRANCH_ID"], errors="ignore")

    # Drop any HAS_PROD_ columns that might exist in the new data
    prod_cols_new = [c for c in profile.columns if c.startswith("HAS_PROD_")]
    if prod_cols_new:
        profile = profile.drop(columns=prod_cols_new)

    # Drop rows where CUSTOMER_ID is missing
    na_count = profile["CUSTOMER_ID"].isna().sum()
    if na_count > 0:
        logs.append(f"  Dropping {na_count} rows with missing CUSTOMER_ID.")
        profile = profile.dropna(subset=["CUSTOMER_ID"]).reset_index(drop=True)

    profile["CUSTOMER_ID"] = profile["CUSTOMER_ID"].astype(int)
    customer_ids = profile["CUSTOMER_ID"].values
    logs.append(f"Built feature profile for {len(profile)} customers.")
    progress_callback(60)

    # ── 3. Align features with trained model & Predict ──
    logs.append("")
    logs.append("=" * 60)
    logs.append(" Step 3/3: Aligning Features & Running Predictions")
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

    results = []

    # XGBoost predictions
    logs.append(f"Running XGBoost predictions for {len(valid_targets)} products...")
    for i, product in enumerate(valid_targets):
        model = models_dict[product]
        proba = model.predict_proba(X)[:, 1]
        threshold = optimal_thresholds[product]
        product_name = product.replace("HAS_PROD_", "")

        mask = proba >= threshold
        if mask.any():
            for cid, prob in zip(customer_ids[mask], proba[mask]):
                results.append(
                    {
                        "CUSTOMER_ID": int(cid),
                        "PRODUCT_NAME": product_name,
                        "PROBABILITY": round(float(prob) * 100, 2),
                        "THRESHOLD": round(float(threshold) * 100, 2),
                        "MODEL": "XGBoost",
                    }
                )

        if (i + 1) % 5 == 0 or (i + 1) == len(valid_targets):
            pct = 70 + int(20 * (i + 1) / len(valid_targets))
            progress_callback(pct)
            logs.append(f"  Predicted {i+1}/{len(valid_targets)} products...")

    logs.append(f"XGBoost: {len(results)} recommended predictions.")

    # CBF predictions
    if (
        sim_matrix is not None
        and cbf_product_cols is not None
        and cbf_thresholds is not None
    ):
        logs.append("Running CBF predictions...")
        cbf_rec_count = 0

        if user_item_df is None:
            logs.append("  No product ownership data in new files — skipping CBF.")
        else:
            common_prods = [
                c
                for c in cbf_product_cols
                if c in user_item_df.columns and c in sim_matrix.index
            ]
            if not common_prods:
                logs.append(
                    "  No matching product columns between CBF model and new data — skipping CBF."
                )
            else:
                logs.append(f"  {len(common_prods)} CBF product columns matched.")
                valid_cids = [
                    int(cid) for cid in customer_ids if cid in user_item_df.index
                ]
                if not valid_cids:
                    logs.append(
                        "  No customers found in user-item matrix — skipping CBF."
                    )
                else:
                    user_held = user_item_df.loc[valid_cids, common_prods].values
                    sim_aligned = sim_matrix.loc[common_prods, common_prods].values
                    scores = user_held @ sim_aligned
                    threshold_arr = np.array(
                        [cbf_thresholds.get(p, 0.5) for p in common_prods]
                    )

                    above_threshold = scores >= threshold_arr
                    not_held = user_held == 0
                    recommend = above_threshold & not_held

                    cust_indices, prod_indices = np.where(recommend)
                    for ci, pi in zip(cust_indices, prod_indices):
                        results.append(
                            {
                                "CUSTOMER_ID": valid_cids[ci],
                                "PRODUCT_NAME": common_prods[pi].replace(
                                    "HAS_PROD_", ""
                                ),
                                "PROBABILITY": round(float(scores[ci, pi]) * 100, 2),
                                "THRESHOLD": round(float(threshold_arr[pi]) * 100, 2),
                                "MODEL": "CBF",
                            }
                        )
                        cbf_rec_count += 1

                    logs.append(f"  CBF: {cbf_rec_count} recommended predictions.")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        logs.append("WARNING: No products predicted for any customer.")
        output_path = config.BATCH_OUTPUT_PATH
        pd.DataFrame(
            columns=[
                "RIMNO",
                "CUSTOMER_ID",
                "PRODUCT_NAME",
                "PROBABILITY",
                "THRESHOLD",
                "MODEL",
            ]
        ).to_csv(output_path, index=False)
        return pd.DataFrame(), output_path, logs

    results_df = results_df.merge(rimno_map, on="CUSTOMER_ID", how="left")

    col_order = [
        "RIMNO",
        "CUSTOMER_ID",
        "PRODUCT_NAME",
        "PROBABILITY",
        "THRESHOLD",
        "MODEL",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    results_df = results_df.sort_values(
        ["CUSTOMER_ID", "MODEL", "PROBABILITY"], ascending=[True, True, False]
    )

    output_path = config.BATCH_OUTPUT_PATH
    results_df.to_csv(output_path, index=False)

    unique_customers = results_df["CUSTOMER_ID"].nunique()
    logs.append("")
    logs.append("=" * 60)
    logs.append(" BATCH PREDICTION SUMMARY")
    logs.append("=" * 60)
    logs.append(f"Total customers processed: {len(profile)}")
    logs.append(f"Customers with recommendations: {unique_customers}")
    logs.append(f"Total recommendations: {len(results_df)}")
    logs.append(f"Saved to: {output_path}")

    return results_df, output_path, logs


def main():
    """
    CLI entry point for the Recommendation System.

    Usage
    -----
        python main.py train                      # Train with default cleaned data dirs
        python main.py train --prime-dir /path    # Train with custom cleaned data dir
        python main.py score --prime-dir /path    # Score new cleaned data
    """
    parser = argparse.ArgumentParser(
        description="Recommendation Pipeline System",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    train_parser = sub.add_parser(
        "train", help="Run preprocessing and train the models"
    )
    train_parser.add_argument(
        "--prime-dir",
        default=None,
        help="Directory with cleaned prime CSVs (default: config)",
    )
    train_parser.add_argument(
        "--txn-dir",
        default=None,
        help="Directory with cleaned transaction CSVs (default: config)",
    )

    # --- score ---
    score_parser = sub.add_parser("score", help="Score new data using saved models")
    score_parser.add_argument(
        "--prime-dir",
        required=True,
        help="Directory with cleaned prime CSVs",
    )
    score_parser.add_argument(
        "--txn-dir",
        default=None,
        help="Directory with cleaned transaction CSVs (default: config)",
    )
    score_parser.add_argument(
        "--output",
        default=config.BATCH_OUTPUT_PATH,
        help="Output CSV path for predictions",
    )

    args = parser.parse_args()

    if args.command == "train":
        print("Running Preprocessing pipeline...")
        df, logs, final_products, final_features = run_PREPROCESSING_pipeline(
            args.prime_dir, args.txn_dir
        )
        print("Preprocessing done.")

        print("Training XGBoost...")
        models, thresholds, feature_cols, valid_targets, metrics, t_logs = (
            train_xgboost(df)
        )
        print("XGBoost training done.")

        print("Training CBF...")
        sim_matrix, cbf_product_cols, cbf_thresholds, cbf_metrics, c_logs = train_cbf(
            df
        )
        print("CBF training done.")

    elif args.command == "score":
        # Load saved models
        with open(config.XGB_MODELS_PKL, "rb") as f:
            models = pickle.load(f)
        with open(config.XGB_META_JSON, "r") as f:
            meta = json.load(f)
        thresholds = meta["thresholds"]
        feature_cols = meta["feature_cols"]
        valid_targets = meta["valid_targets"]

        with open(config.CBF_SIM_PKL, "rb") as f:
            cbf_data = pickle.load(f)
        sim_matrix = cbf_data["sim_matrix"]

        with open(config.CBF_META_JSON, "r") as f:
            cbf_meta = json.load(f)
        cbf_product_cols = cbf_meta["product_cols"]
        cbf_thresholds = cbf_meta["thresholds"]

        results_df, output_path, logs = predict_new_data(
            prime_dir=args.prime_dir,
            transaction_dir=args.txn_dir,
            models_dict=models,
            optimal_thresholds=thresholds,
            trained_feature_cols=feature_cols,
            valid_targets=valid_targets,
            sim_matrix=sim_matrix,
            cbf_product_cols=cbf_product_cols,
            cbf_thresholds=cbf_thresholds,
        )
        print(f"Predictions saved to {output_path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
