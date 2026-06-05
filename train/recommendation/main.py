"""
CLI entry point for the Recommendation System.

Usage
-----
    python main.py train          # Train with default XGBoost params
    python main.py score          # Score new data with a saved model
"""

import argparse
import sys
import pickle
import json
import pandas as pd
import config
from pipeline import run_PREPROCESSING_pipeline, predict_new_data
from model import train_xgboost, train_cbf

def main():
    parser = argparse.ArgumentParser(
        description="Recommendation Pipeline System",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    train_parser = sub.add_parser("train", help="Run preprocessing and train the models")
    train_parser.add_argument("--prime-dir", default="prime_cleaned", help="Directory with cleaned prime CSVs")
    train_parser.add_argument("--txn-dir", default="transaction_cleaned", help="Directory with cleaned transaction CSVs")

    # --- score ---
    score_parser = sub.add_parser("score", help="Score new data using saved models")
    score_parser.add_argument(
        "--raw-prime-dir", required=True,
        help="Directory with new raw prime CSVs",
    )
    score_parser.add_argument(
        "--raw-txn-dir", required=True,
        help="Directory with new raw transaction Excel files",
    )
    score_parser.add_argument(
        "--output", default=config.BATCH_OUTPUT_PATH,
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
        models, thresholds, feature_cols, valid_targets, metrics, t_logs = train_xgboost(df)
        print("XGBoost training done.")
        
        print("Training CBF...")
        sim_matrix, cbf_product_cols, cbf_thresholds, cbf_metrics, c_logs = train_cbf(df)
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
            raw_prime_dir=args.raw_prime_dir,
            raw_transaction_dir=args.raw_txn_dir,
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
