"""
CLI entry point for the Churn Prediction System.

Usage
-----
    python main.py train          # Train all classifiers
    python main.py tune           # Train + GridSearch tuning for RF/XGBoost
"""

import argparse
import sys

from pipeline import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Churn Prediction System",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    sub.add_parser("train", help="Train all classifiers and compare")

    # --- tune ---
    sub.add_parser("tune", help="Train + GridSearchCV hyperparameter tuning")

    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training_pipeline(tune=False)
        print("\nDone. Best model metrics:")
        print(metrics.iloc[0].to_string())

    elif args.command == "tune":
        metrics = run_training_pipeline(tune=True)
        print("\nDone. Best model metrics:")
        print(metrics.iloc[0].to_string())

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
