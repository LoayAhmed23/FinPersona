"""
app.py
======
Flask GUI for the Churn Prediction Pipeline.
Provides a premium web interface to:
  1. Train multi-classifier churn models (with optional GridSearch tuning)
  2. View model comparison and evaluation results
  3. Look up individual customer churn risk
"""

import glob
import json
import os
import sys
import threading
import traceback

import config
from flask import Flask, jsonify, render_template, request

# Ensure the churn module directory is on sys.path so local imports work
CHURN_DIR = os.path.dirname(os.path.abspath(__file__))
if CHURN_DIR not in sys.path:
    sys.path.insert(0, CHURN_DIR)

# Also ensure the project root is on sys.path so data_cleaning can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(CHURN_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


app = Flask(__name__)

# ── Persistence paths ──
SAVE_DIR = config.OUTPUT_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_JSON = os.path.join(SAVE_DIR, "churn_app_state.json")

# ── Global pipeline state ──
pipeline_state = {
    "status": "idle",  # idle | training | trained | error
    "training_logs": [],
    "metrics": None,  # model comparison DataFrame as dict
    "error": None,
    "progress_pct": 0,
    # Data stats
    "total_samples": 0,
    "n_features": 0,
    "churn_rate": 0.0,
    "n_churned": 0,
    "n_retained": 0,
    # Best model
    "best_model_name": "",
    "report_text": "",
}


def _save_state_meta():
    """Persist lightweight state for UI reload."""
    saveable = {
        k: v for k, v in pipeline_state.items() if k not in ("df", "X", "y", "model")
    }
    try:
        with open(STATE_JSON, "w") as f:
            json.dump(saveable, f, default=str)
    except Exception:
        pass


def _load_state_meta():
    """Restore lightweight state on startup."""
    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON, "r") as f:
                saved = json.load(f)
            pipeline_state.update(saved)
            print(f"[Startup] Restored UI state from {STATE_JSON}")
        except Exception as e:
            print(f"[Startup] Could not restore state: {e}")

    if os.path.exists(config.MODEL_PATH):
        status_order = ["idle", "training", "trained", "error"]
        cur = pipeline_state["status"]
        if cur in status_order and status_order.index("trained") > status_order.index(
            cur
        ):
            pipeline_state["status"] = "trained"
        pipeline_state["training_logs"] = pipeline_state.get("training_logs") or [
            "[Loaded from cache] Trained model found on disk."
        ]

    if os.path.exists(config.REPORT_PATH):
        try:
            with open(config.REPORT_PATH, "r") as f:
                pipeline_state["report_text"] = f.read()
        except Exception:
            pass


_load_state_meta()


# ── Data-cleaning helper ──


def _ensure_cleaned_dirs(prime_dir, txn_dir, log_fn=None):
    """Ensure cleaned data directories exist; run cleaning pipelines if needed.

    The churn module needs three directories:
      1. **raw prime** — for churn labeling (reads raw CSVs like FEB2026.csv)
      2. **cleaned prime** — for feature engineering (reads *_active.csv)
      3. **cleaned transaction** — for feature engineering (reads *.csv)

    The user supplies at most two paths from the UI (prime_dir, txn_dir).
    This function resolves all three by:
      - Normalising away any trailing ``_cleaned`` so we never double-suffix
      - Checking if the cleaned sibling exists; if not, running the pipeline

    Returns
    -------
    (raw_prime_dir, cleaned_prime_dir, cleaned_txn_dir)
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)
        print(msg)

    def _resolve_raw_and_cleaned(user_dir, default_raw, default_cleaned):
        """Return (raw_dir, cleaned_dir) from whatever the user gave us."""
        if user_dir:
            base = user_dir.rstrip("/\\")
            if base.endswith("_cleaned"):
                base = base[: -len("_cleaned")]
            return base, base + "_cleaned"
        else:
            return default_raw, default_cleaned

    # Default directories from config
    default_raw_prime = config.RAW_PRIME_DATA_DIR
    default_raw_txn = os.path.join(config.DATA_DIR, "transaction")

    # ── Prime ──
    raw_prime, cleaned_prime = _resolve_raw_and_cleaned(
        prime_dir,
        default_raw_prime,
        config.CLEANED_PRIME_DATA_DIR,
    )

    has_prime = os.path.isdir(cleaned_prime) and glob.glob(
        os.path.join(cleaned_prime, "*_active.csv")
    )
    if not has_prime:
        _log(
            f"[AUTO-CLEAN] '{cleaned_prime}' not found or empty — running prime cleaning pipeline ..."
        )
        _log(f"[AUTO-CLEAN]   raw dir -> {raw_prime}")
        from data_cleaning.prime_id_creation import run as run_prime_cleaning

        run_prime_cleaning(input_dir=raw_prime, output_dir=cleaned_prime)
        _log(f"[AUTO-CLEAN] Prime cleaning complete -> {cleaned_prime}")
    else:
        _log(
            f"[AUTO-CLEAN] Found existing cleaned prime data at '{cleaned_prime}' — skipping."
        )

    # ── Transaction ──
    raw_txn, cleaned_txn = _resolve_raw_and_cleaned(
        txn_dir,
        default_raw_txn,
        config.CLEANED_TRANSACTION_DATA_DIR,
    )

    has_txn = os.path.isdir(cleaned_txn) and glob.glob(
        os.path.join(cleaned_txn, "*.csv")
    )
    if not has_txn:
        _log(
            f"[AUTO-CLEAN] '{cleaned_txn}' not found or empty — running transaction cleaning pipeline ..."
        )
        _log(f"[AUTO-CLEAN]   raw dir -> {raw_txn}")
        from data_cleaning.transaction_id_mapping import run as run_txn_cleaning

        run_txn_cleaning(
            prime_cleaned_dir=cleaned_prime,
            transaction_input_dir=raw_txn,
            transaction_output_dir=cleaned_txn,
        )
        _log(f"[AUTO-CLEAN] Transaction cleaning complete -> {cleaned_txn}")
    else:
        _log(
            f"[AUTO-CLEAN] Found existing cleaned transaction data at '{cleaned_txn}' — skipping."
        )

    return raw_prime, cleaned_prime, cleaned_txn


# ── Routes ──


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return current pipeline state (polled by the frontend)."""
    return jsonify(
        {
            "status": pipeline_state["status"],
            "training_logs": pipeline_state["training_logs"],
            "metrics": pipeline_state["metrics"],
            "error": pipeline_state["error"],
            "progress_pct": pipeline_state["progress_pct"],
            "total_samples": pipeline_state["total_samples"],
            "n_features": pipeline_state["n_features"],
            "churn_rate": pipeline_state["churn_rate"],
            "n_churned": pipeline_state["n_churned"],
            "n_retained": pipeline_state["n_retained"],
            "best_model_name": pipeline_state["best_model_name"],
            "report_text": pipeline_state.get("report_text", ""),
        }
    )


@app.route("/api/train", methods=["POST"])
def api_train():
    """Kick off the full training pipeline in a background thread."""
    if pipeline_state["status"] == "training":
        return jsonify({"error": "Training is already running."}), 400

    data = request.json or {}
    tune = data.get("tune", False)
    prime_dir = data.get("prime_dir", "").strip() or None
    txn_dir = data.get("txn_dir", "").strip() or None

    # Reset state
    pipeline_state.update(
        {
            "status": "training",
            "training_logs": [],
            "metrics": None,
            "error": None,
            "progress_pct": 5,
            "total_samples": 0,
            "n_features": 0,
            "churn_rate": 0.0,
            "n_churned": 0,
            "n_retained": 0,
            "best_model_name": "",
            "report_text": "",
        }
    )

    def _run():
        try:
            logs = []

            def log(msg):
                logs.append(msg)
                pipeline_state["training_logs"] = list(logs)

            log(f"[START] {'Tuned' if tune else 'Default'} training pipeline ...")

            # ── Auto-clean raw directories if needed ──
            pipeline_state["progress_pct"] = 7
            log("[PRE-CHECK] Verifying cleaned data directories ...")
            raw_prime, cleaned_prime, cleaned_txn = _ensure_cleaned_dirs(
                prime_dir,
                txn_dir,
                log_fn=log,
            )

            # Step 1: Churn labeling (uses RAW prime files)
            pipeline_state["progress_pct"] = 10
            log("[STEP 1/7] Creating churn labels ...")
            from data_loader import (
                create_churn_labels,
                load_prime_data,
                load_transaction_data,
            )

            churn_labels = create_churn_labels(raw_prime)
            n_churned_label = int(churn_labels[config.TARGET_COL].sum())
            log(
                f"  Labeled {len(churn_labels):,} customers — {n_churned_label:,} churned"
            )

            # Step 2: Load data (uses CLEANED dirs)
            pipeline_state["progress_pct"] = 20
            log("[STEP 2/7] Loading cleaned data ...")
            prime_df = load_prime_data(cleaned_prime)
            txn_df = load_transaction_data(cleaned_txn)
            log(f"  Transactions: {len(txn_df):,} rows")
            log(f"  Prime: {len(prime_df):,} rows")

            # Step 3: Feature engineering
            pipeline_state["progress_pct"] = 35
            log("[STEP 3/7] Engineering features ...")
            from feature_engineering import (
                engineer_prime_features,
                engineer_transaction_features,
                merge_all,
            )

            txn_features = engineer_transaction_features(txn_df)
            prime_features = engineer_prime_features(prime_df)
            log(f"  Txn features: {txn_features.shape}")
            log(f"  Prime features: {prime_features.shape}")

            # Step 4: Merge
            pipeline_state["progress_pct"] = 45
            log("[STEP 4/7] Merging datasets ...")
            final_df = merge_all(txn_features, prime_features, churn_labels)
            log(f"  Final dataset: {final_df.shape}")

            # Step 5: Preprocessing
            pipeline_state["progress_pct"] = 55
            log("[STEP 5/7] Preprocessing ...")
            from preprocessing import preprocess

            X, y, artifacts = preprocess(final_df, fit=True)

            n_total = len(y)
            n_churned = int(y.sum())
            n_retained = n_total - n_churned
            churn_rate = round(n_churned / n_total * 100, 2) if n_total > 0 else 0

            pipeline_state["total_samples"] = n_total
            pipeline_state["n_features"] = X.shape[1]
            pipeline_state["n_churned"] = n_churned
            pipeline_state["n_retained"] = n_retained
            pipeline_state["churn_rate"] = churn_rate

            log(
                f"  Samples: {n_total:,} | Features: {X.shape[1]} | Churn rate: {churn_rate}%"
            )

            # Step 6: Train/test split + training
            pipeline_state["progress_pct"] = 65
            log("[STEP 6/7] Training models ...")

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=y,
            )
            log(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

            from model import train_classifiers, tune_models

            results = train_classifiers(X_train, y_train, X_test, y_test)

            for name in results:
                log(f"  ✓ {name} trained")

            if tune:
                pipeline_state["progress_pct"] = 80
                log("[STEP 6b] Running GridSearchCV tuning ...")
                tuned = tune_models(X_train, y_train)
                for name, info in tuned.items():
                    model = info["model"]
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)[:, 1]
                    results[f"{name} (tuned)"] = {
                        "model": model,
                        "preds": preds,
                        "probs": probs,
                    }
                    log(f"  ✓ {name} (tuned) — best params: {info['best_params']}")

            # Step 7: Evaluate
            pipeline_state["progress_pct"] = 90
            log("[STEP 7/7] Evaluating models ...")

            from evaluation import evaluate_all, generate_report

            metrics_df = evaluate_all(y_test, results)

            # Convert to serializable format
            metrics_dict = {}
            for model_name in metrics_df.index:
                metrics_dict[model_name] = {
                    col: round(float(metrics_df.loc[model_name, col]), 4)
                    for col in metrics_df.columns
                }
            pipeline_state["metrics"] = metrics_dict

            best_name = metrics_df.index[0]
            best_result = results[best_name]
            pipeline_state["best_model_name"] = best_name

            log(f"\n  🏆 Best model: {best_name}")
            for col in metrics_df.columns:
                val = metrics_df.loc[best_name, col]
                log(f"     {col}: {val:.4f}")

            # Generate report
            report = generate_report(
                metrics_df,
                y_test,
                best_name,
                best_result["preds"],
                output_path=config.REPORT_PATH,
            )
            pipeline_state["report_text"] = report

            # Save best model
            from model import save_model

            save_model(best_result["model"], artifacts)
            log(f"\n  Model saved to {config.MODEL_PATH}")

            # Save scores
            import pandas as pd

            all_probs = best_result["model"].predict_proba(X)[:, 1]
            all_preds = (all_probs >= config.THRESHOLD).astype(int)
            scores_df = pd.DataFrame(
                {
                    config.CUSTOMER_ID: final_df[config.CUSTOMER_ID].values,
                    "churn_probability": all_probs,
                    "predicted_churn": all_preds,
                }
            )
            scores_df.to_csv(config.SCORES_PATH, index=False)
            log(f"  Scores saved to {config.SCORES_PATH}")

            pipeline_state["status"] = "trained"
            pipeline_state["progress_pct"] = 100
            log("\n[COMPLETE] Training pipeline finished successfully.")
            _save_state_meta()

        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = f"{str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["progress_pct"] = 0
            pipeline_state["training_logs"].append(f"[ERROR] {str(e)}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"message": "Training pipeline started."})


@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    """Look up a customer's churn risk from the scored CSV."""
    if not os.path.exists(config.SCORES_PATH):
        return jsonify({"error": "No scores available. Run training first."}), 400

    data = request.json or {}
    customer_id_raw = str(data.get("customer_id", "")).strip()
    if not customer_id_raw:
        return jsonify({"error": "Please provide a Customer ID."}), 400

    try:
        import pandas as pd

        scores = pd.read_csv(config.SCORES_PATH)
        # Try matching as string (flexible)
        scores[config.CUSTOMER_ID] = scores[config.CUSTOMER_ID].astype(str).str.strip()
        match = scores[scores[config.CUSTOMER_ID] == customer_id_raw]

        if match.empty:
            return (
                jsonify(
                    {"error": f"Customer ID {customer_id_raw} not found in scores."}
                ),
                404,
            )

        row = match.iloc[0]
        prob = round(float(row["churn_probability"]) * 100, 2)
        label = int(row["predicted_churn"])

        # Determine risk tier
        if prob >= 70:
            risk_tier = "Critical"
        elif prob >= 40:
            risk_tier = "High"
        elif prob >= 20:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        return jsonify(
            {
                "customer_id": customer_id_raw,
                "churn_probability": prob,
                "predicted_churn": label,
                "risk_tier": risk_tier,
                "label_text": "Likely to Churn" if label == 1 else "Likely to Stay",
            }
        )
    except Exception as e:
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


@app.route("/api/customers")
def api_customers():
    """Return customer IDs from scores CSV for autocomplete."""
    if not os.path.exists(config.SCORES_PATH):
        return jsonify({"customers": []})

    q = request.args.get("q", "").strip()
    try:
        import pandas as pd

        scores = pd.read_csv(config.SCORES_PATH)
        ids = scores[config.CUSTOMER_ID].dropna().astype(str).str.strip().tolist()
        if q:
            ids = [cid for cid in ids if q in cid]
        return jsonify({"customers": ids[:50]})
    except Exception:
        return jsonify({"customers": []})


@app.route("/api/report")
def api_report():
    """Return the evaluation report text."""
    if os.path.exists(config.REPORT_PATH):
        with open(config.REPORT_PATH, "r") as f:
            return jsonify({"report": f.read()})
    return jsonify({"report": ""})


@app.route("/api/browse", methods=["POST"])
def api_browse():
    """Open a native Windows folder picker dialog and return the selected path."""
    import subprocess

    script = (
        "import tkinter as tk; "
        "from tkinter import filedialog; "
        "root = tk.Tk(); "
        "root.withdraw(); "
        "root.attributes('-topmost', True); "
        "path = filedialog.askdirectory(title='Select Directory'); "
        "print(path); "
        "root.destroy()"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        selected = result.stdout.strip()
        if selected:
            return jsonify({"path": selected.replace("\\", "/")})
        else:
            return jsonify({"path": ""})  # user cancelled
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Dialog timed out."}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5006)
