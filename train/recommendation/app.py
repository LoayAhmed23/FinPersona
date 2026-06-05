"""
app.py
======
Flask GUI for the HOLA Product Recommendation Pipeline.
Provides a premium web interface to:
1. Run Preprocessing pipeline on a data directory
2. Train XGBoost models with threshold tuning
3. Predict recommended products for a specific customer
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import threading
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Persistence paths ──
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
PREPROCESSED_CSV  = os.path.join(SAVE_DIR, "final_customer_profile.csv")
XGB_MODELS_PKL = os.path.join(SAVE_DIR, "xgb_models.pkl")
XGB_META_JSON  = os.path.join(SAVE_DIR, "xgb_meta.json")
CBF_SIM_PKL    = os.path.join(SAVE_DIR, "cbf_sim_matrix.pkl")
CBF_META_JSON  = os.path.join(SAVE_DIR, "cbf_meta.json")

# ── Global state ──
pipeline_state = {
    "status": "idle",           # idle | running_PREPROCESSING | PREPROCESSING_done | training | trained | error
    "PREPROCESSING_logs": [],
    "training_logs": [],
    "metrics": None,
    "error": None,
    "df": None,
    "models": None,
    "thresholds": None,
    "feature_cols": None,
    "valid_targets": None,
    "customer_count": 0,
    "product_count": 0,
    "feature_count": 0,
    "progress_pct": 0,
    "cbf_sim_matrix": None,
    "cbf_product_cols": None,
    "cbf_thresholds": None,
    "cbf_logs": [],
    "cbf_status": "idle",       # idle | training_cbf | cbf_trained | error
    "cbf_metrics": None,
    "cbf_progress_pct": 0,
    "batch_status": "idle",     # idle | running_batch | batch_done | error
    "batch_logs": [],
    "batch_progress_pct": 0,
    "batch_output_path": None,
    "batch_prediction_count": 0,
    "batch_customer_count": 0,
}


def load_saved_state():
    """Load previously saved artifacts on server startup."""
    # 1. Load Preprocessed CSV
    if os.path.exists(PREPROCESSED_CSV):
        try:
            df = pd.read_csv(PREPROCESSED_CSV)
            product_cols = [c for c in df.columns if c.startswith('HAS_PROD_')]
            feature_cols = [c for c in df.columns if c not in product_cols and c != 'CUSTOMER_ID' and c != 'BRANCH_ID']
            pipeline_state["df"] = df
            pipeline_state["customer_count"] = len(df)
            pipeline_state["product_count"] = len(product_cols)
            pipeline_state["feature_count"] = len(feature_cols)
            pipeline_state["status"] = "PREPROCESSING_done"
            pipeline_state["PREPROCESSING_logs"] = ["[Loaded from cache] Preprocessed data restored from saved_models/"]
            pipeline_state["progress_pct"] = 100
            print(f"[Startup] Loaded Preprocessed CSV ({len(df)} customers)")
        except Exception as e:
            print(f"[Startup] Failed to load Preprocessed CSV: {e}")

    # 2. Load XGBoost models
    if os.path.exists(XGB_MODELS_PKL) and os.path.exists(XGB_META_JSON):
        try:
            with open(XGB_MODELS_PKL, "rb") as f:
                models = pickle.load(f)
            with open(XGB_META_JSON, "r") as f:
                meta = json.load(f)
            pipeline_state["models"] = models
            pipeline_state["thresholds"] = meta["thresholds"]
            pipeline_state["feature_cols"] = meta["feature_cols"]
            pipeline_state["valid_targets"] = meta["valid_targets"]
            pipeline_state["metrics"] = meta.get("metrics")
            pipeline_state["status"] = "trained"
            pipeline_state["training_logs"] = ["[Loaded from cache] XGBoost models restored from saved_models/"]
            print(f"[Startup] Loaded XGBoost models ({len(models)} products)")
        except Exception as e:
            print(f"[Startup] Failed to load XGBoost models: {e}")

    # 3. Load CBF model
    if os.path.exists(CBF_SIM_PKL) and os.path.exists(CBF_META_JSON):
        try:
            with open(CBF_SIM_PKL, "rb") as f:
                cbf_data = pickle.load(f)
            with open(CBF_META_JSON, "r") as f:
                cbf_meta = json.load(f)
            pipeline_state["cbf_sim_matrix"] = cbf_data["sim_matrix"]
            pipeline_state["cbf_product_cols"] = cbf_meta["product_cols"]
            pipeline_state["cbf_thresholds"] = cbf_meta["thresholds"]
            pipeline_state["cbf_metrics"] = cbf_meta.get("metrics")
            pipeline_state["cbf_status"] = "cbf_trained"
            pipeline_state["cbf_logs"] = ["[Loaded from cache] CBF model restored from saved_models/"]
            pipeline_state["cbf_progress_pct"] = 100
            print(f"[Startup] Loaded CBF model")
        except Exception as e:
            print(f"[Startup] Failed to load CBF model: {e}")


# Auto-load on import
load_saved_state()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return current pipeline state (polled by the frontend)."""
    return jsonify({
        "status": pipeline_state["status"],
        "PREPROCESSING_logs": pipeline_state["PREPROCESSING_logs"],
        "training_logs": pipeline_state["training_logs"],
        "metrics": pipeline_state["metrics"],
        "error": pipeline_state["error"],
        "customer_count": pipeline_state["customer_count"],
        "product_count": pipeline_state["product_count"],
        "feature_count": pipeline_state["feature_count"],
        "progress_pct": pipeline_state["progress_pct"],
        "cbf_logs": pipeline_state["cbf_logs"],
        "cbf_status": pipeline_state["cbf_status"],
        "cbf_metrics": pipeline_state["cbf_metrics"],
        "cbf_progress_pct": pipeline_state["cbf_progress_pct"],
        "batch_status": pipeline_state["batch_status"],
        "batch_logs": pipeline_state["batch_logs"],
        "batch_progress_pct": pipeline_state["batch_progress_pct"],
        "batch_output_path": pipeline_state["batch_output_path"],
        "batch_prediction_count": pipeline_state["batch_prediction_count"],
        "batch_customer_count": pipeline_state["batch_customer_count"],
    })


@app.route("/api/browse_directory")
def api_browse_directory():
    """Opens a native OS folder picker and returns the selected path."""
    import tkinter as tk
    from tkinter import filedialog
    # Setup tkinter
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Directory")
    root.destroy()
    return jsonify({"path": folder_path})


@app.route("/api/run_PREPROCESSING", methods=["POST"])
def api_run_PREPROCESSING():
    """Kick off the Preprocessing pipeline in a background thread."""
    if pipeline_state["status"] not in ("idle", "PREPROCESSING_done", "trained", "error"):
        return jsonify({"error": "Pipeline is already running."}), 400

    data = request.json
    prime_dir = data.get("prime_dir", "").strip()
    transaction_dir = data.get("transaction_dir", "").strip() or None
    raw_prime_dir = data.get("raw_prime_dir", "").strip() or None
    raw_transaction_dir = data.get("raw_transaction_dir", "").strip() or None

    if not prime_dir:
        return jsonify({"error": "Please provide a prime data directory."}), 400

    # Reset state
    pipeline_state.update({
        "status": "running_PREPROCESSING",
        "PREPROCESSING_logs": [],
        "prestep_logs": [],
        "training_logs": [],
        "metrics": None,
        "error": None,
        "df": None,
        "models": None,
        "thresholds": None,
        "feature_cols": None,
        "valid_targets": None,
        "customer_count": 0,
        "product_count": 0,
        "feature_count": 0,
        "progress_pct": 10,
    })

    def _run():
        try:
            from pipeline import run_PREPROCESSING_pipeline
            pipeline_state["progress_pct"] = 20
            df, logs, product_cols, feature_cols = run_PREPROCESSING_pipeline(
                prime_dir, transaction_dir,
                raw_prime_dir=raw_prime_dir,
                raw_transaction_dir=raw_transaction_dir,
                logs=pipeline_state["PREPROCESSING_logs"]
            )
            pipeline_state["df"] = df
            pipeline_state["PREPROCESSING_logs"] = logs
            pipeline_state["customer_count"] = len(df)
            pipeline_state["product_count"] = len(product_cols)
            pipeline_state["feature_count"] = len(feature_cols)
            pipeline_state["status"] = "PREPROCESSING_done"
            pipeline_state["progress_pct"] = 100

            # Save CSV for next restart
            df.to_csv(PREPROCESSED_CSV, index=False)
            logs.append(f"[Saved] Preprocessed CSV cached to {PREPROCESSED_CSV}")
        except Exception as e:
            err_msg = f"CRITICAL ERROR: {str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["PREPROCESSING_logs"].append(err_msg)
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["progress_pct"] = 0

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"message": "Preprocessed pipeline started."})


@app.route("/api/train", methods=["POST"])
def api_train():
    """Kick off XGBoost training in a background thread."""
    if pipeline_state["df"] is None:
        return jsonify({"error": "No data loaded. Run Preprocessed first."}), 400
    if pipeline_state["status"] == "training":
        return jsonify({"error": "Training is already in progress."}), 400

    pipeline_state.update({
        "status": "training",
        "training_logs": [],
        "metrics": None,
        "error": None,
        "progress_pct": 10,
    })

    def _train():
        try:
            from model import train_xgboost
            pipeline_state["progress_pct"] = 30
            models, thresholds, feature_cols, valid_targets, metrics, logs = train_xgboost(
                pipeline_state["df"],
                logs=pipeline_state["training_logs"]
            )
            pipeline_state["models"] = models
            pipeline_state["thresholds"] = thresholds
            pipeline_state["feature_cols"] = feature_cols
            pipeline_state["valid_targets"] = valid_targets
            pipeline_state["training_logs"] = logs
            pipeline_state["metrics"] = metrics
            pipeline_state["status"] = "trained"
            pipeline_state["progress_pct"] = 100

            # Save models for next restart
            with open(XGB_MODELS_PKL, "wb") as f:
                pickle.dump(models, f)
            meta = {
                "thresholds": thresholds,
                "feature_cols": feature_cols,
                "valid_targets": valid_targets,
                "metrics": metrics,
            }
            with open(XGB_META_JSON, "w") as f:
                json.dump(meta, f)
            logs.append(f"[Saved] XGBoost models cached to {SAVE_DIR}/")
        except Exception as e:
            err_msg = f"CRITICAL ERROR: {str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["training_logs"].append(err_msg)
            pipeline_state["status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["progress_pct"] = 0

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return jsonify({"message": "Training started."})


@app.route("/api/train_cbf", methods=["POST"])
def api_train_cbf():
    """Kick off CBF training in a background thread."""
    if pipeline_state["df"] is None:
        return jsonify({"error": "No data loaded. Run Preprocessed first."}), 400
    if pipeline_state["cbf_status"] == "training_cbf":
        return jsonify({"error": "CBF training is already in progress."}), 400

    pipeline_state.update({
        "cbf_status": "training_cbf",
        "cbf_logs": [],
        "cbf_metrics": None,
        "cbf_progress_pct": 10,
    })

    def _train_cbf():
        try:
            from model import train_cbf
            pipeline_state["cbf_progress_pct"] = 40
            sim_matrix, cbf_product_cols, cbf_thresholds, cbf_metrics, cbf_logs = train_cbf(
                pipeline_state["df"],
                logs=pipeline_state["cbf_logs"]
            )
            pipeline_state["cbf_sim_matrix"] = sim_matrix
            pipeline_state["cbf_product_cols"] = cbf_product_cols
            pipeline_state["cbf_thresholds"] = cbf_thresholds
            pipeline_state["cbf_logs"] = cbf_logs
            pipeline_state["cbf_metrics"] = cbf_metrics
            pipeline_state["cbf_status"] = "cbf_trained"
            pipeline_state["cbf_progress_pct"] = 100

            # Save CBF model for next restart
            with open(CBF_SIM_PKL, "wb") as f:
                pickle.dump({"sim_matrix": sim_matrix}, f)
            cbf_meta = {
                "product_cols": cbf_product_cols,
                "thresholds": cbf_thresholds,
                "metrics": cbf_metrics,
            }
            with open(CBF_META_JSON, "w") as f:
                json.dump(cbf_meta, f)
            cbf_logs.append(f"[Saved] CBF model cached to {SAVE_DIR}/")
        except Exception as e:
            err_msg = f"CRITICAL ERROR: {str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["cbf_logs"].append(err_msg)
            pipeline_state["cbf_status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["cbf_progress_pct"] = 0

    thread = threading.Thread(target=_train_cbf, daemon=True)
    thread.start()
    return jsonify({"message": "CBF training started."})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict products for a given customer ID."""
    if pipeline_state["status"] != "trained":
        return jsonify({"error": "Model not trained yet. Complete training first."}), 400

    data = request.json
    customer_id_raw = data.get("customer_id", "").strip()
    if not customer_id_raw:
        return jsonify({"error": "Please provide a Customer ID."}), 400

    try:
        customer_id = int(customer_id_raw)
    except ValueError:
        return jsonify({"error": f"Invalid Customer ID: '{customer_id_raw}'. Must be an integer."}), 400

    try:
        from model import predict_for_customer
        predicted, already_holding, found = predict_for_customer(
            customer_id,
            pipeline_state["df"],
            pipeline_state["models"],
            pipeline_state["thresholds"],
            pipeline_state["feature_cols"],
            pipeline_state["valid_targets"],
        )
        if not found:
            return jsonify({"error": f"Customer ID {customer_id} not found in the dataset."}), 404

        return jsonify({
            "customer_id": customer_id,
            "predicted_products": predicted,
            "already_holding": already_holding,
        })
    except Exception as e:
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


@app.route("/api/predict_cbf", methods=["POST"])
def api_predict_cbf():
    """Predict products via CBF for a given customer ID."""
    if pipeline_state["status"] != "trained":
        return jsonify({"error": "Model not trained yet."}), 400
    if pipeline_state["cbf_sim_matrix"] is None or pipeline_state["cbf_thresholds"] is None:
        return jsonify({"error": "CBF model not trained. Train CBF first."}), 400

    data = request.json
    customer_id_raw = data.get("customer_id", "").strip()
    if not customer_id_raw:
        return jsonify({"error": "Please provide a Customer ID."}), 400

    try:
        customer_id = int(customer_id_raw)
    except ValueError:
        return jsonify({"error": f"Invalid Customer ID: '{customer_id_raw}'."}), 400

    try:
        from model import predict_cbf_for_customer
        recs, already_holding, found = predict_cbf_for_customer(
            customer_id,
            pipeline_state["df"],
            pipeline_state["cbf_sim_matrix"],
            pipeline_state["cbf_product_cols"],
            pipeline_state["cbf_thresholds"],
        )
        if not found:
            return jsonify({"error": f"Customer ID {customer_id} not found."}), 404

        return jsonify({
            "customer_id": customer_id,
            "cbf_recommendations": recs,
            "already_holding": already_holding,
        })
    except Exception as e:
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


@app.route("/api/customers")
def api_customers():
    """Return a sample of customer IDs for the autocomplete."""
    if pipeline_state["df"] is None:
        return jsonify({"customers": []})

    q = request.args.get("q", "").strip()
    df = pipeline_state["df"]
    ids = df["CUSTOMER_ID"].dropna().astype(int).tolist()

    if q:
        ids = [cid for cid in ids if q in str(cid)]

    return jsonify({"customers": ids[:50]})


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    """Run batch prediction on new customer data."""
    if pipeline_state["models"] is None:
        return jsonify({"error": "XGBoost models not trained yet. Complete training first."}), 400
    if pipeline_state["batch_status"] == "running_batch":
        return jsonify({"error": "Batch prediction is already running."}), 400

    data = request.json
    raw_prime_dir = data.get("raw_prime_dir", "").strip()
    raw_transaction_dir = data.get("raw_transaction_dir", "").strip()

    if not raw_prime_dir or not raw_transaction_dir:
        return jsonify({"error": "Please provide both raw prime and transaction directories."}), 400

    pipeline_state.update({
        "batch_status": "running_batch",
        "batch_logs": [],
        "batch_progress_pct": 10,
        "batch_output_path": None,
        "batch_prediction_count": 0,
        "batch_customer_count": 0,
    })

    def _run_batch():
        try:
            from pipeline import predict_new_data
            pipeline_state["batch_progress_pct"] = 10

            # Shared logs list — predict_new_data appends to it live
            live_logs = pipeline_state["batch_logs"]
            def _progress(pct):
                pipeline_state["batch_progress_pct"] = pct

            results_df, output_path, logs = predict_new_data(
                raw_prime_dir=raw_prime_dir,
                raw_transaction_dir=raw_transaction_dir,
                models_dict=pipeline_state["models"],
                optimal_thresholds=pipeline_state["thresholds"],
                trained_feature_cols=pipeline_state["feature_cols"],
                valid_targets=pipeline_state["valid_targets"],
                sim_matrix=pipeline_state.get("cbf_sim_matrix"),
                cbf_product_cols=pipeline_state.get("cbf_product_cols"),
                cbf_thresholds=pipeline_state.get("cbf_thresholds"),
                logs=pipeline_state["batch_logs"],
                progress_callback=_progress,
            )

            pipeline_state["batch_output_path"] = output_path
            pipeline_state["batch_prediction_count"] = len(results_df)
            pipeline_state["batch_customer_count"] = results_df['CUSTOMER_ID'].nunique() if len(results_df) > 0 else 0
            pipeline_state["batch_status"] = "batch_done"
            pipeline_state["batch_progress_pct"] = 100
        except Exception as e:
            err_msg = f"CRITICAL ERROR: {str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["batch_logs"].append(err_msg)
            pipeline_state["batch_status"] = "error"
            pipeline_state["error"] = str(e)
            pipeline_state["batch_progress_pct"] = 0

    thread = threading.Thread(target=_run_batch, daemon=True)
    thread.start()
    return jsonify({"message": "Batch prediction started."})


@app.route("/api/download_batch")
def api_download_batch():
    """Download the batch predictions CSV."""
    output_path = pipeline_state.get("batch_output_path")
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "No batch predictions available."}), 404

    from flask import send_file
    return send_file(output_path, as_attachment=True, download_name="batch_predictions.csv")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
