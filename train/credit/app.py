"""
app.py
======
Flask GUI for the Credit Risk Prediction Pipeline.
Provides a premium web interface to:
  1. Load & engineer features from prime + transaction data
  2. Train the XGBoost credit default model (default or tuned)
  3. Score customers and view risk predictions
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import threading
import traceback
import io

# Ensure the credit module directory is on sys.path so local imports work
CREDIT_DIR = os.path.dirname(os.path.abspath(__file__))
if CREDIT_DIR not in sys.path:
    sys.path.insert(0, CREDIT_DIR)

import config

app = Flask(__name__)

# ── Persistence paths ──
SAVE_DIR = os.path.join(config.OUTPUT_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_JSON = os.path.join(SAVE_DIR, "credit_app_state.json")

# ── Global pipeline state ──
pipeline_state = {
    "status": "idle",           # idle | loading | loaded | training | trained | scoring | scored | error
    "loading_logs": [],
    "training_logs": [],
    "scoring_logs": [],
    "metrics": None,
    "error": None,
    "progress_pct": 0,
    # Data stats
    "total_samples": 0,
    "n_features": 0,
    "default_rate": 0.0,
    "n_default": 0,
    "n_non_default": 0,
    # Training results
    "threshold": None,
    "model_params": None,
    # Scoring results
    "n_scored": 0,
    "n_flagged": 0,
    "flagged_pct": 0.0,
}


def _save_state_meta():
    """Persist lightweight state (no large objects) for UI reload."""
    saveable = {k: v for k, v in pipeline_state.items()
                if k not in ("df", "X", "y", "model")}
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

    # Check if a trained model exists
    if os.path.exists(config.MODEL_PATH):
        pipeline_state["status"] = max(
            pipeline_state["status"],
            "trained",
            key=lambda s: ["idle", "loading", "loaded", "training", "trained", "scoring", "scored", "error"].index(s)
            if s in ["idle", "loading", "loaded", "training", "trained", "scoring", "scored", "error"] else 0
        )
        pipeline_state["training_logs"] = pipeline_state.get("training_logs") or [
            "[Loaded from cache] Trained model found on disk."
        ]

    # Check if scores exist
    if os.path.exists(config.SCORES_PATH):
        pipeline_state["scoring_logs"] = pipeline_state.get("scoring_logs") or [
            f"[Loaded from cache] Scores found at {config.SCORES_PATH}"
        ]

    # Check if report exists
    if os.path.exists(config.REPORT_PATH):
        try:
            with open(config.REPORT_PATH, "r") as f:
                report_text = f.read()
            pipeline_state["report_text"] = report_text
        except Exception:
            pass


_load_state_meta()


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return current pipeline state (polled by the frontend)."""
    return jsonify({
        "status": pipeline_state["status"],
        "loading_logs": pipeline_state["loading_logs"],
        "training_logs": pipeline_state["training_logs"],
        "scoring_logs": pipeline_state["scoring_logs"],
        "metrics": pipeline_state["metrics"],
        "error": pipeline_state["error"],
        "progress_pct": pipeline_state["progress_pct"],
        "total_samples": pipeline_state["total_samples"],
        "n_features": pipeline_state["n_features"],
        "default_rate": pipeline_state["default_rate"],
        "n_default": pipeline_state["n_default"],
        "n_non_default": pipeline_state["n_non_default"],
        "threshold": pipeline_state["threshold"],
        "model_params": pipeline_state["model_params"],
        "n_scored": pipeline_state["n_scored"],
        "n_flagged": pipeline_state["n_flagged"],
        "flagged_pct": pipeline_state["flagged_pct"],
        "report_text": pipeline_state.get("report_text", ""),
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    """Kick off the full training pipeline in a background thread."""
    if pipeline_state["status"] in ("training", "loading", "scoring"):
        return jsonify({"error": "A pipeline step is already running."}), 400

    data = request.json or {}
    tune = data.get("tune", False)
    sample = data.get("sample", False)

    # Reset state
    pipeline_state.update({
        "status": "training",
        "loading_logs": [],
        "training_logs": [],
        "scoring_logs": [],
        "metrics": None,
        "error": None,
        "progress_pct": 5,
        "total_samples": 0,
        "n_features": 0,
        "default_rate": 0.0,
        "n_default": 0,
        "n_non_default": 0,
        "threshold": None,
        "model_params": None,
        "n_scored": 0,
        "n_flagged": 0,
        "flagged_pct": 0.0,
        "report_text": "",
    })

    def _run():
        try:
            # Capture stdout to stream as logs
            import pipeline as credit_pipeline

            logs = []

            def log(msg):
                logs.append(msg)
                pipeline_state["training_logs"] = list(logs)

            log(f"[START] {'Tuned' if tune else 'Default'} training pipeline ...")
            if sample:
                log("[INFO] Using 25% stratified sample for fast iteration.")

            pipeline_state["progress_pct"] = 10
            log("[STEP 1/13] Loading data ...")

            from data_loader import load_prime_data, load_transaction_data, merge_data
            prime_df = load_prime_data()
            txn_df = load_transaction_data()
            log(f"  Prime rows: {len(prime_df):,}")
            log(f"  Transaction rows: {len(txn_df):,}")

            pipeline_state["progress_pct"] = 20
            log("[STEP 2/13] Feature engineering ...")

            from feature_engineering import (
                engineer_prime_features, engineer_transaction_features,
                engineer_temporal_features, create_target,
            )
            prime_df = engineer_prime_features(prime_df)
            txn_features = engineer_transaction_features(txn_df)

            pipeline_state["progress_pct"] = 30
            log("[STEP 3/13] Merging data ...")
            merged = merge_data(prime_df, txn_features)
            log(f"  Merged shape: {merged.shape}")

            pipeline_state["progress_pct"] = 35
            log("[STEP 4/13] Creating target variable ...")
            merged = create_target(merged)

            pipeline_state["progress_pct"] = 40
            log("[STEP 5/13] Temporal feature engineering ...")
            merged = engineer_temporal_features(merged)

            target = merged[config.TARGET_COL]
            n_default = int(target.sum())
            n_total = len(target)
            pipeline_state["total_samples"] = n_total
            pipeline_state["n_default"] = n_default
            pipeline_state["n_non_default"] = n_total - n_default
            pipeline_state["default_rate"] = round(n_default / n_total * 100, 1) if n_total > 0 else 0

            log(f"  Total: {n_total:,} | Default: {n_default:,} ({pipeline_state['default_rate']}%)")

            pipeline_state["progress_pct"] = 50
            log("[STEP 6-13] Running full pipeline ...")

            # Now run the actual pipeline
            metrics = credit_pipeline.run_training_pipeline(tune=tune, sample=sample)

            pipeline_state["metrics"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
            }
            pipeline_state["threshold"] = metrics.get("Threshold")
            pipeline_state["progress_pct"] = 95

            # Load the report if it was written
            if os.path.exists(config.REPORT_PATH):
                with open(config.REPORT_PATH, "r") as f:
                    pipeline_state["report_text"] = f.read()
                log(f"[DONE] Report saved to {config.REPORT_PATH}")

            # Load scores summary
            if os.path.exists(config.SCORES_PATH):
                import pandas as pd
                scores = pd.read_csv(config.SCORES_PATH)
                n_flagged = int(scores["predicted_label"].sum())
                pipeline_state["n_scored"] = len(scores)
                pipeline_state["n_flagged"] = n_flagged
                pipeline_state["flagged_pct"] = round(n_flagged / len(scores) * 100, 1)
                log(f"[DONE] Scored {len(scores):,} customers — {n_flagged:,} flagged as default.")

            pipeline_state["status"] = "trained"
            pipeline_state["progress_pct"] = 100
            log("[COMPLETE] Training pipeline finished successfully.")
            _save_state_meta()

        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = f"{str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["progress_pct"] = 0
            pipeline_state["training_logs"].append(f"[ERROR] {str(e)}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"message": "Training pipeline started."})


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score new data using the saved model."""
    if not os.path.exists(config.MODEL_PATH):
        return jsonify({"error": "No trained model found. Train the model first."}), 400
    if pipeline_state["status"] in ("training", "scoring"):
        return jsonify({"error": "A pipeline step is already running."}), 400

    data = request.json or {}
    prime_dir = data.get("prime_dir", "").strip() or None
    txn_dir = data.get("txn_dir", "").strip() or None

    pipeline_state.update({
        "status": "scoring",
        "scoring_logs": [],
        "error": None,
        "progress_pct": 10,
    })

    def _run():
        try:
            import pipeline as credit_pipeline
            logs = []

            def log(msg):
                logs.append(msg)
                pipeline_state["scoring_logs"] = list(logs)

            log("[START] Scoring pipeline ...")
            if prime_dir:
                log(f"  Prime dir: {prime_dir}")
            if txn_dir:
                log(f"  Transaction dir: {txn_dir}")
            pipeline_state["progress_pct"] = 30

            scores_df = credit_pipeline.run_scoring_pipeline(
                prime_dir=prime_dir,
                txn_dir=txn_dir,
            )

            n_flagged = int(scores_df["predicted_label"].sum())
            pipeline_state["n_scored"] = len(scores_df)
            pipeline_state["n_flagged"] = n_flagged
            pipeline_state["flagged_pct"] = round(n_flagged / len(scores_df) * 100, 1)

            log(f"[DONE] Scored {len(scores_df):,} customers.")
            log(f"  Flagged as default: {n_flagged:,} ({pipeline_state['flagged_pct']}%)")

            pipeline_state["status"] = "scored"
            pipeline_state["progress_pct"] = 100
            _save_state_meta()

        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = f"{str(e)}\n\n{traceback.format_exc()}"
            pipeline_state["progress_pct"] = 0

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"message": "Scoring pipeline started."})


@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    """Look up a customer's risk score from the scored CSV."""
    if not os.path.exists(config.SCORES_PATH):
        return jsonify({"error": "No scores available. Run training or scoring first."}), 400

    data = request.json or {}
    customer_id_raw = str(data.get("customer_id", "")).strip()
    if not customer_id_raw:
        return jsonify({"error": "Please provide a Customer ID."}), 400

    try:
        customer_id = int(customer_id_raw)
    except ValueError:
        return jsonify({"error": f"Invalid Customer ID: '{customer_id_raw}'. Must be an integer."}), 400

    try:
        import pandas as pd
        scores = pd.read_csv(config.SCORES_PATH)
        match = scores[scores[config.CUSTOMER_ID] == customer_id]

        if match.empty:
            return jsonify({"error": f"Customer ID {customer_id} not found in scores."}), 404

        row = match.iloc[0]
        prob = round(float(row["default_probability"]) * 100, 2)
        label = int(row["predicted_label"])

        # Determine risk tier
        if prob >= 70:
            risk_tier = "Critical"
        elif prob >= 40:
            risk_tier = "High"
        elif prob >= 20:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        return jsonify({
            "customer_id": customer_id,
            "default_probability": prob,
            "predicted_label": label,
            "risk_tier": risk_tier,
            "label_text": "Default" if label == 1 else "Non-Default",
        })
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
        ids = scores[config.CUSTOMER_ID].dropna().astype(int).tolist()
        if q:
            ids = [cid for cid in ids if q in str(cid)]
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


@app.route("/api/browse")
def api_browse():
    """List contents of a directory for the folder browser.

    Query params:
        path  – absolute or relative directory to list (defaults to project root)
    """
    raw_path = request.args.get("path", "").strip()

    if not raw_path:
        browse_dir = config.PROJECT_ROOT
    else:
        # Resolve relative paths against the project root
        browse_dir = os.path.abspath(raw_path) if os.path.isabs(raw_path) else os.path.abspath(
            os.path.join(config.PROJECT_ROOT, raw_path)
        )

    if not os.path.isdir(browse_dir):
        return jsonify({"error": f"Not a directory: {browse_dir}"}), 400

    entries = []
    try:
        for name in sorted(os.listdir(browse_dir)):
            full = os.path.join(browse_dir, name)
            entries.append({
                "name": name,
                "path": full.replace("\\", "/"),
                "is_dir": os.path.isdir(full),
            })
    except PermissionError:
        return jsonify({"error": f"Permission denied: {browse_dir}"}), 403

    # Sort: directories first, then files
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

    parent = os.path.dirname(browse_dir)
    return jsonify({
        "current": browse_dir.replace("\\", "/"),
        "parent": parent.replace("\\", "/") if parent != browse_dir else None,
        "entries": entries,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5005)
