"""
FinPersona Unified Gateway
==========================
Single Flask app that serves the unified UI and proxies API calls
to the three module servers:
  - Credit Risk   → localhost:5005
  - Churn          → localhost:5006
  - Recommendation → localhost:5000
"""

import os
import subprocess
import sys

import requests as http_requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# ── Configuration ──
CREDIT_URL = "http://127.0.0.1:5005"
CHURN_URL = "http://127.0.0.1:5006"
RECOMMEND_URL = "http://127.0.0.1:5000"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assests")

app = Flask(__name__)


# ── Static routes ──


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/assets/logo.png")
def serve_logo():
    return send_from_directory(ASSETS_DIR, "logo.PNG")


# ── Browse endpoint (runs on the gateway itself) ──


@app.route("/api/browse", methods=["POST"])
def api_browse():
    """Open a native Windows folder picker dialog."""
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
        return jsonify({"path": ""})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Dialog timed out."}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Proxy helpers ──

BACKEND_MAP = {
    "credit": CREDIT_URL,
    "churn": CHURN_URL,
    "recommend": RECOMMEND_URL,
}

# Headers we pass through
HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ]
)


def _proxy(module: str, path: str):
    """Forward the current Flask request to the backend module."""
    base = BACKEND_MAP.get(module)
    if not base:
        return jsonify({"error": f"Unknown module: {module}"}), 404

    url = f"{base}/{path}"

    # Build headers (skip hop-by-hop)
    headers = {k: v for k, v in request.headers if k.lower() not in HOP_BY_HOP}
    headers.pop("Host", None)

    try:
        resp = http_requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            params=request.args,
            timeout=300,
            stream=True,
        )
    except http_requests.ConnectionError:
        return (
            jsonify(
                {"error": f"Cannot reach {module} server at {base}. Is it running?"}
            ),
            502,
        )

    # Build response
    excluded = HOP_BY_HOP | {"content-encoding", "content-length"}
    response_headers = [
        (k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded
    ]

    return Response(
        resp.content,
        status=resp.status_code,
        headers=response_headers,
    )


# ── Proxy routes ──


@app.route("/credit/<path:path>", methods=["GET", "POST"])
def proxy_credit(path):
    return _proxy("credit", path)


@app.route("/churn/<path:path>", methods=["GET", "POST"])
def proxy_churn(path):
    return _proxy("churn", path)


@app.route("/recommend/<path:path>", methods=["GET", "POST"])
def proxy_recommend(path):
    return _proxy("recommend", path)


if __name__ == "__main__":
    app.run(debug=False, port=5050)
