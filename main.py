"""
FinPersona — Unified Launcher
==============================
Starts all 4 servers (Credit Risk, Churn, Recommendation, Gateway)
and opens the browser to the unified UI.

Usage:
    python start.py
"""

import os
import sys
import time
import subprocess
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SERVERS = [
    {
        "name": "Recommendation",
        "cwd": os.path.join(PROJECT_ROOT, "train", "recommendation"),
        "script": "app.py",
        "port": 5000,
    },
    {
        "name": "Credit Risk",
        "cwd": os.path.join(PROJECT_ROOT, "train", "credit"),
        "script": "app.py",
        "port": 5005,
    },
    {
        "name": "Churn",
        "cwd": os.path.join(PROJECT_ROOT, "train", "churn"),
        "script": "app.py",
        "port": 5006,
    },
    {
        "name": "Gateway (UI)",
        "cwd": os.path.join(PROJECT_ROOT, "UI"),
        "script": "app.py",
        "port": 5050,
    },
]


def main():
    procs = []

    print("=" * 55)
    print("  F I N P E R S O N A   —   Starting All Servers")
    print("=" * 55)
    print()

    for srv in SERVERS:
        script_path = os.path.join(srv["cwd"], srv["script"])
        if not os.path.exists(script_path):
            print(f"  [SKIP] {srv['name']}: {script_path} not found")
            continue

        print(f"  Starting {srv['name']} on port {srv['port']}...")
        proc = subprocess.Popen(
            [sys.executable, srv["script"]],
            cwd=srv["cwd"],
        )
        procs.append((srv["name"], proc))
        print(f"    -> PID {proc.pid}")

    print()
    print("  Waiting for servers to initialize...")
    time.sleep(3)

    gateway_url = f"http://127.0.0.1:{SERVERS[-1]['port']}"
    print(f"\n  Opening {gateway_url} in your browser...")
    webbrowser.open(gateway_url)

    print()
    print("=" * 55)
    print("  All servers running. Press Ctrl+C to stop all.")
    print("=" * 55)
    print()

    try:
        # Keep main process alive
        while True:
            time.sleep(1)
            # Check if any process died
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"  [WARNING] {name} exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\n  Shutting down all servers...")
        for name, proc in procs:
            proc.terminate()
            print(f"  Stopped {name} (PID {proc.pid})")
        print("  Done.")


if __name__ == "__main__":
    main()
