#!/usr/bin/env python3
"""
start_pipeline.py
==================
Launches all three services as sub-processes and waits for them.

Usage:
    python start_pipeline.py

Services started:
  - Detector  → http://localhost:5001
  - Grouping  → http://localhost:5002
  - Flask API → http://localhost:5000
"""

import subprocess
import sys
import os
import time
import signal
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE, "yolov8n.pt")

SERVICES = [
    {
        "name": "Detector",
        "cmd": [sys.executable, os.path.join(BASE, "detector_service", "detector.py")],
        "url": "http://localhost:5001/health",
        "env": {
            "DETECTOR_PORT": "5001",
            "YOLO_MODEL": DEFAULT_MODEL_PATH,
            "DETECTOR_CONF_THRESH": "0.12",
            "DETECTOR_MAX_DIM": "1280",
        },
    },
    {
        "name": "Grouping",
        "cmd": [sys.executable, os.path.join(BASE, "grouping_service", "grouping.py")],
        "url": "http://localhost:5002/health",
        "env": {"GROUPING_PORT": "5002"},
    },
    {
        "name": "Flask API",
        "cmd": [sys.executable, os.path.join(BASE, "flask_server", "app.py")],
        "url": "http://localhost:5000/health",
        "env": {
            "FLASK_PORT": "5000",
            "DETECTOR_URL": "http://localhost:5001",
            "GROUPING_URL": "http://localhost:5002",
        },
    },
]

procs = []


def wait_for_service(url: str, name: str, timeout: int = 60) -> bool:
    print(f"  Waiting for {name}...", end="", flush=True)
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(" ✓")
                return True
        except Exception:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print(" ✗ TIMEOUT")
    return False


def shutdown(sig=None, frame=None):
    print("\n\nShutting down services...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main():
    print("=" * 55)
    print("  RetailVision AI Pipeline — Startup")
    print("=" * 55)

    env_base = os.environ.copy()

    for svc in SERVICES:
        print(f"\n▶  Starting {svc['name']}...")
        env = {**env_base, **svc.get("env", {})}
        p = subprocess.Popen(
            svc["cmd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        time.sleep(1.5)  # brief pause between starts

    print()
    all_ok = True
    for svc in SERVICES:
        ok = wait_for_service(svc["url"], svc["name"])
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("=" * 55)
        print("  ✓ All services running!")
        print()
        print("  Web UI  →  http://localhost:5000")
        print("  API     →  POST http://localhost:5000/pipeline")
        print("  Health  →  http://localhost:5000/health")
        print("=" * 55)
        print("\n  Press Ctrl+C to stop all services.\n")

        # Keep alive + stream logs
        try:
            while True:
                for p in procs:
                    line = p.stdout.readline()
                    if line:
                        print(line.decode("utf-8", errors="replace"), end="")
                time.sleep(0.05)
        except KeyboardInterrupt:
            shutdown()
    else:
        print("Some services failed to start. Check logs above.")
        shutdown()


if __name__ == "__main__":
    main()
