"""
Flask Orchestration Server
===========================
Port 5000 – public-facing entry point.

Responsibilities:
  1. Accept image upload (multipart/form-data or JSON with base64).
  2. Forward image to Detector service (port 5001).
  3. Forward detections + image to Grouping service (port 5002).
  4. Run visualization, save annotated image to /outputs/.
  5. Return final JSON response to client.

Scalability notes:
  - Each microservice is a separate process → can be scaled independently.
  - Flask uses threaded=True (or gunicorn workers in production).
  - Async fan-out to services can be added with concurrent.futures if needed.
  - Redis queue / Celery can replace direct HTTP calls for heavy load.
"""

import os
import sys
import time
import uuid
import base64
import io
import logging
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np

# Make visualize importable from same package
sys.path.insert(0, os.path.dirname(__file__))
from visualize import annotate_and_encode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask_server")

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ── Service URLs (overridable via env) ────────────────────────────────────────
DETECTOR_URL = os.environ.get("DETECTOR_URL", "http://localhost:5001")
GROUPING_URL = os.environ.get("GROUPING_URL", "http://localhost:5002")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Request timeout (seconds)
SERVICE_TIMEOUT = int(os.environ.get("SERVICE_TIMEOUT", 60))


def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def call_detector(image_b64: str, conf_thresh: float = 0.12) -> dict:
    resp = requests.post(
        f"{DETECTOR_URL}/detect",
        json={"image_base64": image_b64, "conf_thresh": conf_thresh},
        timeout=SERVICE_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def call_grouper(image_b64: str, detections: list, eps: float = 0.9) -> dict:
    resp = requests.post(
        f"{GROUPING_URL}/group",
        json={"image_base64": image_b64, "detections": detections, "eps": eps},
        timeout=SERVICE_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def load_image_from_request() -> tuple:
    """
    Returns (img_bytes, image_b64, img_rgb_array).
    Accepts:
      - multipart/form-data with key 'image'
      - JSON body with key 'image_base64'
    """
    if "image" in request.files:
        f = request.files["image"]
        img_bytes = f.read()
    elif request.is_json and "image_base64" in request.get_json():
        img_bytes = base64.b64decode(request.get_json()["image_base64"])
    else:
        raise ValueError("No image provided. Use multipart 'image' field or JSON 'image_base64'.")

    img_b64 = image_to_base64(img_bytes)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(pil)
    return img_bytes, img_b64, img_rgb


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    services = {}
    for name, url in [("detector", DETECTOR_URL), ("grouping", GROUPING_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            services[name] = r.json()
        except Exception as e:
            services[name] = {"status": "unreachable", "error": str(e)}
    return jsonify({"status": "ok", "services": services})


@app.route("/pipeline", methods=["POST"])
def pipeline():
    """
    Main pipeline endpoint.

    Input (multipart form OR JSON):
      - image          : image file (multipart)
      - image_base64   : base64-encoded image (JSON)
      - conf_thresh    : float, detection confidence threshold (default 0.12)
      - eps            : float, DBSCAN epsilon for grouping (default 0.9)
      - save_viz       : bool, save visualization to disk (default true)

    Output JSON:
    {
      "status": "ok",
      "request_id": "<uuid>",
      "image_size": {"width": W, "height": H},
      "num_detections": N,
      "num_groups": K,
      "detections": [
        {
          "detection_id": 0,
          "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
          "confidence": float,
          "class_name": str,
          "group_id": str,
          "group_label": int,
          "group_color_hex": str,
          "group_color_rgb": [R, G, B]
        }, ...
      ],
      "groups": [
        {
          "group_id": str,
          "group_label": int,
          "color_hex": str,
          "member_count": int,
          "member_ids": [int, ...]
        }, ...
      ],
      "visualization": {
        "image_base64": "<base64 JPEG>",
        "saved_path": "outputs/<filename>.jpg"  // if saved
      },
      "latency": {
        "detection_ms": float,
        "grouping_ms": float,
        "visualization_ms": float,
        "total_ms": float
      }
    }
    """
    request_id = str(uuid.uuid4())[:8]
    t_total = time.time()

    try:
        # ── Parse parameters ──────────────────────────────────────────────────
        conf_thresh = float(request.form.get("conf_thresh",
                            request.args.get("conf_thresh", 0.12)))
        eps = float(request.form.get("eps",
                    request.args.get("eps", 0.9)))
        save_viz = request.form.get("save_viz", "true").lower() != "false"

        # ── Load image ────────────────────────────────────────────────────────
        _, img_b64, img_rgb = load_image_from_request()
        h, w = img_rgb.shape[:2]
        logger.info(f"[{request_id}] Image loaded: {w}x{h}")

        # ── Step 2: Detection ─────────────────────────────────────────────────
        t_det = time.time()
        det_result = call_detector(img_b64, conf_thresh=conf_thresh)
        detection_ms = round((time.time() - t_det) * 1000, 1)

        if det_result.get("status") != "ok":
            raise RuntimeError(f"Detector error: {det_result.get('message')}")

        detections = det_result["detections"]
        logger.info(f"[{request_id}] Detections: {len(detections)}")

        # ── Step 3: Grouping ──────────────────────────────────────────────────
        t_grp = time.time()
        grp_result = call_grouper(img_b64, detections, eps=eps)
        grouping_ms = round((time.time() - t_grp) * 1000, 1)

        if grp_result.get("status") != "ok":
            raise RuntimeError(f"Grouper error: {grp_result.get('message')}")

        enriched_detections = grp_result["detections"]
        groups = grp_result["groups"]
        logger.info(f"[{request_id}] Groups: {grp_result['num_groups']}")

        # ── Step 4: Visualization ─────────────────────────────────────────────
        t_viz = time.time()
        out_filename = f"result_{request_id}_{int(time.time())}.jpg"
        out_path = os.path.join(OUTPUTS_DIR, out_filename) if save_viz else None

        viz_b64 = annotate_and_encode(img_rgb, enriched_detections, groups, out_path)
        viz_ms = round((time.time() - t_viz) * 1000, 1)

        total_ms = round((time.time() - t_total) * 1000, 1)
        logger.info(f"[{request_id}] Done in {total_ms}ms")

        # ── Build response ────────────────────────────────────────────────────
        response = {
            "status": "ok",
            "request_id": request_id,
            "image_size": {"width": w, "height": h},
            "num_detections": len(enriched_detections),
            "num_groups": grp_result["num_groups"],
            "detections": enriched_detections,
            "groups": groups,
            "visualization": {
                "image_base64": viz_b64,
                "saved_path": f"outputs/{out_filename}" if save_viz else None,
            },
            "latency": {
                "detection_ms": detection_ms,
                "grouping_ms": grouping_ms,
                "visualization_ms": viz_ms,
                "total_ms": total_ms,
            },
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"status": "error", "request_id": request_id, "message": str(e)}), 400
    except requests.exceptions.ConnectionError as e:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "message": f"Microservice unreachable: {str(e)}. Ensure detector and grouping services are running.",
        }), 503
    except Exception as e:
        logger.exception(f"[{request_id}] Pipeline error")
        return jsonify({"status": "error", "request_id": request_id, "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
