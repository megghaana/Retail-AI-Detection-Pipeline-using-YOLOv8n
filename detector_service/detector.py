"""
Detector Microservice
=====================
Runs a lightweight HTTP server (port 5001) that accepts images and
returns bounding-box detections.

Model choice: YOLOv8n  – the smallest, fastest Ultralytics YOLO variant.
It is pretrained on COCO which includes a broad "object" class pool.
For retail shelf images we detect every object the model sees (any class),
then let the grouping service cluster them by visual similarity.
This avoids needing a custom-trained SKU model while still giving
reasonable box proposals on shelf products.
"""

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import io
import os
import time
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
DEFAULT_CONF_THRESH = float(os.environ.get("DETECTOR_CONF_THRESH", 0.12))
MAX_IMAGE_DIM = int(os.environ.get("DETECTOR_MAX_DIM", 1280))
MAX_BOX_AREA_RATIO = float(os.environ.get("DETECTOR_MAX_BOX_AREA_RATIO", 0.28))
MAX_BOX_WIDTH_RATIO = float(os.environ.get("DETECTOR_MAX_BOX_WIDTH_RATIO", 0.75))
MAX_BOX_HEIGHT_RATIO = float(os.environ.get("DETECTOR_MAX_BOX_HEIGHT_RATIO", 0.6))


# ── Load model once at startup ────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    MODEL_PATH = os.environ.get("YOLO_MODEL", DEFAULT_MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logger.info(f"Loaded YOLO model: {MODEL_PATH}")
    USE_YOLO = True
except Exception as e:
    logger.warning(f"YOLO unavailable ({e}), falling back to selective-search detector")
    MODEL_PATH = "grid-fallback"
    USE_YOLO = False


def decode_image(payload: dict) -> np.ndarray:
    """Accept base64-encoded image or raw bytes from multipart."""
    b64 = payload.get("image_base64")
    if b64:
        img_bytes = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil)
    raise ValueError("No image data found in payload")


def resize_for_detection(img_rgb: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> tuple:
    """Shrink large shelf images to improve latency while preserving aspect ratio."""
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img_rgb, 1.0

    scale = max_dim / float(longest)
    resized = cv2.resize(
        img_rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def detect_yolo(img_rgb: np.ndarray, conf_thresh: float = DEFAULT_CONF_THRESH) -> list:
    """Run YOLOv8 and return list of detection dicts."""
    infer_img, scale = resize_for_detection(img_rgb)
    results = model.predict(
        infer_img,
        conf=conf_thresh,
        imgsz=min(MAX_IMAGE_DIM, max(infer_img.shape[:2])),
        verbose=False,
    )[0]
    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if scale != 1.0:
            x1, y1, x2, y2 = [coord / scale for coord in (x1, y1, x2, y2)]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append({
            "detection_id": i,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "confidence": round(conf, 4),
            "class_id": cls_id,
            "class_name": "product",
        })
    return detections


def filter_implausible_boxes(
    detections: list,
    image_shape: tuple,
    max_area_ratio: float = MAX_BOX_AREA_RATIO,
    max_width_ratio: float = MAX_BOX_WIDTH_RATIO,
    max_height_ratio: float = MAX_BOX_HEIGHT_RATIO,
) -> list:
    """
    Remove shelf-sized boxes from generic YOLO output so grouping sees
    product-like crops rather than whole shelf regions.
    """
    if not detections:
        return detections

    h, w = image_shape[:2]
    image_area = max(h * w, 1)
    filtered = []

    for det in detections:
        b = det["bbox"]
        box_w = max(0, b["x2"] - b["x1"])
        box_h = max(0, b["y2"] - b["y1"])
        area_ratio = (box_w * box_h) / image_area
        width_ratio = box_w / max(w, 1)
        height_ratio = box_h / max(h, 1)

        if area_ratio > max_area_ratio:
            continue
        if width_ratio > max_width_ratio:
            continue
        if height_ratio > max_height_ratio:
            continue

        filtered.append(det)

    for new_id, det in enumerate(filtered):
        det["detection_id"] = new_id

    return filtered


def detect_grid_fallback(img_rgb: np.ndarray) -> list:
    """
    Fallback: sliding-window grid segmentation to get rough product regions.
    Divides the image into a grid and uses edge density to identify product cells.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Estimate number of rows/cols based on image aspect ratio
    cols = max(3, w // 80)
    rows = max(4, h // 80)
    cell_w = w // cols
    cell_h = h // rows

    detections = []
    det_id = 0
    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = min(x1 + cell_w, w)
            y2 = min(y1 + cell_h, h)
            cell_edges = edges[y1:y2, x1:x2]
            density = float(cell_edges.mean())
            if density > 8:  # threshold: only keep cells with content
                detections.append({
                    "detection_id": det_id,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": round(min(density / 80.0, 1.0), 4),
                    "class_id": 0,
                    "class_name": "product",
                })
                det_id += 1
    return detections


def merge_overlapping_boxes(detections: list, iou_thresh: float = 0.5) -> list:
    """Simple NMS-style box merger to reduce duplicates."""
    if not detections:
        return detections

    boxes = np.array([
        [d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]]
        for d in detections
    ], dtype=float)
    scores = np.array([d["confidence"] for d in detections])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                 (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_j - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]

    merged = []
    for new_id, old_id in enumerate(keep):
        d = detections[old_id].copy()
        d["detection_id"] = new_id
        merged.append(d)
    return merged


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH if USE_YOLO else "grid-fallback",
        "default_conf_thresh": DEFAULT_CONF_THRESH,
        "max_image_dim": MAX_IMAGE_DIM,
    })


@app.route("/detect", methods=["POST"])
def detect():
    t0 = time.time()
    try:
        payload = request.get_json(force=True)
        img = decode_image(payload)
        conf_thresh = float(payload.get("conf_thresh", DEFAULT_CONF_THRESH))

        if USE_YOLO:
            detections = detect_yolo(img, conf_thresh=conf_thresh)
        else:
            detections = detect_grid_fallback(img)

        detections = filter_implausible_boxes(detections, img.shape)
        detections = merge_overlapping_boxes(detections)

        h, w = img.shape[:2]
        elapsed = round((time.time() - t0) * 1000, 1)
        return jsonify({
            "status": "ok",
            "image_size": {"width": w, "height": h},
            "num_detections": len(detections),
            "detections": detections,
            "latency_ms": elapsed,
            "model": MODEL_PATH if USE_YOLO else "grid-fallback",
        })

    except Exception as e:
        logger.exception("Detection failed")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("DETECTOR_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
