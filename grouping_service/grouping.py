"""
Product Grouping Microservice
==============================
Runs on port 5002.

Algorithm:
    1. Accept detected bounding boxes + original image (base64).
    2. Crop each bounding box from the image.
    3. Extract a compact color-histogram + spatial-position feature vector
       for each crop (no heavy model required → low latency).
    4. Cluster features with DBSCAN (density-based, no need to pre-specify K).
       Products that share a similar color palette AND are spatially nearby
       are considered the same brand group (e.g., all green Head & Shoulders
       bottles on one shelf row form one cluster).
    5. Assign a stable group_id (UUID-based) to each cluster.
    6. Return enriched detections with group_id + group_color (for visualization).

Why color histograms?
  - Very fast: microseconds per crop.
  - Brand packaging has highly consistent dominant colors (logo colors,
    bottle colors). This gives a reliable grouping signal without a GPU.
  - Spatial proximity is a secondary feature: on a real shelf, same-brand
    products appear together. Adding (x_center/W, y_center/H) as features
    helps separate spatially distinct brand sections.
"""

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import io
import time
import os
import hashlib
import logging
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grouping")

app = Flask(__name__)

# Deterministic color palette for group visualization (up to 30 groups)
GROUP_COLORS = [
    "#E63946", "#457B9D", "#2DC653", "#F4A261", "#A8DADC",
    "#6A4C93", "#F72585", "#4CC9F0", "#7B2D8B", "#F9C74F",
    "#90BE6D", "#43AA8B", "#577590", "#F94144", "#F3722C",
    "#F8961E", "#F9844A", "#277DA1", "#4D908E", "#84A98C",
    "#52B788", "#B7E4C7", "#95D5B2", "#74C69D", "#40916C",
    "#1B4332", "#081C15", "#D62828", "#023E8A", "#0077B6",
]

NOISE_COLOR = "#AAAAAA"  # for DBSCAN noise points (group_id = -1)


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def decode_image(b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)


def extract_color_histogram(crop_rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Compute a compact color histogram in HSV space.
    Using HSV makes the descriptor robust to lighting variation.
    Returns a 1-D normalized feature vector of length bins*3.
    """
    if crop_rgb.size == 0:
        return np.zeros(bins * 3)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(crop_hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(crop_hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(crop_hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(float)
    norm = feat.sum()
    if norm > 0:
        feat /= norm
    return feat


def build_feature_matrix(detections: list, img_rgb: np.ndarray,
                          spatial_weight: float = 0.02) -> np.ndarray:
    """
    For each detection, build a feature vector:
      [color_histogram (24-d), crop_aspect_ratio (1-d), crop_area_ratio (1-d),
       norm_cx (1-d), norm_cy (1-d)]
    This keeps grouping broad and stable for the demo by emphasizing overall
    packaging color and only lightly considering shelf position.
    """
    h, w = img_rgb.shape[:2]
    features = []
    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        # Clamp to image bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = img_rgb[y1c:y2c, x1c:x2c]

        hist_feat = extract_color_histogram(crop, bins=8)    # 24-d
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        shape_feat = np.array([
            box_w / max(box_h, 1),
            (box_w * box_h) / max(w * h, 1),
        ])

        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        spatial = np.array([cx * spatial_weight, cy * spatial_weight])

        feat = np.concatenate([hist_feat, shape_feat, spatial])
        features.append(feat)

    return np.array(features, dtype=float)


def cluster_detections(features: np.ndarray,
                        eps: float = 0.9,
                        min_samples: int = 1) -> np.ndarray:
    """
    DBSCAN clustering on standardized feature vectors.
    min_samples=1 means every point is at least its own cluster (no noise by
    default on small sets). Returns array of integer cluster labels.
    """
    if len(features) == 0:
        return np.array([], dtype=int)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(scaled)
    return labels


def label_to_group_id(label: int, seed: str = "retail") -> str:
    """Convert an integer cluster label to a stable short UUID-style group id."""
    if label < 0:
        return "group_noise"
    raw = f"{seed}_{label}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"group_{h}"


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "algorithm": "DBSCAN+ColorHistogram"})


@app.route("/group", methods=["POST"])
def group():
    t0 = time.time()
    try:
        payload = request.get_json(force=True)
        image_b64 = payload.get("image_base64")
        detections = payload.get("detections", [])
        eps = float(payload.get("eps", 0.9))

        if not image_b64:
            return jsonify({"status": "error", "message": "image_base64 required"}), 400

        img = decode_image(image_b64)

        if not detections:
            return jsonify({
                "status": "ok",
                "groups": [],
                "detections": [],
                "num_groups": 0,
                "latency_ms": 0,
            })

        features = build_feature_matrix(detections, img)
        labels = cluster_detections(features, eps=eps)

        # Build unique group registry
        unique_labels = sorted(set(labels))
        group_registry = {}
        color_idx = 0
        for lbl in unique_labels:
            gid = label_to_group_id(lbl)
            color = NOISE_COLOR if lbl < 0 else GROUP_COLORS[color_idx % len(GROUP_COLORS)]
            if lbl >= 0:
                color_idx += 1
            group_registry[lbl] = {
                "group_id": gid,
                "group_label": int(lbl),
                "color_hex": color,
                "color_rgb": list(hex_to_rgb(color)),
            }

        # Enrich detections
        enriched = []
        for det, lbl in zip(detections, labels):
            g = group_registry[lbl]
            enriched.append({
                **det,
                "group_id": g["group_id"],
                "group_label": g["group_label"],
                "group_color_hex": g["color_hex"],
                "group_color_rgb": g["color_rgb"],
            })

        groups = []
        for lbl, info in group_registry.items():
            members = [d["detection_id"] for d, l in zip(enriched, labels) if l == lbl]
            groups.append({**info, "member_count": len(members), "member_ids": members})

        elapsed = round((time.time() - t0) * 1000, 1)
        return jsonify({
            "status": "ok",
            "num_groups": len([g for g in groups if g["group_label"] >= 0]),
            "groups": groups,
            "detections": enriched,
            "latency_ms": elapsed,
        })

    except Exception as e:
        logger.exception("Grouping failed")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("GROUPING_PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
