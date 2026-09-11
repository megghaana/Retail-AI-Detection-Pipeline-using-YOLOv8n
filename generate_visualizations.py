#!/usr/bin/env python3
"""
generate_visualizations.py
===========================
Standalone script that runs the full detection + grouping + visualization
pipeline on sample images WITHOUT needing the HTTP microservices running.
Saves output images to /outputs/.

Run: python generate_visualizations.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flask_server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detector_service"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "grouping_service"))

import numpy as np
import cv2
import base64
import io
import time
from PIL import Image

# ── inline import of modules ──────────────────────────────────────────────────
from visualize import annotate_and_encode, draw_detections, draw_group_legend

# ── Detector logic (inline, no server needed) ─────────────────────────────────
def run_detection(img_rgb, conf_thresh=0.25):
    """YOLOv8 detection."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model.predict(img_rgb, conf=conf_thresh, verbose=False)[0]
        dets = []
        for i, box in enumerate(results.boxes):
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            dets.append({
                "detection_id": i,
                "bbox": {"x1": x1,"y1": y1,"x2": x2,"y2": y2},
                "confidence": round(conf,4),
                "class_id": cls_id,
                "class_name": cls_name,
            })
        print(f"  YOLO detected {len(dets)} objects")
        return dets
    except Exception as e:
        print(f"  YOLO failed ({e}), using grid fallback")
        return run_grid_detection(img_rgb)


def run_grid_detection(img_rgb):
    """Grid-based fallback detector."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cols = max(3, w // 70)
    rows = max(4, h // 70)
    cell_w, cell_h = w // cols, h // rows
    dets = []
    det_id = 0
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = min(x1+cell_w, w), min(y1+cell_h, h)
            density = float(edges[y1:y2, x1:x2].mean())
            if density > 6:
                dets.append({
                    "detection_id": det_id,
                    "bbox": {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
                    "confidence": round(min(density/80,1.0),4),
                    "class_id": 0,
                    "class_name": "product",
                })
                det_id += 1
    print(f"  Grid detector found {len(dets)} regions")
    return dets


# ── Grouping logic (inline) ───────────────────────────────────────────────────
def run_grouping(img_rgb, detections, eps=0.35):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import hashlib

    GROUP_COLORS = [
        "#E63946","#457B9D","#2DC653","#F4A261","#A8DADC",
        "#6A4C93","#F72585","#4CC9F0","#7B2D8B","#F9C74F",
        "#90BE6D","#43AA8B","#577590","#F94144","#F3722C",
        "#F8961E","#F9844A","#277DA1","#4D908E","#84A98C",
        "#52B788","#B7E4C7","#95D5B2","#74C69D","#40916C",
        "#1B4332","#081C15","#D62828","#023E8A","#0077B6",
    ]

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i+2],16) for i in (0,2,4)]

    h, w = img_rgb.shape[:2]
    features = []
    for det in detections:
        b = det["bbox"]
        x1c,y1c = max(0,b["x1"]),max(0,b["y1"])
        x2c,y2c = min(w,b["x2"]),min(h,b["y2"])
        crop = img_rgb[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            feat = np.zeros(33)
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            bins = 8
            hh = np.histogram(hsv[:,:,0],bins=bins,range=(0,180))[0]
            sh = np.histogram(hsv[:,:,1],bins=bins,range=(0,256))[0]
            vh = np.histogram(hsv[:,:,2],bins=bins,range=(0,256))[0]
            hist = np.concatenate([hh,sh,vh]).astype(float)
            s = hist.sum(); hist /= (s + 1e-6)

            pixels = crop.reshape(-1,3).astype(np.float32)
            if len(pixels) > 500:
                idx = np.random.choice(len(pixels),500,replace=False)
                pixels = pixels[idx]
            k = min(3, len(pixels))
            criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
            _, labels2, centers = cv2.kmeans(pixels,k,None,criteria,3,cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels2.flatten(),minlength=k)
            order = np.argsort(-counts)
            dom = (centers[order].flatten()/255.0)
            if len(dom) < 9: dom = np.pad(dom, (0, 9-len(dom)))

            cx = ((b["x1"]+b["x2"])/2)/w * 0.4
            cy = ((b["y1"]+b["y2"])/2)/h * 0.4
            feat = np.concatenate([hist, dom[:9], [cx, cy]])

        features.append(feat)

    features = np.array(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    db = DBSCAN(eps=eps, min_samples=1, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(scaled)

    color_idx = 0
    group_map = {}
    for lbl in sorted(set(labels)):
        if lbl < 0:
            group_map[lbl] = {"group_id":"group_noise","color_hex":"#AAAAAA"}
        else:
            h_str = hashlib.md5(f"retail_{lbl}".encode()).hexdigest()[:8]
            group_map[lbl] = {
                "group_id": f"group_{h_str}",
                "color_hex": GROUP_COLORS[color_idx % len(GROUP_COLORS)],
            }
            color_idx += 1

    enriched = []
    for det, lbl in zip(detections, labels):
        g = group_map[lbl]
        enriched.append({
            **det,
            "group_id": g["group_id"],
            "group_label": int(lbl),
            "group_color_hex": g["color_hex"],
            "group_color_rgb": hex_to_rgb(g["color_hex"]),
        })

    groups = []
    for lbl, info in group_map.items():
        members = [d["detection_id"] for d, l in zip(enriched, labels) if l == lbl]
        groups.append({
            **info,
            "group_label": int(lbl),
            "member_count": len(members),
            "member_ids": members,
        })

    num_groups = len([g for g in groups if g["group_label"] >= 0])
    print(f"  Grouped into {num_groups} brand groups")
    return enriched, groups


# ── Main ───────────────────────────────────────────────────────────────────────
def process_image(input_path, output_path, conf_thresh=0.25, eps=0.35):
    print(f"\n{'='*55}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*55}")

    img = Image.open(input_path).convert("RGB")
    img_rgb = np.array(img)
    h, w = img_rgb.shape[:2]
    print(f"  Image size: {w}x{h}")

    t0 = time.time()
    dets = run_detection(img_rgb, conf_thresh)
    det_ms = (time.time()-t0)*1000
    print(f"  Detection: {det_ms:.1f}ms")

    t1 = time.time()
    enriched, groups = run_grouping(img_rgb, dets, eps)
    grp_ms = (time.time()-t1)*1000
    print(f"  Grouping: {grp_ms:.1f}ms")

    t2 = time.time()
    annotate_and_encode(img_rgb, enriched, groups, output_path)
    viz_ms = (time.time()-t2)*1000
    total = det_ms + grp_ms + viz_ms
    print(f"  Visualization: {viz_ms:.1f}ms")
    print(f"  TOTAL: {total:.1f}ms")
    print(f"  Saved → {output_path}")

    return enriched, groups, {"detection_ms": det_ms, "grouping_ms": grp_ms, "total_ms": total}


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    images = [
        ("test_shelf.png", "outputs/shelf_visualization.jpg"),
    ]
    for inp, out in images:
        if os.path.exists(inp):
            process_image(inp, out)
        else:
            print(f"Skipping {inp} (not found)")

    print("\n✓ Visualizations saved to outputs/")
