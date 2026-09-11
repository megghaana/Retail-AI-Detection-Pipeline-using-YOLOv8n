"""
Visualization Utility
=====================
Draws color-coded bounding boxes and group labels on shelf images.
Used by the Flask server after receiving grouped detections.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import time


def hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)  # OpenCV is BGR


def draw_detections(
    img_rgb: np.ndarray,
    detections: list,
    show_confidence: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Draw color-coded bounding boxes on the image.
    Each group gets a unique color. Label shows group_id abbreviation + confidence.
    Returns annotated BGR image (numpy array).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color_hex = det.get("group_color_hex", "#AAAAAA")
        group_id = det.get("group_id", "?")
        conf = det.get("confidence", 0.0)
        color_bgr = hex_to_bgr(color_hex)

        # Draw filled semi-transparent rectangle
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.addWeighted(overlay, 0.15, img_bgr, 0.85, 0, img_bgr)

        # Draw border
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, box_thickness)

        # Label text
        short_id = group_id.replace("group_", "G")[:6]
        label = f"{short_id}"
        if show_confidence:
            label += f" {conf:.2f}"

        # Background for text
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            img_bgr,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + baseline),
            color_bgr,
            -1,
        )
        cv2.putText(
            img_bgr,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img_bgr


def draw_group_legend(img_bgr: np.ndarray, groups: list) -> np.ndarray:
    """
    Append a color-coded legend panel to the right of the image.
    """
    h, w = img_bgr.shape[:2]
    legend_w = 160
    legend = np.ones((h, legend_w, 3), dtype=np.uint8) * 30  # dark background

    y_offset = 20
    cv2.putText(legend, "GROUPS", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    y_offset += 20

    visible_groups = [g for g in groups if g.get("group_label", -1) >= 0]
    for g in visible_groups[:30]:  # max 30 in legend
        color_hex = g.get("color_hex", "#888888")
        color_bgr = hex_to_bgr(color_hex)
        gid = g.get("group_id", "?").replace("group_", "G")[:8]
        cnt = g.get("member_count", 0)

        cv2.rectangle(legend, (8, y_offset - 10), (22, y_offset + 2), color_bgr, -1)
        cv2.putText(
            legend,
            f"{gid} ({cnt})",
            (28, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y_offset += 18
        if y_offset > h - 20:
            break

    combined = np.hstack([img_bgr, legend])
    return combined


def annotate_and_encode(
    img_rgb: np.ndarray,
    detections: list,
    groups: list,
    output_path: str = None,
) -> str:
    """
    Full pipeline: draw boxes + legend, save to file, return base64 JPEG string.
    """
    annotated = draw_detections(img_rgb, detections)
    if groups:
        annotated = draw_group_legend(annotated, groups)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, annotated)

    # Encode to base64 JPEG for JSON response
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def create_summary_card(groups: list, num_detections: int, latency_ms: float) -> np.ndarray:
    """
    Create a small summary statistics card image.
    """
    card = np.ones((120, 300, 3), dtype=np.uint8) * 20
    cv2.putText(card, "Pipeline Summary", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 1)
    cv2.putText(card, f"Detections : {num_detections}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    num_groups = len([g for g in groups if g.get("group_label", -1) >= 0])
    cv2.putText(card, f"Groups     : {num_groups}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(card, f"Latency    : {latency_ms:.1f} ms", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1)
    return card
