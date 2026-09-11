# RetailVision - Retail Shelf AI Pipeline

## Overview

RetailVision is a modular computer-vision pipeline for retail shelf analysis. It accepts a shelf image, detects product-like objects, groups visually similar detections, and returns:

- an annotated image with color-coded boxes
- structured JSON output for detections and groups
- timing metrics for each stage

The project is implemented as a small microservice-style system with three main blocks:

- `Flask Server` - public API and web UI
- `Detector Service` - YOLOv8n-based product detection
- `Grouping Service` - DBSCAN-based grouping of similar detections

## Architecture

```text
Browser / API Client
        |
        | POST /pipeline
        v
+-----------------------+
| Flask Server :5000    |
| Orchestrator + UI     |
+-----------+-----------+
            |
            | POST /detect
            v
+-----------------------+
| Detector Service      |
| YOLOv8n :5001         |
+-----------+-----------+
            |
            | POST /group
            v
+-----------------------+
| Grouping Service      |
| DBSCAN :5002          |
+-----------------------+
            |
            v
   Annotated image + JSON response
```

## Project Structure

```text
retail-ai-pipeline/
├── detector_service/
│   └── detector.py
├── grouping_service/
│   └── grouping.py
├── flask_server/
│   ├── app.py
│   └── visualize.py
├── templates/
│   └── index.html
├── sample_images/
├── outputs/
├── models/
├── generate_visualizations.py
├── start_pipeline.py
├── requirements.txt
├── yolov8n.pt
└── README.md
```

## How To Run Locally

### 1. Open the project folder

```powershell
cd "D:\Retail ai pipeline\retail-ai-pipeline"
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Make sure the detector weights are present

The current demo is locked to the local base YOLO model:

```text
yolov8n.pt
```

Place this file in the project root if it is missing.

### 4. Start the pipeline

```powershell
python start_pipeline.py
```

This launches:

- Detector service on `http://localhost:5001`
- Grouping service on `http://localhost:5002`
- Flask server on `http://localhost:5000`

### 5. Open the web UI

```text
http://localhost:5000
```

Upload a sample shelf image, adjust `CONF_THRESH` and `DBSCAN EPS`, then click `Run Pipeline`.

### 6. Offline visualization script

If you want to generate annotated outputs without launching the Flask UI, you can use:

```powershell
python generate_visualizations.py
```

This script runs the detection, grouping, and visualization flow directly on local images and saves the output images into:

```text
outputs/
```

This is useful for:

- quick testing on local sample images
- generating submission screenshots
- verifying the pipeline without opening the web interface

## Current Demo Defaults

These are the values currently used by the project:

- Detection confidence threshold: `0.12`
- Grouping epsilon: `0.9`
- Max image dimension for detection resize: `1280`
- Detector model: `yolov8n.pt`

## End-to-End Pipeline

### Step 1. Input image upload

The user uploads a shelf image using the Flask UI or sends an API request to `/pipeline`.

### Step 2. Detection

The Flask server forwards the image to the detector service.

The detector:

- loads `yolov8n.pt`
- resizes large images for lower latency
- runs object detection
- filters implausibly large shelf-wide boxes
- returns product-like bounding boxes

### Step 3. Grouping

The grouped stage receives:

- original image
- detector output bounding boxes

It extracts visual features from each crop and clusters detections using DBSCAN so similar-looking products can share the same group.

### Step 4. Visualization

The final visualization stage:

- draws a color-coded box per detection
- uses the same color for detections in the same group
- appends a legend panel
- saves the annotated image to `outputs/`

### Step 5. Final response

The Flask server returns:

- detections
- groups
- image size
- saved visualization path
- base64 image
- stage-wise latency metrics

### Optional offline flow

The file `generate_visualizations.py` provides an offline version of the same pipeline. Instead of communicating through HTTP microservices, it runs the stages directly in a single script and saves annotated images locally. This is helpful for producing quick output visualizations for documentation and submission.

## Generic Input/Output Format

This section defines the JSON contracts between blocks.

### 1. Flask Block

#### Input to Flask (`POST /pipeline`)

This can be sent as multipart form-data from the UI, or represented logically as JSON:

```json
{
  "image_base64": "<base64-image>",
  "conf_thresh": 0.12,
  "eps": 0.9,
  "save_viz": true
}
```

#### Output from Flask

```json
{
  "status": "ok",
  "request_id": "abcd1234",
  "image_size": {
    "width": 1080,
    "height": 1920
  },
  "num_detections": 57,
  "num_groups": 18,
  "detections": [
    {
      "detection_id": 0,
      "bbox": {
        "x1": 120,
        "y1": 220,
        "x2": 210,
        "y2": 430
      },
      "confidence": 0.74,
      "class_id": 39,
      "class_name": "product",
      "group_id": "group_a1b2c3d4",
      "group_label": 0,
      "group_color_hex": "#E63946",
      "group_color_rgb": [230, 57, 70]
    }
  ],
  "groups": [
    {
      "group_id": "group_a1b2c3d4",
      "group_label": 0,
      "color_hex": "#E63946",
      "color_rgb": [230, 57, 70],
      "member_count": 4,
      "member_ids": [0, 4, 7, 10]
    }
  ],
  "visualization": {
    "image_base64": "<base64-annotated-image>",
    "saved_path": "outputs/result_abcd1234.jpg"
  },
  "latency": {
    "detection_ms": 3164.2,
    "grouping_ms": 5110.6,
    "visualization_ms": 108.0,
    "total_ms": 8508.9
  }
}
```

### 2. Detector Block

#### Input to Detector (`POST /detect`)

```json
{
  "image_base64": "<base64-image>",
  "conf_thresh": 0.12
}
```

#### Output from Detector

```json
{
  "status": "ok",
  "image_size": {
    "width": 1080,
    "height": 1920
  },
  "num_detections": 57,
  "detections": [
    {
      "detection_id": 0,
      "bbox": {
        "x1": 120,
        "y1": 220,
        "x2": 210,
        "y2": 430
      },
      "confidence": 0.74,
      "class_id": 39,
      "class_name": "product"
    }
  ],
  "latency_ms": 3164.2,
  "model": "D:\\Retail ai pipeline\\retail-ai-pipeline\\yolov8n.pt"
}
```

### 3. Grouping Block

#### Input to Grouping (`POST /group`)

```json
{
  "image_base64": "<base64-image>",
  "detections": [
    {
      "detection_id": 0,
      "bbox": {
        "x1": 120,
        "y1": 220,
        "x2": 210,
        "y2": 430
      },
      "confidence": 0.74,
      "class_id": 39,
      "class_name": "product"
    }
  ],
  "eps": 0.9
}
```

#### Output from Grouping

```json
{
  "status": "ok",
  "num_groups": 18,
  "groups": [
    {
      "group_id": "group_a1b2c3d4",
      "group_label": 0,
      "color_hex": "#E63946",
      "color_rgb": [230, 57, 70],
      "member_count": 4,
      "member_ids": [0, 4, 7, 10]
    }
  ],
  "detections": [
    {
      "detection_id": 0,
      "bbox": {
        "x1": 120,
        "y1": 220,
        "x2": 210,
        "y2": 430
      },
      "confidence": 0.74,
      "class_id": 39,
      "class_name": "product",
      "group_id": "group_a1b2c3d4",
      "group_label": 0,
      "group_color_hex": "#E63946",
      "group_color_rgb": [230, 57, 70]
    }
  ],
  "latency_ms": 120.4
}
```

## Design Choices

### Why YOLOv8n?

`YOLOv8n` was selected because it provides a good speed/accuracy tradeoff for a low-latency prototype. It is lightweight, simple to integrate, and fast enough for local demo execution.

### Why DBSCAN for grouping?

DBSCAN was used because:

- the number of product groups is not known in advance
- it can cluster detections based on visual similarity
- it is easy to tune with a single `eps` parameter

### Why separate services?

Separating the pipeline into Flask, detector, and grouping blocks makes the system:

- easier to debug
- easier to document
- easier to replace or upgrade later
- closer to a production-style architecture

## Observed Results

The pipeline successfully produces:

- dense product detections on retail shelves
- annotated visualization images
- group metadata in JSON
- stage-wise latency breakdowns

The detector now performs better than the earlier baseline because:

- large shelf-wide boxes are filtered out
- YOLOv8n is used consistently
- the UI and backend thresholds are aligned

## Limitations

The current solution is a working prototype, but there are still clear limitations:

- generic `yolov8n.pt` is not retail-specific
- grouping may still over-segment visually similar products
- very large images increase latency significantly
- product-level grouping would improve with fine-tuned retail weights

## Alternative Approaches

Other approaches that could improve this problem:

- Fine-tune `YOLOv8n` on `SKU-110K` for better retail-specific detection
- Use stronger visual embeddings for grouping instead of color-only features
- Resize images automatically at the Flask layer before processing
- Replace DBSCAN grouping with embedding-based nearest-neighbor clustering

## Suggested Future Improvements

- fine-tune detector on SKU-110K or grocery product datasets
- improve grouping using learned embeddings
- add automatic image resizing before detection
- reduce latency for high-resolution images
- support batch image processing

## Output Visualizations

The final output includes color-coded bounding boxes and a legend showing group assignments. Example output images can be stored in:

```text
outputs/
```

For submission, include a few representative examples showing:

- input shelf image
- output annotated image
- grouped JSON response snippet

The file `generate_visualizations.py` can be used specifically to create these saved output images for the final submission package.

## API Summary

### Flask

- `GET /` - web UI
- `POST /pipeline` - full pipeline inference
- `GET /health` - service health
- `GET /outputs/<filename>` - saved visualization access

### Detector

- `GET /health`
- `POST /detect`

### Grouping

- `GET /health`
- `POST /group`


