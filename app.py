"""
Plant Detection Web App — Flask Backend
Loads a YOLOv9 model (best.pt) via the WongKinYiu/yolov9 repo and runs
inference on uploaded images and videos.
"""

import os
import sys
import uuid
import cv2
import torch
from flask import Flask, render_template, request, jsonify, url_for

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV9_REPO = os.path.join(BASE_DIR, "yolov9_repo")   # local clone
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# ---------------------------------------------------------------------------
# Load YOLOv9 model via torch.hub from the local repo clone
# ---------------------------------------------------------------------------
print("[INFO] Loading YOLOv9 model …")
model = torch.hub.load(
    YOLOV9_REPO,        # local path to cloned WongKinYiu/yolov9 repo
    "custom",            # custom weights
    path=MODEL_PATH,     # path to best.pt
    source="local",      # use local repo (not GitHub download)
    force_reload=True,
)
model.conf = 0.25   # confidence threshold
model.iou  = 0.45   # NMS IoU threshold
print("[INFO] Model loaded successfully.")

# Retrieve class names from the model
CLASS_NAMES = model.names  # dict  {0: 'classA', 1: 'classB', …}

# ---------------------------------------------------------------------------
# Helper: draw bounding boxes on a frame
# ---------------------------------------------------------------------------
BOX_COLOR = (0, 255, 100)   # green-ish BGR
TEXT_BG   = (0, 0, 0)
FONT      = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame, detections):
    """
    Draw bounding boxes on *frame* in-place.

    *detections* is a tensor of shape (N, 6) with columns:
        x1, y1, x2, y2, confidence, class_id
    (this is the standard output of YOLOv9 / YOLOv5 .xyxy[0]).

    Returns the number of detections drawn.
    """
    count = 0
    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(conf)
        cls_id = int(cls_id)
        label = CLASS_NAMES.get(cls_id, str(cls_id)) if isinstance(CLASS_NAMES, dict) else str(cls_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Label background + text
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), TEXT_BG, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, 0.6, BOX_COLOR, 1, cv2.LINE_AA)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Accept an image, run YOLO detection, return processed image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    uid = uuid.uuid4().hex[:8]
    upload_path = os.path.join(UPLOAD_DIR, f"img_{uid}{ext}")
    file.save(upload_path)

    # Read image
    img = cv2.imread(upload_path)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Resize for speed (max width 640)
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)))

    # Run YOLOv9 inference  (model expects BGR numpy or filepath)
    results = model(img)

    # results.xyxy[0] → tensor (N, 6): x1,y1,x2,y2,conf,cls
    detections = results.xyxy[0].cpu()

    # Draw bounding boxes
    detection_count = draw_detections(img, detections)

    # Save processed image
    out_name = f"detected_{uid}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, img)

    return jsonify({
        "output_url": url_for("static", filename=f"outputs/{out_name}"),
        "detections": detection_count,
    })


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Accept a video file, process frame-by-frame with YOLO, return processed video."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded video
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    uid = uuid.uuid4().hex[:8]
    upload_path = os.path.join(UPLOAD_DIR, f"vid_{uid}{ext}")
    file.save(upload_path)
    print(f"[UPLOAD] Video saved: {upload_path}")

    # Open video
    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize for speed (max width 640)
    if orig_w > 640:
        scale = 640 / orig_w
        frame_w, frame_h = 640, int(orig_h * scale)
    else:
        scale = 1.0
        frame_w, frame_h = orig_w, orig_h

    # Output video writer (mp4)
    out_name = f"detected_{uid}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    total_detections = 0
    frame_idx = 0
    skip = 5  # process every 5th frame for performance

    last_detections = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype("uint8")
        # Resize
        frame = cv2.resize(frame, (frame_w, frame_h))
        # Run detection on every Nth frame; reuse results for skipped frames
        if frame_idx % skip == 0:
            results = model(frame)
            last_detections = results.xyxy[0].cpu()

        if last_detections is not None:
            count = draw_detections(frame, last_detections)
            total_detections = max(total_detections, count)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    
    # Diagnostics
    exists = os.path.exists(out_path)
    size = os.path.getsize(out_path) if exists else 0
    print(f"[OUTPUT] File: {out_path}")
    print(f"[OUTPUT] Exists: {exists}")
    print(f"[OUTPUT] Size: {size} bytes")
    
    return jsonify({
        "output_url": url_for("static", filename=f"outputs/{out_name}"),
        "detections": total_detections,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting Flask server at http://127.0.0.1:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
