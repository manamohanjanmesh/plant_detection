# Plant Detection Web App — Walkthrough

## What Was Built

A complete Flask web application that uses a YOLOv9 model (`best.pt`) to detect plants in user-uploaded images and videos.

### Project Structure

```
Plant Detection/
├── app.py                      # Flask backend (3 routes, YOLO inference)
├── best.pt                     # ← You provide this (trained YOLOv9 model)
├── templates/
│   └── index.html              # Dark-themed frontend with 2-column grid
└── static/
    ├── uploads/                # Saved user uploads
    └── outputs/                # Processed results with bounding boxes
```

---

## How to Run

```bash
# 1. Install dependencies
pip install flask opencv-python torch ultralytics

# 2. Place your trained model as best.pt in the project root

# 3. Start the server
python app.py

# 4. Open in browser
#    http://127.0.0.1:5000
```

---

## Key Features

| Feature | Details |
|---|---|
| **Image detection** | Upload JPG/PNG → YOLO inference → bounding boxes + confidence labels → result displayed inline |
| **Video detection** | Upload MP4/AVI → frame-by-frame processing (every 2nd frame for speed) → output MP4 plays in browser |
| **Detection count** | Badge shows number of plants detected |
| **Performance** | Frames resized to max 640px width; frame skipping for videos; 200 MB upload limit |
| **Dark UI** | Glassmorphic cards, gradient header, hover effects, loading spinner, responsive 2-column layout |

---

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the main page |
| `/upload_image` | POST | Accepts image, runs YOLO, returns JSON with processed image URL + detection count |
| `/upload_video` | POST | Accepts video, processes frame-by-frame, returns JSON with processed video URL + detection count |

> [!IMPORTANT]
> You must place your trained `best.pt` model file in `c:\Users\medha\Documents\Plant Detection\` before running the app. The app will fail to start without it.
