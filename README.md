# Object Detection - Contact Tracing

Real-time contamination detection using YOLOv8 with a built-in tkinter UI.

## Features
- Detects persons and objects (bottles, cups, glasses)
- Tracks when hands touch objects
- Marks objects as **CONTAMINATED**
- Live camera feed embedded in UI
- Stats dashboard (Persons / Objects / Contaminated)
- Timestamped alert log
- Camera selector (0 / 1 / 2)
- Start, Stop and Clear Alerts controls

## Setup
```bash
pip install ultralytics opencv-python numpy Pillow
```

## Run
```bash
python main.py
```
Click **▶ Start** in the UI to begin detection. Click **⏹ Stop** to pause.

## How it Works
1. **YOLOv8** detects persons and objects in each frame
2. **YOLOv8-Pose** detects hand/wrist positions
3. **IoU algorithm** detects hand-object contact
4. Objects turn **RED** and log an alert when contaminated
5. UI updates live via a background detection thread

## UI Overview
| Panel | Description |
|---|---|
| Live Feed | 480×360 camera view with bounding box overlays |
| Statistics | Live counts for Persons, Objects, Contaminated |
| Controls | Camera selector, Start/Stop, Clear Alerts |
| Alert Log | Timestamped contamination events |

## Files
- `main.py` - Main detection + UI script
- `requirements.txt` - Dependencies
- `.gitignore` - Excludes model files (auto-download on first run)
