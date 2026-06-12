# ⚽ Football Match Analytics Pipeline

A computer-vision pipeline that analyzes football match footage end-to-end: detecting and tracking players/referees/ball, assigning players to teams by jersey color, detecting pitch landmarks, and rendering a tactical top-down view of the match. 🎥📊

## ✨ Features

- 🎬 **Video I/O** — read and write match footage
- 🎯 **Object Detection & Tracking** — YOLO-based detection with persistent IDs via ByteTrack
- 👕 **Team Assignment** — clusters players into two teams based on jersey color (KMeans)
- 🏟️ **Pitch Keypoint Detection** — YOLO-pose model detects 32 standard pitch landmarks
- 🗺️ **Coordinate Transformation** — homography maps camera-view positions to real-world pitch coordinates (meters)
- 🧭 **Tactical (Top-Down) View** — renders players and ball as a 2D tactical map
- 🖼️ **Combined Output** — picture-in-picture video combining match footage with the tactical view
- 💾 **Caching (stubs)** — tracking and pitch-keypoint results can be cached to skip expensive re-computation

## 📂 Project Structure

```
├── main.py                          # Pipeline entry point (runs all stages)
├── config.py                        # Central configuration (paths, models, pitch geometry, colors)
├── tracking/
│   ├── object_detector.py           # Stage 2 - YOLO object detection
│   ├── tracker.py                   # Stage 3 - Multi-object tracking (ByteTrack)
│   └── pitch_detector.py            # Stage 5 - Pitch keypoint detection
├── team_assignment/
│   └── team_assigner.py             # Stage 4 - Team assignment via jersey color clustering
├── transformation/
│   └── transformer.py               # Stage 6 - Homography & coordinate transformation
├── visualization/
│   └── renderer.py                  # Drawing helpers for all checkpoint/output videos
├── utils/
│   └── video_utils.py               # Stage 1 - Video read/write helpers
└── stubs/                            # Cached tracking & pitch keypoint results (.pkl)
```

## 🚀 Getting Started

### Requirements

```bash
pip install ultralytics opencv-python numpy scikit-learn tqdm
```

### Models

Place the required YOLO weights in a `models/` folder:

```
models/
├── football.pt           # Player/referee/ball detection model
└── pitch_keypoints.pt    # Pitch landmark (keypoint) detection model
```

### Input Video

Place your match footage at:

```
data/input/input_video.mp4
```

### Run

```bash
python main.py
```

## 🛠️ Pipeline Stages

1. **Stage 1 — Video I/O**: reads all frames from the input video.
2. **Stage 2 — Object Detection**: a YOLO model detects players, referees, and the ball in each frame.
3. **Stage 3 — Tracking**: ByteTrack assigns stable IDs to detections across frames.
4. **Stage 4 — Team Assignment**: samples jersey colors per player and clusters them into 2 teams with KMeans.
5. **Stage 5 — Pitch Keypoints**: a YOLO-pose model detects 32 standard pitch landmarks per frame.
6. **Stage 6 — Coordinate Transformation**: computes a homography from pitch keypoints to map player/ball positions into real-world pitch coordinates (meters), rendered as a top-down tactical view.
7. **Stage 7 — Combined Output**: overlays the tactical view as picture-in-picture on the annotated match footage.

## 📤 Outputs

All outputs are written to `data/output/`:

| File | Description |
|------|-------------|
| `stage4_teams.mp4` | Match footage annotated with bounding boxes, IDs, and team colors |
| `stage5.mp4` | Match footage with pitch keypoints overlaid |
| `stage6_tactical.mp4` | Top-down tactical view with player/ball positions |
| `stage7_combined.mp4` | Match footage + tactical view (picture-in-picture) |

## ⚡ Caching with Stubs

To avoid re-running detection/tracking on every execution, results are cached as pickle files in `stubs/`:

- `stubs/tracks.pkl` — cached tracking results
- `stubs/pitch_keypoints.pkl` — cached pitch keypoint detections

Set `read_from_stub=True` (default in `main.py`) to reuse cached results, or delete the stub files to force recomputation.

## ⚙️ Configuration

All key settings live in `config.py`, including:

- Model paths and device (`cpu`/`cuda`)
- Detection confidence thresholds
- Class mappings (players, referees, ball)
- Team colors
- Pitch dimensions, reference keypoints, and tactical view rendering settings

## 📝 Notes

- `config.DEVICE` defaults to `"cpu"` — change to `"cuda"` if a GPU is available for significantly faster detection and tracking.
- The pitch reference points follow the standard 32-point Roboflow SoccerPitch layout for a 120m x 70m pitch.

## 👤 Author

Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
