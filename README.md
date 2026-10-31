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
└── stubs/                            # Local generated cache (not tracked)
```

## 🚀 Getting Started

### Requirements

```bash
pip install ultralytics opencv-python numpy scikit-learn tqdm
```

### Model Checkpoints and License

The default configuration expects `football.pt` and `pitch_keypoints.pt` in `models/`. These checkpoints were trained with Ultralytics YOLO. Ultralytics states that YOLO code and trained models are covered by AGPL-3.0 by default; an Enterprise license is the alternative for projects that cannot use AGPL-3.0. This repository is therefore released under AGPL-3.0 (see [`LICENSE`](LICENSE)).

The checkpoint metadata points to these Roboflow training datasets, both listed under CC BY 4.0:

- [football-players-detection, version 12](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/12)
- [football-field-detection, version 14](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/14)

If you replace either checkpoint, update its path in `config.py` and retain the model's license and dataset attribution.

### Input Video

No match footage is included. Use footage you have permission to process and place it at `data/input/input_video.mp4`, or change the path in `config.py`. Input videos are ignored by Git.

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

All outputs are generated in data/output/ when the pipeline runs. The video artifacts are not committed, which keeps the source repository lightweight.

| File | Description |
|------|-------------|
| `stage4_teams.mp4` | Match footage annotated with bounding boxes, IDs, and team colors |
| `stage5.mp4` | Match footage with pitch keypoints overlaid |
| `stage6_tactical.mp4` | Top-down tactical view with player/ball positions |
| `stage7_combined.mp4` | Match footage + tactical view (picture-in-picture) |

## ⚡ Caching with Stubs

The first run computes tracking and pitch keypoints from the input video. The pipeline may cache those results in `stubs/` for later runs. Cache files are ignored by Git and are regenerated automatically when missing.

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
