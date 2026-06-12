import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import config


class Tracker:
    """Stage 3 - Multi Object Tracking با ByteTrack داخلی Ultralytics."""

    def __init__(self, model_path=None, confidence=None, tracker_cfg="bytetrack.yaml"):
        self.model_path = model_path or config.MODEL_PATH
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.tracker_cfg = tracker_cfg

        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)
        self.model_names = self.model.names
        self._fallback_id = -1

    def _empty_tracks(self):
        return {name: [] for name in config.TARGET_CLASSES}

    def _parse_result(self, result):
        frame_tracks = {name: {} for name in config.TARGET_CLASSES}
        boxes = result.boxes
        if boxes is None:
            return frame_tracks

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None \
              else [None] * len(xyxy)

        for bbox, conf, cid, tid in zip(xyxy, confs, class_ids, ids):
            raw = self.model_names.get(int(cid), str(cid)).lower()
            name = config.CLASS_NAME_MAP.get(raw, raw)
            if name not in config.TARGET_CLASSES:
                continue
            if tid is None:
                tid = self._fallback_id
                self._fallback_id -= 1
            frame_tracks[name][int(tid)] = {
                "bbox": np.array(bbox, dtype=float),   # [x1, y1, x2, y2]
                "confidence": float(conf),
                "class_id": int(cid),
                "class_name": name,
            }
        return frame_tracks

    def track_frames(self, frames):
        tracks = self._empty_tracks()
        print("Warming up GPU & tracking (first frame may take 30-60s)...",
              flush=True)
        for frame in tqdm(frames, desc="Tracking", unit="frame",
                          file=sys.stdout):
            result = self.model.track(
                frame,
                conf=self.confidence,
                iou=0.5,
                imgsz=1280,
                device=config.DEVICE,
                tracker=self.tracker_cfg,
                persist=True,        # نگه‌داری ID بین فریم‌ها
                verbose=False,
            )[0]
            parsed = self._parse_result(result)
            for name in config.TARGET_CLASSES:
                tracks[name].append(parsed[name])
        return tracks

    def get_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading cached tracks from {stub_path}", flush=True)
            return self.load_tracks(stub_path)
        tracks = self.track_frames(frames)
        if stub_path:
            self.save_tracks(tracks, stub_path)
        return tracks

    @staticmethod
    def save_tracks(tracks, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(tracks, f)

    @staticmethod
    def load_tracks(path):
        with open(path, "rb") as f:
            return pickle.load(f)
