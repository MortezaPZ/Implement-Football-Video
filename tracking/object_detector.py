"""Stage 2 - Object detection with a YOLO model (ultralytics).

Frame-level detection only. Returns bounding boxes, class labels and
confidence scores per frame. Stable IDs across frames are Stage 3.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

import config


class ObjectDetector:
    """Wraps a YOLO model and produces per-frame detections."""

    def __init__(self, model_path=None, confidence=None):
        self.model_path = str(model_path or config.MODEL_PATH)
        self.confidence = (
            confidence if confidence is not None
            else config.CONFIDENCE_THRESHOLD
        )
        self.model = YOLO(self.model_path)
        self.model_names = self.model.names  # raw names from the weights
        self.model.to(config.DEVICE)

    def _parse_result(self, result):
        """Convert one ultralytics Result into a list of detection dicts."""
        detections = []
        boxes = result.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, class_id in zip(xyxy, confs, class_ids):
            raw = self.model_names.get(int(class_id), str(class_id)).lower()
            name = config.CLASS_NAME_MAP.get(raw, raw)
            if name not in config.TARGET_CLASSES:
                continue
            detections.append({
                "bbox": np.array(bbox, dtype=float),  # [x1, y1, x2, y2]
                "class_id": int(class_id),
                "class_name": name,
                "confidence": float(conf),
            })
        return detections

    def detect_frame(self, frame):
        """Detect objects in a single BGR frame."""
        result = self.model.predict(
            frame, conf=self.confidence, device=config.DEVICE, verbose=False
        )[0]
        return self._parse_result(result)


    def detect_frames(self, frames, batch_size=None):
        """Batched detection over a list of frames.

        Returns a list with one detection list per input frame.
        """
        batch_size = batch_size or config.DETECTION_BATCH_SIZE
        all_detections = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            results = self.model.predict(
                batch, conf=self.confidence, device=config.DEVICE, verbose=False
            )
            for result in results:
                all_detections.append(self._parse_result(result))
        return all_detections

