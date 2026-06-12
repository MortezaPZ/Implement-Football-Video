"""Stage 5: detect pitch landmark keypoints with a trained YOLO-pose model."""

import os
import pickle

import numpy as np
from ultralytics import YOLO

import config


class PitchDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = str(model_path or config.PITCH_MODEL_PATH)
        self.confidence = confidence or config.KEYPOINT_CONFIDENCE
        self.num_keypoints = config.NUM_PITCH_KEYPOINTS
        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)

    def _parse_result(self, result):
        n = self.num_keypoints
        empty = np.zeros((n, 3), dtype=np.float32)

        if result.keypoints is None or result.keypoints.xy is None:
            return empty
        if len(result.keypoints.xy) == 0:
            return empty

        xy = result.keypoints.xy[0].cpu().numpy()
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[0].cpu().numpy()
        else:
            conf = np.ones(xy.shape[0], dtype=np.float32)

        kpts = np.zeros((xy.shape[0], 3), dtype=np.float32)
        kpts[:, :2] = xy
        kpts[:, 2] = conf
        kpts[kpts[:, 2] < self.confidence] = 0.0
        return kpts

    def detect_frame(self, frame):
        result = self.model.predict(
            frame,
            conf=self.confidence,
            device=config.DEVICE,
            verbose=False,
        )[0]
        return self._parse_result(result)

    def detect_frames(self, frames):
        return [self.detect_frame(f) for f in frames]

    def get_keypoints(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        keypoints = self.detect_frames(frames)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(keypoints, f)
        return keypoints
