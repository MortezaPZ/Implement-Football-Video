# team_assignment/team_assigner.py
"""Stage 4 - Team Assignment based on jersey color."""

import sys
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import config


class TeamAssigner:
    """رنگ پیراهن هر بازیکن را جمع می‌کند و با KMeans به دو تیم کلاستر می‌کند."""

    def __init__(self, n_teams=2):
        self.n_teams = n_teams
        self.kmeans = None
        self.player_team = {}     # track_id -> team_id (0/1)
        self.team_colors = {}     # team_id -> BGR (میانگین رنگ پیراهن واقعی)

    # ---------- استخراج رنگ یک بازیکن از یک فریم ----------
    def _player_color(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        # نیمه‌ی بالایی = پیراهن
        crop = crop[: crop.shape[0] // 2, :]
        if crop.size == 0:
            return None

        # حذف پیکسل‌های سبز چمن
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35, 40, 40]),
                            np.array([85, 255, 255]))
        pixels = crop[green == 0]
        if len(pixels) < 10:
            pixels = crop.reshape(-1, 3)
        return self._dominant_color(pixels)

    @staticmethod
    def _dominant_color(pixels):
        pixels = pixels.reshape(-1, 3).astype(float)
        if len(pixels) < 2:
            return pixels.mean(axis=0) if len(pixels) else None
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(pixels)
        counts = np.bincount(km.labels_)
        return km.cluster_centers_[counts.argmax()]

    # ---------- آموزش روی کل ویدیو (per track ID) ----------
    def fit(self, frames, tracks, class_name="players"):
        per_frame = tracks[class_name]
        colors_by_id = defaultdict(list)

        for i, frame in enumerate(
            tqdm(frames, desc="Team color sampling",
                 unit="frame", file=sys.stdout)
        ):
            if i >= len(per_frame):
                break
            for tid, info in per_frame[i].items():
                if tid < 0:                      # detection بدون ID پایدار
                    continue
                c = self._player_color(frame, info["bbox"])
                if c is not None:
                    colors_by_id[tid].append(c)

        ids, reps = [], []
        for tid, cs in colors_by_id.items():
            ids.append(tid)
            reps.append(np.median(np.array(cs), axis=0))   # رنگ نماینده پایدار

        if len(reps) < self.n_teams:
            print("Not enough players to form teams.", flush=True)
            return self

        reps = np.array(reps)
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10,
                             random_state=0).fit(reps)

        for tid, label in zip(ids, self.kmeans.labels_):
            self.player_team[tid] = int(label)
        for t in range(self.n_teams):
            self.team_colors[t] = tuple(
                int(v) for v in self.kmeans.cluster_centers_[t]
            )
        return self

    # ---------- نوشتن برچسب تیم روی tracks ----------
    def assign(self, tracks, class_name="players"):
        for per_frame in tracks[class_name]:
            for tid, info in per_frame.items():
                info["team"] = self.player_team.get(tid)
        return tracks

