import numpy as np
import cv2

import config


class CoordinateTransformer:
    """
    Stage 6: Coordinate Transformation.

    از کی‌پوینت‌های زمین (مرحله ۵) یک هموگرافی محاسبه می‌کند که نقاطِ
    نمای دوربین را به نمای از-بالای زمین (متر) می‌برد. سپس موقعیت
    بازیکنان و توپ را به فضای زمین نگاشت می‌کند.

    مدیریت خطا:
      - اگر تعداد کی‌پوینت‌های معتبر کمتر از حد لازم باشد، آخرین هموگرافی
        معتبر دوباره استفاده می‌شود.
      - اگر هیچ هموگرافی معتبری تا کنون وجود نداشته باشد، نگاشت None برمی‌گردد.
    """

    def __init__(self):
        self.reference_points = config.PITCH_REFERENCE_POINTS_M
        self.min_points = config.MIN_KEYPOINTS_FOR_HOMOGRAPHY
        self.conf_threshold = config.HOMOGRAPHY_KEYPOINT_CONFIDENCE
        self.scale = config.PITCH_SCALE
        self.padding = config.PITCH_PADDING
        self._last_homography = None

    # ---------- هموگرافی ----------

    def _meters_to_pixels(self, x_m, y_m):
        """مختصات متری زمین را به پیکسلِ تصویر top-down تبدیل می‌کند."""
        px = x_m * self.scale + self.padding
        py = y_m * self.scale + self.padding
        return px, py

    def compute_homography(self, keypoints):
        """
        keypoints: آرایه‌ی (NUM_PITCH_KEYPOINTS, 3) -> (x, y, conf) در نمای دوربین.
        خروجی: ماتریس هموگرافی 3x3 یا None اگر نقاط کافی نباشد.
        """
        src_points = []  # نمای دوربین (پیکسل)
        dst_points = []  # نمای زمین (پیکسل top-down)

        for idx, (x, y, conf) in enumerate(keypoints):
            if conf < self.conf_threshold:
                continue
            if x == 0 and y == 0:
                continue
            if idx not in self.reference_points:
                continue

            ref_x_m, ref_y_m = self.reference_points[idx]
            dst_px, dst_py = self._meters_to_pixels(ref_x_m, ref_y_m)

            src_points.append([x, y])
            dst_points.append([dst_px, dst_py])

        if len(src_points) < self.min_points:
            # نقاط کافی نیست؛ از آخرین هموگرافی معتبر استفاده کن
            return self._last_homography

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            return self._last_homography

        self._last_homography = H
        return H

    # ---------- نگاشت نقاط ----------

    def transform_points(self, points, homography):
        """
        points: آرایه‌ی (N, 2) در نمای دوربین.
        خروجی: آرایه‌ی (N, 2) در نمای زمین (پیکسل top-down) یا None.
        """
        if homography is None or len(points) == 0:
            return None

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, homography)
        return transformed.reshape(-1, 2)

    @staticmethod
    def get_foot_position(bbox):
        """کف-وسط باکس را برمی‌گرداند (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2.0, y2]

    # ---------- API اصلی روی کل فریم‌ها ----------

    def transform_tracks(self, tracks, pitch_keypoints):
        """
        tracks: ساختار خروجی مرحله ۳/۴ به ازای هر فریم، شامل players/ball
                که هر کدام dict[track_id] با کلید 'bbox' (و اختیاری 'team').
        pitch_keypoints: لیست آرایه‌های (NUM_PITCH_KEYPOINTS, 3) به ازای هر فریم.

        خروجی: لیستی هم‌طول با فریم‌ها، هر عضو dict شامل موقعیت‌های نگاشت‌شده.
        """
        results = []
        num_frames = len(pitch_keypoints)

        for frame_idx in range(num_frames):
            H = self.compute_homography(pitch_keypoints[frame_idx])

            frame_result = {"players": {}, "ball": {}, "homography_valid": H is not None}

            if H is None:
                results.append(frame_result)
                continue

            # بازیکنان: کف-وسط باکس
            players = tracks["players"][frame_idx] if frame_idx < len(tracks["players"]) else {}
            foot_points = []
            ids = []
            for track_id, info in players.items():
                foot_points.append(self.get_foot_position(info["bbox"]))
                ids.append(track_id)

            mapped = self.transform_points(foot_points, H)
            if mapped is not None:
                for track_id, pt in zip(ids, mapped):
                    frame_result["players"][track_id] = {
                        "position": pt.tolist(),
                        "team": players[track_id].get("team"),
                    }

            # توپ: مرکز باکس
            ball = tracks["ball"][frame_idx] if frame_idx < len(tracks["ball"]) else {}
            for track_id, info in ball.items():
                x1, y1, x2, y2 = info["bbox"]
                center = [[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
                mapped_ball = self.transform_points(center, H)
                if mapped_ball is not None:
                    frame_result["ball"][track_id] = {"position": mapped_ball[0].tolist()}

            results.append(frame_result)

        return results
