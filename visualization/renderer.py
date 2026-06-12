"""Drawing helpers for checkpoint videos."""

import os
import cv2
import config
import numpy as np


def _resolve_color(info):
    """رنگ کادر: اگر بازیکن تیم دارد از رنگ تیم، وگرنه از رنگ کلاس."""
    team = info.get("team")
    if team is not None:
        return config.TEAM_COLORS.get(team, (200, 200, 200))
    name = info["class_name"]
    return config.CLASS_COLORS.get(name, (200, 200, 200))


def draw_track(frame, info, track_id):
    """Draw one tracked box with its ID, in-place."""
    x1, y1, x2, y2 = info["bbox"].astype(int)
    name = info["class_name"]
    color = _resolve_color(info)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = name if track_id < 0 else f"{name} #{track_id}"
    team = info.get("team")
    if team is not None:
        label += f" (T{team})"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def draw_tracks_on_frames(frames, tracks):
    """Annotate frames using the Stage 3 tracks dict."""
    output_frames = []
    for i, frame in enumerate(frames):
        out = frame.copy()
        for per_frame in tracks.values():
            if i >= len(per_frame):
                continue
            for track_id, info in per_frame[i].items():
                draw_track(out, info, track_id)
        output_frames.append(out)
    return output_frames


# ---------------------------------------------------------------------------
# Stage 5 — Pitch keypoints
# ---------------------------------------------------------------------------

def draw_pitch_keypoints(frame, keypoints):
    """Draw indexed pitch keypoints on a copy of the frame."""
    out = frame.copy()
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf <= 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(out, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(
            out, str(idx), (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
        )
    return out


def draw_keypoints_on_frames(frames, keypoints_per_frame):
    """Overlay keypoints across all frames (for the Stage 5 output video)."""
    return [
        draw_pitch_keypoints(f, kp)
        for f, kp in zip(frames, keypoints_per_frame)
    ]


def save_pitch_debug_frame(frame, keypoints, path=None):
    """Save a single annotated debug frame to disk."""
    path = str(path or config.PITCH_DEBUG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, draw_pitch_keypoints(frame, keypoints))
    return path


# ---------------------------------------------------------------------------
# Stage 6 — Tactical (top-down) view
# ---------------------------------------------------------------------------

def draw_tactical_pitch():
    """یک تصویر خالیِ زمین از-بالا (top-down) با خطوط استاندارد می‌سازد."""
    w, h = config.TACTICAL_VIEW_SIZE
    pitch = np.full((h, w, 3), config.TACTICAL_PITCH_COLOR, dtype=np.uint8)

    scale = config.PITCH_SCALE
    pad = config.PITCH_PADDING
    color = config.TACTICAL_LINE_COLOR

    def m2p(x_m, y_m):
        return int(x_m * scale + pad), int(y_m * scale + pad)

    L, W = config.PITCH_LENGTH_M, config.PITCH_WIDTH_M

    cv2.rectangle(pitch, m2p(0, 0), m2p(L, W), color, 2)
    cv2.line(pitch, m2p(L / 2, 0), m2p(L / 2, W), color, 2)
    cv2.circle(pitch, m2p(L / 2, W / 2), int(9.15 * scale), color, 2)
    cv2.circle(pitch, m2p(L / 2, W / 2), 3, color, -1)
    cv2.rectangle(pitch, m2p(0, 13.84), m2p(16.5, 54.16), color, 2)
    cv2.rectangle(pitch, m2p(L - 16.5, 13.84), m2p(L, 54.16), color, 2)
    cv2.rectangle(pitch, m2p(0, 24.84), m2p(5.5, 43.16), color, 2)
    cv2.rectangle(pitch, m2p(L - 5.5, 24.84), m2p(L, 43.16), color, 2)
    cv2.circle(pitch, m2p(11.0, W / 2), 3, color, -1)
    cv2.circle(pitch, m2p(L - 11.0, W / 2), 3, color, -1)

    return pitch


def draw_tactical_frame(transformed_frame, team_colors=None):
    """
    موقعیت‌های نگاشت‌شده‌ی یک فریم را روی زمین top-down رسم می‌کند.
    نکته: positions از ترانسفورمر در پیکسلِ همین بوم هستند (scale و padding
    قبلاً اعمال شده‌اند)، پس مستقیم با int(x), int(y) رسم می‌شوند.
    """
    pitch = draw_tactical_pitch()

    if not transformed_frame.get("homography_valid", False):
        cv2.putText(
            pitch, "NO HOMOGRAPHY", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return pitch

    team_colors = team_colors or {}

    for track_id, info in transformed_frame["players"].items():
        x, y = info["position"]
        team = info.get("team")
        color = team_colors.get(team, (200, 200, 200)) if team is not None else (200, 200, 200)
        color = tuple(int(c) for c in color)
        cv2.circle(pitch, (int(x), int(y)), 6, color, -1)
        cv2.circle(pitch, (int(x), int(y)), 6, (0, 0, 0), 1)

    for track_id, info in transformed_frame["ball"].items():
        x, y = info["position"]
        cv2.circle(pitch, (int(x), int(y)), 5, (0, 255, 255), -1)

    return pitch


def draw_tactical_frames(transformed, team_colors=None):
    """لیست فریم‌های تاکتیکی را برمی‌گرداند تا با video_utils ذخیره شوند."""
    return [draw_tactical_frame(tf, team_colors) for tf in transformed]


# ---------------------------------------------------------------------------
# Stage 7 — Combined visualization (match footage + tactical panel)
# ---------------------------------------------------------------------------

def draw_combined_frame(annotated_frame, tactical_frame):
    """
    فریمِ annotate‌شده‌ی ویدیوی اصلی را کنار نمای تاکتیکی قرار می‌دهد (side-by-side).
    نمای تاکتیکی هم‌ارتفاع با فریم اصلی resize می‌شود تا hstack ممکن باشد.
    """
    h_main = annotated_frame.shape[0]

    th, tw = tactical_frame.shape[:2]
    new_w = max(1, int(round(tw * (h_main / th))))
    if new_w % 2 != 0:          # عرض زوج برای سازگاری با ویدیو-رایتر
        new_w += 1

    tactical_resized = cv2.resize(tactical_frame, (new_w, h_main))
    return np.hstack([annotated_frame, tactical_resized])


def draw_combined_frame(annotated_frame, tactical_frame, scale=0.32, margin=20, alpha=0.5):
    """
    نمای تاکتیکی را به صورت overlay کوچک در گوشه‌ی پایین-راست
    فریم اصلیِ annotate شده قرار می‌دهد (picture-in-picture).

    annotated_frame : فریم اصلی با باکس/ID/رنگ تیم (BGR)
    tactical_frame  : نمای تاکتیکی رندرشده (BGR)
    scale           : نسبت عرض overlay به عرض فریم اصلی
    margin          : فاصله از لبه‌ها به پیکسل
    alpha           : شفافیت overlay (1.0 = کاملاً مات)
    """
    out = annotated_frame.copy()
    H, W = out.shape[:2]

    # عرض هدف برای نمای تاکتیکی و حفظ نسبت ابعاد
    target_w = int(W * scale)
    th, tw = tactical_frame.shape[:2]
    target_h = int(target_w * th / tw)

    overlay = cv2.resize(tactical_frame, (target_w, target_h),
                         interpolation=cv2.INTER_AREA)

    # موقعیت گوشه‌ی پایین-راست
    x2, y2 = W - margin, H - margin
    x1, y1 = x2 - target_w, y2 - target_h

    # کادر سفید دور overlay برای جداسازی بصری
    cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

    roi = out[y1:y2, x1:x2]
    if alpha >= 1.0:
        out[y1:y2, x1:x2] = overlay
    else:
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        out[y1:y2, x1:x2] = roi

    return out


def save_combined_video(annotated_frames, tactical_frames, out_path, fps):
    """
    ویدیوی ترکیبی Stage 7 را به‌صورت استریم می‌نویسد؛ هیچ لیست بزرگی
    در حافظه ساخته نمی‌شود (هر فریم بعد از نوشته‌شدن آزاد می‌شود).
    """
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)

    writer = None
    try:
        for a, t in zip(annotated_frames, tactical_frames):
            combined = draw_combined_frame(a, t)

            if writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            writer.write(combined)
    finally:
        if writer is not None:
            writer.release()
