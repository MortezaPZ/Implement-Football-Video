"""Central configuration for the football analytics pipeline."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
STUBS_DIR = ROOT_DIR / "stubs"
STUB_PATH = STUBS_DIR / "tracks.pkl"

# --- Model ---
MODEL_PATH = ROOT_DIR / "models" / "football.pt"   # مدل عمومی COCO که واقعاً موجوده
DEVICE = "cpu"                          # "cuda" اگه GPU داری
CONFIDENCE = 0.30
TRACKER_CONFIG = str(ROOT_DIR / "bytetrack_custom.yaml")

# --- Classes ---
# کلید = نام کلاس COCO، مقدار = نام داخلی پروژه
CLASS_NAME_MAP = {
    "player": "players",
    "referee": "referees",
    "ball": "ball",
}

TARGET_CLASSES = ["players", "referees", "ball"]

# BGR colors for the checkpoint video.
CLASS_COLORS = {
    "players": (0, 255, 0),
    "referees": (0, 255, 255),
    "ball": (0, 0, 255),
}

CONFIDENCE_THRESHOLD = 0.25

# --- Stage 4: Team Assignment ---
N_TEAMS = 2
TEAM_COLORS = {
    0: (0, 0, 255),      # قرمز
    1: (255, 0, 0),      # آبی
}
TEAM_OUTPUT = DATA_DIR / "output" / "stage4_teams.mp4"

# ---------------------------------------------------------------------------
# Stage 5 — Pitch keypoint detection
# ---------------------------------------------------------------------------
PITCH_MODEL_PATH = ROOT_DIR / "models" / "pitch_keypoints.pt"
PITCH_STUB_PATH = ROOT_DIR / "stubs" / "pitch_keypoints.pkl"

KEYPOINT_CONFIDENCE = 0.5
NUM_PITCH_KEYPOINTS = 32
PITCH_KEYPOINT_NAMES = {i: f"KP{i}" for i in range(NUM_PITCH_KEYPOINTS)}

# تنظیمات رسم کی‌پوینت‌ها (BGR)
KEYPOINT_COLOR = (0, 255, 255)        # زرد
KEYPOINT_LABEL_COLOR = (0, 0, 0)      # مشکی
KEYPOINT_RADIUS = 5

PITCH_DEBUG_PATH = DATA_DIR / "pitch_keypoints_debug.jpg"
PITCH_OUTPUT = DATA_DIR / "output" / "stage5.mp4"

# ============================================================
# Stage 6: Coordinate Transformation (Homography)
# ============================================================

# ابعاد استاندارد زمین (متر)
PITCH_LENGTH_M = 120.0
PITCH_WIDTH_M = 70.0

# رندر نمای از-بالا: مقیاس (پیکسل به ازای هر متر) و حاشیه (پیکسل)
PITCH_SCALE = 10.0      # 120m x 70m  ->  1200 x 700 px
PITCH_PADDING = 50

# آستانه‌ی اطمینان کی‌پوینت برای استفاده در هوموگرافی
HOMOGRAPHY_KEYPOINT_CONFIDENCE = 0.5

# حداقل تعداد نقاط معتبر برای محاسبه‌ی هوموگرافی (cv2.findHomography نیاز به >=4 دارد)
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4

# نگاشت اندیس کی‌پوینت (۰-ایندکس) به مختصات متری زمین.
# ترتیب استاندارد Roboflow SoccerPitch (۳۲ نقطه)، زمین 120m x 70m.
PITCH_REFERENCE_POINTS_M = {
    0:  (0.0,   0.0),     # گوشه‌ی چپ-بالا
    1:  (0.0,   14.5),    # خط پنالتی چپ - لبه‌ی بالا
    2:  (0.0,   25.84),   # دروازه‌ی کوچک چپ - لبه‌ی بالا
    3:  (0.0,   44.16),   # دروازه‌ی کوچک چپ - لبه‌ی پایین
    4:  (0.0,   55.5),    # خط پنالتی چپ - لبه‌ی پایین
    5:  (0.0,   70.0),    # گوشه‌ی چپ-پایین
    6:  (5.5,   25.84),   # محوطه‌ی شش قدم چپ - بالا
    7:  (5.5,   44.16),   # محوطه‌ی شش قدم چپ - پایین
    8:  (11.0,  35.0),    # نقطه‌ی پنالتی چپ
    9:  (20.15, 14.5),    # محوطه‌ی جریمه چپ - گوشه‌ی بالا
    10: (20.15, 25.84),   # محوطه‌ی جریمه چپ - بالا داخلی
    11: (20.15, 44.16),   # محوطه‌ی جریمه چپ - پایین داخلی
    12: (20.15, 55.5),    # محوطه‌ی جریمه چپ - گوشه‌ی پایین
    13: (60.0,  0.0),     # تقاطع خط میانی با خط طولی بالا
    14: (60.0,  25.85),   # دایره‌ی مرکز - بالا
    15: (60.0,  44.15),   # دایره‌ی مرکز - پایین
    16: (60.0,  70.0),    # تقاطع خط میانی با خط طولی پایین
    17: (99.85, 14.5),    # محوطه‌ی جریمه راست - گوشه‌ی بالا
    18: (99.85, 25.84),   # محوطه‌ی جریمه راست - بالا داخلی
    19: (99.85, 44.16),   # محوطه‌ی جریمه راست - پایین داخلی
    20: (99.85, 55.5),    # محوطه‌ی جریمه راست - گوشه‌ی پایین
    21: (109.0, 35.0),    # نقطه‌ی پنالتی راست
    22: (114.5, 25.84),   # محوطه‌ی شش قدم راست - بالا
    23: (114.5, 44.16),   # محوطه‌ی شش قدم راست - پایین
    24: (120.0, 0.0),     # گوشه‌ی راست-بالا
    25: (120.0, 14.5),    # خط پنالتی راست - لبه‌ی بالا
    26: (120.0, 25.84),   # دروازه‌ی کوچک راست - لبه‌ی بالا
    27: (120.0, 44.16),   # دروازه‌ی کوچک راست - لبه‌ی پایین
    28: (120.0, 55.5),    # خط پنالتی راست - لبه‌ی پایین
    29: (120.0, 70.0),    # گوشه‌ی راست-پایین
    30: (50.85, 35.0),    # دایره‌ی مرکز - چپ
    31: (69.15, 35.0),    # دایره‌ی مرکز - راست
}
# اندازه‌ی بوم نمای تاکتیکی (پیکسل): باید با _meters_to_pixels هماهنگ باشد
# عرض  = PITCH_LENGTH_M * PITCH_SCALE + 2 * PITCH_PADDING = 1300
# ارتفاع = PITCH_WIDTH_M  * PITCH_SCALE + 2 * PITCH_PADDING = 800
TACTICAL_VIEW_SIZE = (
    int(PITCH_LENGTH_M * PITCH_SCALE + 2 * PITCH_PADDING),
    int(PITCH_WIDTH_M * PITCH_SCALE + 2 * PITCH_PADDING),
)
# --- Stage 6: tactical view colors (BGR) ---

# رنگ پس‌زمینه‌ی زمین top-down (سبز چمنی)
TACTICAL_PITCH_COLOR = (50, 120, 50)

# رنگ خطوط زمین (سفید)
TACTICAL_LINE_COLOR = (255, 255, 255)

# مسیر خروجی ویدیوی نمای تاکتیکی
TACTICAL_OUTPUT = DATA_DIR / "output" / "stage6_tactical.mp4"

# مسیر خروجی ویدیوی ترکیبیِ Stage 7 (فوتیج اصلی + پنل تاکتیکی)
COMBINED_OUTPUT = DATA_DIR / "output" / "stage7_combined.mp4"
