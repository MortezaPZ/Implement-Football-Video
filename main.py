"""Pipeline entry point. Stage 1 (I/O) + 3 (tracking) + 4 (teams) + 5 (pitch) + 6 (transform) + 7 (viz)."""

import config
from utils.video_utils import read_video, save_video
from tracking.tracker import Tracker
from visualization.renderer import (
    draw_tracks_on_frames,
    draw_keypoints_on_frames,
    save_pitch_debug_frame,
    draw_tactical_frames,
    save_combined_video,
)
from team_assignment.team_assigner import TeamAssigner
from tracking.pitch_detector import PitchDetector
from transformation.transformer import CoordinateTransformer
import gc
gc.collect()


def main():
    input_path = config.DATA_DIR / "input" / "input_video.mp4"
    output_path = config.DATA_DIR / "output" / "stage4.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: read frames.
    frames, fps = read_video(str(input_path))
    print(f"Read {len(frames)} frames at {fps:.2f} FPS")

    # Stage 3: track objects with stable IDs.
    tracker = Tracker()
    print(f"Loaded model: {tracker.model_path}")

    tracks = tracker.get_tracks(
        frames,
        read_from_stub=True,
        stub_path=config.STUB_PATH,
    )

    for name in config.TARGET_CLASSES:
        unique_ids = set()
        for frame_tracks in tracks[name]:
            unique_ids.update(frame_tracks.keys())
        print(f"{name}: {len(unique_ids)} unique track IDs")

    # Stage 4: assign teams by jersey color.
    assigner = TeamAssigner(n_teams=config.N_TEAMS)
    assigner.fit(frames, tracks, class_name="players")
    tracks = assigner.assign(tracks, class_name="players")
    for t, c in assigner.team_colors.items():
        print(f"Team {t} sampled jersey color (BGR): {c}")

    # Checkpoint: annotated video with team colors.
    annotated = draw_tracks_on_frames(frames, tracks)
    save_video(annotated, str(config.TEAM_OUTPUT), fps)
    print(f"Saved Stage 4 checkpoint to {config.TEAM_OUTPUT}")

    # Stage 5: detect pitch landmark keypoints.
    pitch_detector = PitchDetector()
    print(f"Loaded pitch model: {pitch_detector.model_path}")

    pitch_keypoints = pitch_detector.get_keypoints(
        frames,
        read_from_stub=True,
        stub_path=config.PITCH_STUB_PATH,
    )

    # Checkpoint: video with pitch keypoints overlaid.
    pitch_annotated = draw_keypoints_on_frames(frames, pitch_keypoints)
    save_video(pitch_annotated, str(config.PITCH_OUTPUT), fps)
    print(f"Saved Stage 5 checkpoint to {config.PITCH_OUTPUT}")

    # ---- Stage 6: Coordinate Transformation ----
    transformer = CoordinateTransformer()
    transformed = transformer.transform_tracks(tracks, pitch_keypoints)

    tactical_frames = draw_tactical_frames(transformed, assigner.team_colors)
    save_video(tactical_frames, str(config.TACTICAL_OUTPUT), fps)
    print(f"[Stage 6] tactical view saved: {config.TACTICAL_OUTPUT}")

    # ---- Stage 7: Combined visualization (footage + tactical panel) ----
    # آزادکردن لیست‌هایی که دیگر لازم نیستند تا حافظه برای ویدیوی ترکیبی بماند.

    save_combined_video(annotated, tactical_frames, config.COMBINED_OUTPUT, fps)
    print(f"[Stage 7] combined view saved: {config.COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
