from .io import (
    load_annotations, save_annotations,
    load_flagged, save_flagged,
    load_frame_annotations, save_frame_annotations,
    load_batch_history, save_batch_history,
    load_batch_meta, update_batch_meta,
    load_skipped, save_skipped,
    load_thresholds, save_thresholds,
)

# prepare_cropped_frames (from utils.video) is NOT imported here —
# utils/video.py imports cv2 + mediapipe which are heavy.
# Import directly when needed: from utils.video import prepare_cropped_frames
