"""
core/face_detector.py

Deteksi wajah menggunakan MediaPipe BlazeFace dan crop area wajah dari frame BGR.
Model diunduh otomatis ke ~/.cache/siglip_labeler/ pada pertama kali digunakan.

Alur crop_face():
    frame BGR -> deteksi wajah -> square crop dengan padding
               -> jika tidak ada wajah: fallback ke center crop
"""

import os
import cv2
import urllib.request

_MP_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
_MP_MODEL_NAME = "blaze_face_short_range.tflite"
_mp_detector   = None


def _get_mp_detector():
    """Lazy-load BlazeFace detector. Model di-cache di ~/.cache/siglip_labeler/."""
    global _mp_detector
    if _mp_detector is None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "siglip_labeler")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, _MP_MODEL_NAME)

        if not os.path.exists(model_path):
            urllib.request.urlretrieve(_MP_MODEL_URL, model_path)

        base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts      = mp_vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=0.50,
        )
        _mp_detector = mp_vision.FaceDetector.create_from_options(opts)
    return _mp_detector


def crop_face(frame_bgr, padding_scale: float = 0.60):
    """
    Crop area wajah terbesar dari frame BGR.

    Padding ditambahkan di sekitar bounding box agar kepala dan leher ikut masuk.
    Jika tidak ada wajah terdeteksi, fallback ke center crop 80% dari sisi terpendek.
    Return frame asli jika hasil crop berukuran 0.

    Returns:
        (cropped_img, face_found: bool)
    """
    import mediapipe as mp

    h, w      = frame_bgr.shape[:2]
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = _get_mp_detector().detect(mp_image)

    if result.detections:
        best     = max(result.detections,
                       key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb       = best.bounding_box
        max_dim  = max(bb.width, bb.height)
        pad_px   = int(max_dim * padding_scale)
        cx, cy   = bb.origin_x + (bb.width // 2), bb.origin_y + (bb.height // 2)
        half_dim = (max_dim + (pad_px * 2)) // 2
        x1, y1  = max(0, cx - half_dim), max(0, cy - half_dim)
        x2, y2  = min(w, cx + half_dim), min(h, cy + half_dim)
        crop    = frame_bgr[y1:y2, x1:x2]
        return (crop if crop.size > 0 else frame_bgr), True

    # Fallback: center crop — wajah tidak terdeteksi
    cx, cy   = w // 2, h // 2
    half_dim = int(min(w, h) * 0.40)
    fallback = frame_bgr[
        max(0, cy - half_dim): cy + half_dim,
        max(0, cx - half_dim): cx + half_dim,
    ]
    return fallback, False

