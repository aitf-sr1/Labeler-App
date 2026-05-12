"""
ai_service.py

Fungsi bantu untuk mode REST API (main.py).
Menggabungkan ekstraksi frame, crop wajah, dan inferensi SigLIP dalam satu modul.

Catatan: modul ini adalah duplikat dari core/ + utils/video.py yang dibuat khusus
untuk kebutuhan API tanpa bergantung pada struktur paket app.py.

Flow prepare_cropped_frames:
    video_path -> extract_6_frames() -> crop_face() per frame
              -> simpan ke disk (cache) -> return list PIL.Image

Flow run_siglip_on_frames:
    pil_images + prompts + thresholds -> model SigLIP2
    -> normalisasi min-max per frame -> voting per label -> return hasil
"""

import os
import glob
import urllib.request

import cv2
import torch
from PIL import Image

_siglip_model     = None
_siglip_processor = None
_device           = "cuda" if torch.cuda.is_available() else "cpu"

_MP_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
_MP_MODEL_NAME = "blaze_face_short_range.tflite"
_mp_detector   = None


def get_siglip():
    """
    Lazy-load SigLIP2 ke GPU atau CPU.

    Returns:
        (model, processor) — siap digunakan untuk inferensi.
    """
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        import os
        MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-224")
        print(f"Memuat model SigLIP2: {MODEL_ID} ke {_device.upper()}...")
        _siglip_processor = AutoProcessor.from_pretrained(MODEL_ID)
        _siglip_model     = AutoModel.from_pretrained(MODEL_ID)
        _siglip_model.to(_device)
        _siglip_model.eval()
    return _siglip_model, _siglip_processor


def _get_mp_detector():
    """
    Lazy-load MediaPipe BlazeFace detector.

    Model di-cache di ~/.cache/siglip_labeler/. Otomatis diunduh jika belum ada.

    Returns:
        FaceDetector instance yang siap digunakan.
    """
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


def crop_face(frame_bgr, padding_scale: float = 0.20):
    """
    Crop area wajah terbesar dari frame BGR.

    Jika tidak ada wajah terdeteksi, fallback ke center crop 80% frame.

    Args:
        frame_bgr:     Frame dalam format BGR (output OpenCV).
        padding_scale: Rasio padding terhadap ukuran bounding box.

    Returns:
        Cropped frame BGR. Jika crop gagal, kembalikan frame asli.
    """
    import mediapipe as mp

    h, w      = frame_bgr.shape[:2]
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = _get_mp_detector().detect(mp_image)

    if result.detections:
        best     = max(result.detections,
                       key=lambda det: det.bounding_box.width * det.bounding_box.height)
        bb       = best.bounding_box
        max_dim  = max(bb.width, bb.height)
        pad_px   = int(max_dim * padding_scale)
        cx, cy   = bb.origin_x + (bb.width // 2), bb.origin_y + (bb.height // 2)
        half_dim = (max_dim + (pad_px * 2)) // 2
        x1, y1  = max(0, cx - half_dim), max(0, cy - half_dim)
        x2, y2  = min(w, cx + half_dim), min(h, cy + half_dim)
        crop    = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else frame_bgr

    # Fallback: center crop
    cx, cy   = w // 2, h // 2
    half_dim = int(min(w, h) * 0.40)
    return frame_bgr[
        max(0, cy - half_dim): cy + half_dim,
        max(0, cx - half_dim): cx + half_dim,
    ]


def extract_6_frames(video_path: str) -> list:
    """
    Ekstrak 4 frame dari video secara merata dari awal hingga akhir.

    Args:
        video_path: Path absolut ke file video.

    Returns:
        List of frame BGR. Kosong jika video tidak bisa dibuka.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = [int(total * i / 4) for i in range(4)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def prepare_cropped_frames(video_path: str, root_folder: str, crop_dir_base: str) -> list:
    """
    Siapkan 6 frame crop wajah dari video dalam format PIL.Image.

    Jika cache sudah ada (6 file frame_*.jpg), langsung load dari disk.
    Jika belum, ekstrak frame -> crop wajah -> simpan ke disk -> return PIL list.

    Args:
        video_path:    Path absolut ke file video.
        root_folder:   Folder root dataset untuk menghitung rel_path.
        crop_dir_base: Folder dasar tempat cache crop disimpan.

    Returns:
        List of 6 PIL.Image RGB. Kosong jika video tidak bisa diproses.
    """
    rel_path        = os.path.relpath(video_path, root_folder)
    base_name       = os.path.splitext(rel_path)[0]
    target_crop_dir = os.path.join(crop_dir_base, base_name)
    os.makedirs(target_crop_dir, exist_ok=True)

    saved_files = sorted(glob.glob(os.path.join(target_crop_dir, "frame_*.jpg")))
    if len(saved_files) == 4:
        return [Image.open(f).convert("RGB") for f in saved_files]

    frames_bgr = extract_6_frames(video_path)
    if not frames_bgr:
        return []

    pil_images = []
    for i, frame in enumerate(frames_bgr):
        cropped     = crop_face(frame)
        resized     = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)
        out_path    = os.path.join(target_crop_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(out_path, resized)
        pil_images.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    return pil_images


def run_siglip_on_frames(
    pil_images: list,
    prompt_groups: list,
    thresholds: list,
    ambiguity_margin: float = 0.02,
) -> dict:
    """
    Jalankan inferensi SigLIP2 pada 6 frame dari satu video.

    Args:
        pil_images:       List of PIL.Image — 6 frame hasil crop wajah.
        prompt_groups:    List of (pos_lines, _) per label.
        thresholds:       List of float threshold per label.
        ambiguity_margin: Tidak digunakan. Dipertahankan untuk kompatibilitas.

    Returns:
        {
            "per_label":  {i: {"prediction", "vote_pos", "vote_neg", "skipped",
                               "avg_score", "frame_scores", "frame_preds"}},
            "n_frames":   int,
            "thresholds": list,
        }
    """
    model, processor = get_siglip()
    n_labels = len(prompt_groups)

    all_texts, group_indices, current_idx = [], [], 0
    for pos_lines, _neg_lines in prompt_groups:
        all_texts.extend(pos_lines)
        group_indices.append(list(range(current_idx, current_idx + len(pos_lines))))
        current_idx += len(pos_lines)

    inputs = processor(
        text=all_texts, images=pil_images,
        return_tensors="pt", padding="max_length",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_per_image = model(**inputs).logits_per_image  # [n_frames, n_texts]

    n_frames = len(pil_images)

    # Normalisasi min-max per frame agar skor antar prompt bisa dibandingkan
    logits_min  = logits_per_image.min(dim=1, keepdim=True).values
    logits_max  = logits_per_image.max(dim=1, keepdim=True).values
    norm_logits = (logits_per_image - logits_min) / (logits_max - logits_min + 1e-8)

    per_label_result = {}
    for i in range(n_labels):
        pos_idx = group_indices[i]
        scores  = [
            round(norm_logits[f][pos_idx].mean().item(), 4)
            for f in range(n_frames)
        ]
        avg_score   = round(sum(scores) / n_frames, 4)
        thr         = thresholds[i]
        vote_pos    = sum(1 for s in scores if s >= thr)
        frame_preds = [1 if s >= thr else 0 for s in scores]

        per_label_result[i] = {
            "prediction":   1 if avg_score >= thr else 0,
            "vote_pos":     vote_pos,
            "vote_neg":     n_frames - vote_pos,
            "skipped":      0,
            "avg_score":    avg_score,
            "frame_scores": scores,
            "frame_preds":  frame_preds,
        }

    return {"per_label": per_label_result, "n_frames": n_frames, "thresholds": thresholds}
