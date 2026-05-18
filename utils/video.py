"""
utils/video.py
--------------
Utilitas ekstraksi frame dari video.
Fungsi utama:
  - extract_6_frames(video_path)      → list of BGR numpy arrays
  - prepare_cropped_frames(...)        → list of PIL.Image
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image

from core.face_detector import crop_face


_N_FRAMES = 2  # jumlah frame yang diambil per video


def extract_6_frames(video_path: str) -> list:
    """Ambil _N_FRAMES frame yang terdistribusi merata dari video."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = [int(total * (2 * i + 1) / (2 * _N_FRAMES)) for i in range(_N_FRAMES)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def prepare_cropped_frames(
    video_path: str,
    root_folder: str,
    crop_dir_base: str,
    raw_cache_dir: str = None,
    cfg: dict = None,
) -> tuple:
    """
    Ekstrak, crop wajah (BlazeFace), jalankan landmark analysis (FaceLandmarker),
    simpan ke disk, dan kembalikan hasil.

    Selalu menjalankan landmark analysis, bahkan jika crop sudah ada di cache,
    agar viz image selalu up-to-date.

    Returns:
        (pil_images, no_face_count, landmark_results, viz_pil_images)
        - pil_images       : list[PIL.Image] — 6 crop bersih untuk SigLIP
        - no_face_count    : int — jumlah frame tanpa deteksi wajah (dari FaceLandmarker)
        - landmark_results : list[LandmarkResult] — satu per frame
        - viz_pil_images   : list[PIL.Image] — crop dengan overlay landmark untuk galeri viz
        - multi_face_count : int — jumlah frame dengan >1 wajah terdeteksi (untuk auto-flag)
    """
    from core.landmark_analyzer import (
        analyze_frame, compute_emotion_scores, draw_landmark_viz,
        detect_hands_from_full_frame, LandmarkResult
    )

    rel_path        = os.path.relpath(video_path, root_folder)
    base_name       = os.path.splitext(rel_path)[0]

    clean_crop_dir  = os.path.join(crop_dir_base, "clean", base_name)
    viz_crop_dir    = os.path.join(crop_dir_base, "viz", base_name)

    os.makedirs(clean_crop_dir, exist_ok=True)
    os.makedirs(viz_crop_dir, exist_ok=True)

    # ── Full-cache fast path: skip semua AI jika clean + viz + raw_cache ada ─
    saved_files = sorted(glob.glob(os.path.join(clean_crop_dir, "frame_*.jpg")))
    clean_files = [f for f in saved_files if "_viz" not in os.path.basename(f)]
    viz_files   = sorted(glob.glob(os.path.join(viz_crop_dir, "frame_*_viz.jpg")))

    if raw_cache_dir:
        safe_name  = rel_path.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
        safe_name  = os.path.splitext(safe_name)[0]
        cache_path = os.path.join(raw_cache_dir, safe_name + ".json")
    else:
        cache_path = None

    if (len(clean_files) == _N_FRAMES
            and len(viz_files) == _N_FRAMES
            and cache_path and os.path.exists(cache_path)):
        try:
            import json
            pil_images     = [Image.open(f).convert("RGB") for f in clean_files]
            viz_pil_images = [Image.open(f).convert("RGB") for f in viz_files]
            with open(cache_path) as fp:
                raw_data = json.load(fp)
            landmark_results = []
            for fd in raw_data.get("frames", []):
                landmark_results.append(LandmarkResult(
                    yaw          = fd.get("yaw", 0.0),
                    pitch        = fd.get("pitch", 0.0),
                    iris_x       = fd.get("iris_x", 0.0),
                    iris_y       = fd.get("iris_y", 0.0),
                    iris_img_x   = fd.get("iris_img_x", 0.0),
                    iris_img_y   = fd.get("iris_img_y", 0.0),
                    blendshapes  = fd.get("blendshapes", {}),
                    face_found   = fd.get("face_found", False),
                    hand_forehead= fd.get("hand_forehead", 0.0),
                    hand_chin    = fd.get("hand_chin", 0.0),
                ))
            no_face_count = sum(1 for lr in landmark_results if not lr.face_found)
            return pil_images, no_face_count, 0, landmark_results, viz_pil_images
        except Exception as e:
            print(f"[FastPath] Gagal baca cache {rel_path}: {e}, fallback ke pipeline penuh")

    # ── Load atau generate clean crops ─────────────────────────────────────
    frames_bgr_full = []
    face_bboxes     = []

    if len(clean_files) == _N_FRAMES:
        pil_images       = [Image.open(f).convert("RGB") for f in clean_files]
        multi_face_count = 0  # tidak diketahui dari cache
        # Re-extract frame asli agar hand detection bisa berjalan dari full frame
        frames_bgr_full = extract_6_frames(video_path)
        for frame in frames_bgr_full:
            _, _, _, face_bbox = crop_face(frame)
            face_bboxes.append(face_bbox)
    else:
        frames_bgr = extract_6_frames(video_path)
        if not frames_bgr:
            return [], 0, 0, [], []

        pil_images       = []
        multi_face_count = 0
        frames_bgr_full  = frames_bgr  # simpan frame asli untuk hand detection
        for i, frame in enumerate(frames_bgr):
            cropped_sq, _, n_faces, face_bbox = crop_face(frame)
            if n_faces > 1:
                multi_face_count += 1
            resized_full = cv2.resize(cropped_sq, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(clean_crop_dir, f"frame_{i:02d}.jpg"), resized_full)
            pil_images.append(
                Image.fromarray(cv2.cvtColor(resized_full, cv2.COLOR_BGR2RGB))
            )
            face_bboxes.append(face_bbox)

    # ── Landmark analysis + viz ──────────────────────────────────────────────
    landmark_results = []
    viz_pil_images   = []
    no_face_count    = 0

    for i, pil_img in enumerate(pil_images):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Deteksi tangan dari full frame, remap ke ruang crop wajah
        injected_hand = None
        if frames_bgr_full and i < len(frames_bgr_full) and face_bboxes:
            hand_top, hand_mid_bot, hand_pts_px, hand_raw = detect_hands_from_full_frame(
                frames_bgr_full[i], face_bboxes[i], crop_size=512
            )
            injected_hand = (hand_top, hand_mid_bot, hand_pts_px, hand_raw)

        lr  = analyze_frame(bgr, injected_hand=injected_hand)
        if not lr.face_found:
            no_face_count += 1
        landmark_results.append(lr)

        emotion_sc = compute_emotion_scores(lr, cfg)
        viz_bgr    = draw_landmark_viz(bgr, lr, emotion_sc)
        viz_path   = os.path.join(viz_crop_dir, f"frame_{i:02d}_viz.jpg")
        cv2.imwrite(viz_path, viz_bgr)
        viz_pil_images.append(
            Image.fromarray(cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB))
        )

    # Simpan raw feature cache
    if raw_cache_dir and landmark_results:
        try:
            import json, datetime
            safe = rel_path.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
            safe = os.path.splitext(safe)[0]
            cache_path = os.path.join(raw_cache_dir, safe + ".json")
            os.makedirs(raw_cache_dir, exist_ok=True)
            frames_data = []
            for i, lr in enumerate(landmark_results):
                frames_data.append({
                    "frame_idx":    i,
                    "face_found":   lr.face_found,
                    "yaw":          lr.yaw,
                    "pitch":        lr.pitch,
                    "iris_x":       lr.iris_x,
                    "iris_y":       lr.iris_y,
                    "iris_img_x":   lr.iris_img_x,
                    "iris_img_y":   lr.iris_img_y,
                    "blendshapes":  lr.blendshapes,
                    "hand_forehead": lr.hand_forehead,
                    "hand_chin":    lr.hand_chin,
                })
            with open(cache_path, "w") as fp:
                json.dump({
                    "video_rel":    rel_path,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "frames":       frames_data,
                }, fp, indent=2)
        except Exception as e:
            print(f"[Raw Cache] Gagal simpan {rel_path}: {e}")

    return pil_images, no_face_count, multi_face_count, landmark_results, viz_pil_images


