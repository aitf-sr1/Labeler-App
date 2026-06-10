"""
utils/video.py
--------------
Utilitas ekstraksi frame dari video.
Fungsi utama:
  - extract_6_frames(video_path)      → list of BGR numpy arrays
  - prepare_cropped_frames(...)        → list of PIL.Image + LandmarkResult

ARSITEKTUR: MediaPipe-only — tidak ada py-feat, tidak ada subprocess eksternal.
Sinyal emosi dihitung dari blendshape MediaPipe via core/blendshape_features.py (sinkron).
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image

from core.face_detector import crop_face


_N_FRAMES = 2  # jumlah frame yang diambil per video

# Resolusi render visualisasi landmark (display-only). Viz digambar di kanvas ini
# dari crop wajah TAJAM frame asli, lalu galeri men-downscale ke ~360 → tidak pecah.
# Tidak memengaruhi SigLIP/scoring yang tetap pakai crop 224.
_VIZ_SIZE = 512


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

    Pipeline: Frame BGR → BlazeFace crop → FaceLandmarker (blendshape+pose) →
              HandLandmarker (full-frame, remap ke crop) → compute_emotion_scores

    Tidak ada py-feat, tidak ada subprocess. Semua sinkron dalam satu proses.

    Returns:
        (pil_images, no_face_count, multi_face_count, landmark_results, viz_pil_images)
        - pil_images       : list[PIL.Image] — crop bersih untuk SigLIP
        - no_face_count    : int — jumlah frame tanpa deteksi wajah
        - multi_face_count : int — jumlah frame dengan >1 wajah (untuk auto-flag)
        - landmark_results : list[LandmarkResult] — satu per frame
        - viz_pil_images   : list[PIL.Image] — crop dengan overlay landmark untuk galeri
    """
    from core.landmark_analyzer import (
        analyze_frame, compute_emotion_scores, draw_landmark_viz,
        detect_hands_from_full_frame, LandmarkResult, _BLENDSHAPE_SOURCE as _cur_bsrc
    )

    rel_path       = os.path.relpath(video_path, root_folder)
    base_name      = os.path.splitext(rel_path)[0]
    clean_crop_dir = os.path.join(crop_dir_base, "clean", base_name)
    viz_crop_dir   = os.path.join(crop_dir_base, "viz", base_name)

    os.makedirs(clean_crop_dir, exist_ok=True)
    os.makedirs(viz_crop_dir, exist_ok=True)

    # Baseline netral AU per-ORANG (Bosch 2023 + FACS: intensitas AU = deviasi dari netral pribadi).
    # Disimpan di person_neutrals.json via utils/person_neutral; format AU MediaPipe (AU1, AU2, ...).
    # Jika belum dikalibrasi → None → compute_blendshape_features() pakai baseline populasi (DEFAULT_AU_CALIB).
    _person_neutral = None
    if raw_cache_dir:
        try:
            from utils.person_neutral import get_person_neutral
            _person_neutral = get_person_neutral(os.path.dirname(raw_cache_dir), rel_path)
        except Exception:
            _person_neutral = None

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

    # Sumber blendshape cache vs sekarang: kalau beda (mis. ganti ke mp_blendshapes),
    # JANGAN pakai fast-path → paksa re-proses agar blendshape dihitung ulang dgn sumber baru.
    # Cache lama tanpa field ini dianggap "mediapipe" (default lama).
    _cached_bsrc = None
    if cache_path and os.path.exists(cache_path):
        try:
            import json as _json
            with open(cache_path) as _fp:
                _cached_bsrc = _json.load(_fp).get("blendshape_source", "mediapipe")
        except Exception:
            _cached_bsrc = None

    if (len(clean_files) == _N_FRAMES
            and len(viz_files) == _N_FRAMES
            and cache_path and os.path.exists(cache_path)
            and _cached_bsrc == _cur_bsrc):
        try:
            import json
            pil_images     = [Image.open(f).convert("RGB") for f in clean_files]
            viz_pil_images = [Image.open(f).convert("RGB") for f in viz_files]
            with open(cache_path) as fp:
                raw_data = json.load(fp)
            landmark_results = []
            for fd in raw_data.get("frames", []):
                landmark_results.append(LandmarkResult(
                    yaw        = fd.get("yaw", 0.0),
                    pitch      = fd.get("pitch", 0.0),
                    roll       = fd.get("roll", 0.0),
                    iris_x     = fd.get("iris_x", 0.0),
                    iris_y     = fd.get("iris_y", 0.0),
                    iris_img_x = fd.get("iris_img_x", 0.0),
                    iris_img_y = fd.get("iris_img_y", 0.0),
                    blendshapes= fd.get("blendshapes", {}),
                    face_found = fd.get("face_found", False),
                    # Count-based hand signals (Grafsgaard 2013b).
                    # Cache lama mungkin pakai key zona berbeda → default 0.0.
                    hand_one   = fd.get("hand_one", 0.0),
                    hand_two   = fd.get("hand_two", 0.0),
                ))

            # Retroaktif: isi hand_one/hand_two untuk cache lama yang belum punya
            needs_hand = any(
                lr.face_found and lr.hand_one == 0.0 and lr.hand_two == 0.0
                and "hand_one" not in raw_data.get("frames", [{}])[i if i < len(raw_data.get("frames",[])) else 0]
                for i, lr in enumerate(landmark_results)
            )
            if needs_hand and clean_files:
                from core.landmark_analyzer import _analyze_hands as _ah
                import mediapipe as _mp
                changed_hand = False
                for i, (lr, crop_path) in enumerate(zip(landmark_results, clean_files)):
                    if not lr.face_found:
                        continue
                    try:
                        import cv2 as _cv2
                        bgr = _cv2.imread(crop_path)
                        if bgr is None:
                            continue
                        h_img, w_img = bgr.shape[:2]
                        rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                        mp_img_h = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                        h1, h2, _dummy, _pts = _ah(mp_img_h, h_img, w_img)
                        lr.hand_one = h1
                        lr.hand_two = h2
                        if i < len(raw_data.get("frames", [])):
                            raw_data["frames"][i]["hand_one"] = h1
                            raw_data["frames"][i]["hand_two"] = h2
                        changed_hand = True
                    except Exception:
                        pass
                if changed_hand:
                    try:
                        with open(cache_path, "w") as fp:
                            json.dump(raw_data, fp, indent=2)
                    except Exception:
                        pass

            no_face_count = sum(1 for lr in landmark_results if not lr.face_found)
            # Inject person_neutral ke tiap LandmarkResult sebelum scoring
            if _person_neutral:
                for lr in landmark_results:
                    lr.person_neutral = _person_neutral
            return pil_images, no_face_count, 0, landmark_results, viz_pil_images
        except Exception as e:
            print(f"[FastPath] Gagal baca cache {rel_path}: {e}, fallback ke pipeline penuh")

    # ── Load atau generate clean crops ─────────────────────────────────────
    frames_bgr_full = []
    face_bboxes     = []

    if len(clean_files) == _N_FRAMES:
        pil_images       = [Image.open(f).convert("RGB") for f in clean_files]
        multi_face_count = 0
        frames_bgr_full  = extract_6_frames(video_path)
        for frame in frames_bgr_full:
            _, _, _, face_bbox = crop_face(frame)
            face_bboxes.append(face_bbox)
    else:
        frames_bgr = extract_6_frames(video_path)
        if not frames_bgr:
            return [], 0, 0, [], []

        pil_images       = []
        multi_face_count = 0
        frames_bgr_full  = frames_bgr
        for i, frame in enumerate(frames_bgr):
            cropped_sq, _, n_faces, face_bbox = crop_face(frame)
            if n_faces > 1:
                multi_face_count += 1
            resized_full = cv2.resize(cropped_sq, (224, 224), interpolation=cv2.INTER_AREA)
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
            hand_one, hand_two, hand_pts_px, hand_raw = detect_hands_from_full_frame(
                frames_bgr_full[i], face_bboxes[i], crop_size=224
            )
            injected_hand = (hand_one, hand_two, hand_pts_px, hand_raw)

        lr = analyze_frame(bgr, injected_hand=injected_hand)
        if not lr.face_found:
            no_face_count += 1
        landmark_results.append(lr)

    # Inject person_neutral ke semua LandmarkResult (Bosch 2023 per-person calibration)
    if _person_neutral:
        for lr in landmark_results:
            lr.person_neutral = _person_neutral

    for i, (lr, pil_img) in enumerate(zip(landmark_results, pil_images)):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        emotion_sc = compute_emotion_scores(lr, cfg)
        # Viz hi-res (display-only): crop wajah TAJAM dari frame asli, bukan 224 turun resolusi.
        viz_canvas = bgr
        if frames_bgr_full and i < len(frames_bgr_full) and face_bboxes and face_bboxes[i]:
            x1, y1, x2, y2 = face_bboxes[i]
            hires = frames_bgr_full[i][y1:y2, x1:x2]
            if hires.size > 0:
                _interp = cv2.INTER_AREA if max(hires.shape[:2]) > _VIZ_SIZE else cv2.INTER_CUBIC
                viz_canvas = cv2.resize(hires, (_VIZ_SIZE, _VIZ_SIZE), interpolation=_interp)
        viz_bgr  = draw_landmark_viz(viz_canvas, lr, emotion_sc, src_size=bgr.shape[0])
        viz_path = os.path.join(viz_crop_dir, f"frame_{i:02d}_viz.jpg")
        cv2.imwrite(viz_path, viz_bgr)
        viz_pil_images.append(
            Image.fromarray(cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB))
        )

    # Simpan raw feature cache (tanpa pyfeat_aus — MediaPipe-only)
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
                    "frame_idx":  i,
                    "face_found": lr.face_found,
                    "yaw":        lr.yaw,
                    "pitch":      lr.pitch,
                    "roll":       lr.roll,
                    "iris_x":     lr.iris_x,
                    "iris_y":     lr.iris_y,
                    "iris_img_x": lr.iris_img_x,
                    "iris_img_y": lr.iris_img_y,
                    "blendshapes":lr.blendshapes,
                    "hand_one":   lr.hand_one,
                    "hand_two":   lr.hand_two,
                })
            with open(cache_path, "w") as fp:
                json.dump({
                    "video_rel":        rel_path,
                    "generated_at":     datetime.datetime.now().isoformat(),
                    "pipeline":         "mediapipe-only",
                    "blendshape_source": _cur_bsrc,   # 'mediapipe' | 'mp_blendshapes' — utk cache invalidation
                    "frames":           frames_data,
                }, fp, indent=2)
        except Exception as e:
            print(f"[Raw Cache] Gagal simpan {rel_path}: {e}")

    return pil_images, no_face_count, multi_face_count, landmark_results, viz_pil_images
