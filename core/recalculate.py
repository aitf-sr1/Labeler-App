"""
core/recalculate.py

Hitung ulang prediksi batch dari raw feature cache + siglip cache,
tanpa memproses ulang video atau menjalankan model AI.

Flow:
    raw_cache/{video}.json   +  siglip_cache/{video}.json
    ↓                               ↓
    compute_emotion_scores(cfg)     siglip_scores
    ↓_______________________________↓
    hybrid_scores → frame_preds → batch_history
"""

import os
import json
import numpy as np

from .landmark_analyzer import LandmarkResult, compute_emotion_scores


def _safe_name(rel_path: str) -> str:
    """Konversi rel_path ke nama file aman tanpa ekstensi."""
    safe = rel_path.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
    return os.path.splitext(safe)[0]


def _reconstruct_lr(frame_data: dict) -> LandmarkResult:
    """Rekonstruksi LandmarkResult dari data cache."""
    return LandmarkResult(
        yaw=frame_data.get("yaw", 0.0),
        pitch=frame_data.get("pitch", 0.0),
        roll=frame_data.get("roll", 0.0),
        iris_x=frame_data.get("iris_x", 0.0),
        iris_y=frame_data.get("iris_y", 0.0),
        iris_img_x=frame_data.get("iris_img_x", 0.0),
        iris_img_y=frame_data.get("iris_img_y", 0.0),
        blendshapes=frame_data.get("blendshapes", {}),
        face_found=frame_data.get("face_found", False),
        hand_forehead=frame_data.get("hand_forehead", 0.0),
        hand_chin=frame_data.get("hand_chin", 0.0),
    )


def recalculate_batch(
    batch_history: dict,
    rules: dict,
    thresholds: list,
    raw_cache_dir: str,
    siglip_cache_dir: str,
    frame_annotations: dict,
) -> tuple:
    """
    Hitung ulang batch_history dan frame_annotations dari cache.

    Args:
        batch_history:      Dict riwayat AI — hanya video yg ada di sini yang diproses.
        rules:              Dict rules dari RulesPanel.
        thresholds:         List[float] — satu per label, [Bore, Eng, Conf, Frus].
        raw_cache_dir:      Folder berisi {safe_name}.json per video.
        siglip_cache_dir:   Folder berisi {safe_name}.json per video.
        frame_annotations:  Dict {rel_path: {frame_idx: {label: 0|1, _rejected: bool}}}.

    Returns:
        (updated_batch_history, updated_frame_annotations, skipped_count)
        skipped_count = jumlah video yang tidak ada cache-nya (dipertahankan apa adanya).
    """
    hcfg = rules["hybrid"]
    siglip_ws = hcfg["siglip_w"]  # list per label
    land_ws   = hcfg["land_w"]

    updated_history  = {}
    updated_fa       = {k: v for k, v in frame_annotations.items()}  # shallow copy
    skipped_count    = 0

    for rel_path, old_entry in batch_history.items():
        safe = _safe_name(rel_path)
        raw_path    = os.path.join(raw_cache_dir,    safe + ".json")
        siglip_path = os.path.join(siglip_cache_dir, safe + ".json")

        if not os.path.exists(raw_path) or not os.path.exists(siglip_path):
            updated_history[rel_path] = old_entry
            skipped_count += 1
            continue

        try:
            with open(raw_path)    as f: raw_data    = json.load(f)
            with open(siglip_path) as f: siglip_data = json.load(f)
        except Exception as e:
            print(f"[Recalc] Gagal baca cache {rel_path}: {e}")
            updated_history[rel_path] = old_entry
            skipped_count += 1
            continue

        raw_frames    = raw_data["frames"]
        siglip_frames = siglip_data["frames"]
        n_frames      = len(raw_frames)

        # Landmark scores per frame dengan rules baru
        land_scores = []
        for fd in raw_frames:
            lr = _reconstruct_lr(fd)
            land_scores.append(compute_emotion_scores(lr, rules))

        # Filter dead frames (semua skor ≈ 0 = glitch MediaPipe)
        DEAD_THRESHOLD = 0.01
        valid_land_frames = [
            f for f in range(n_frames)
            if sum(land_scores[f]) > DEAD_THRESHOLD
        ]
        if not valid_land_frames:
            valid_land_frames = list(range(n_frames))  # fallback

        # Rejected frames dari frame_annotations
        fa_vid = frame_annotations.get(rel_path, {})
        rejected_set = {
            i for i in range(n_frames)
            if fa_vid.get(str(i), {}).get("_rejected", False)
        }

        per_label_history = {}
        new_fa_vid = {str(i): dict(fa_vid.get(str(i), {})) for i in range(n_frames)}

        for i in range(4):
            sw_raw = siglip_ws[i]
            lw_raw = land_ws[i]
            total  = (sw_raw + lw_raw) or 1.0
            sw     = sw_raw / total
            lw     = lw_raw / total

            siglip_scores = [
                siglip_frames[f]["siglip_scores"][i] if f < len(siglip_frames) else 0.5
                for f in range(n_frames)
            ]
            hybrid_scores = [
                round(sw * siglip_scores[f] + lw * land_scores[f][i], 4)
                for f in range(n_frames)
            ]

            # Temporal restlessness bonus untuk Boredom (i==0)
            if i == 0:
                yaws = [raw_frames[f]["yaw"] for f in range(n_frames) if raw_frames[f].get("face_found")]
                if len(yaws) >= 2:
                    yaw_std = float(np.std(yaws))
                    std_min   = hcfg["restless_std_min"]
                    std_range = hcfg["restless_std_range"]
                    bonus_max = hcfg["restless_bonus_max"]
                    bonus = min(max((yaw_std - std_min) / std_range, 0.0), 1.0) * bonus_max
                    if bonus > 0.01:
                        hybrid_scores = [round(min(s + bonus, 1.0), 4) for s in hybrid_scores]

            thr         = thresholds[i]
            frame_preds = [1 if s >= thr else 0 for s in hybrid_scores]

            # Zero-out rejected frames dalam voting
            valid_preds = [p for j, p in enumerate(frame_preds) if j not in rejected_set]
            n_valid     = len(valid_preds)
            vote_pos    = sum(valid_preds)
            prediction  = 1 if (n_valid > 0 and vote_pos >= max(1, (n_valid + 1) // 2)) else 0

            avg_score   = round(
                sum(hybrid_scores[f] for f in valid_land_frames) / max(len(valid_land_frames), 1), 4
            )
            siglip_avg  = round(sum(siglip_scores) / n_frames, 4)
            land_avg    = round(
                sum(land_scores[f][i] for f in valid_land_frames) / max(len(valid_land_frames), 1), 4
            )

            per_label_history[str(i)] = {
                "prediction":   prediction,
                "vote_pos":     vote_pos,
                "vote_neg":     n_valid - vote_pos,
                "skipped":      len(rejected_set),
                "avg_score":    avg_score,
                "siglip_avg":   siglip_avg,
                "landmark_avg": land_avg,
                "threshold":    thr,
                "frame_scores": hybrid_scores,
                "frame_preds":  frame_preds,
            }



        # Update frame_annotations dari frame_preds baru
        from ui.constants import LABELS
        for f_idx in range(n_frames):
            if str(f_idx) not in new_fa_vid:
                new_fa_vid[str(f_idx)] = {}
            for li, lbl in enumerate(LABELS):
                new_fa_vid[str(f_idx)][lbl] = per_label_history[str(li)]["frame_preds"][f_idx]

        # Auto-reject per FRAME: hanya frame tanpa label yang ditolak
        rejected_count = 0
        for f_idx in range(n_frames):
            frame_has_label = any(
                per_label_history[str(i)]["frame_preds"][f_idx] == 1
                for i in range(4)
            )
            if not frame_has_label:
                new_fa_vid[str(f_idx)]["_rejected"] = True
                rejected_count += 1
        if rejected_count > 0:
            print(f"  [NO_LABEL] {rel_path} → {rejected_count}/{n_frames} frame ditolak")

        # Update no_label flag
        no_label_flag = rejected_count == n_frames

        updated_history[rel_path] = {
            "per_label":  per_label_history,
            "thresholds": thresholds,
            "no_label":   no_label_flag,
        }

        updated_fa[rel_path] = new_fa_vid

    return updated_history, updated_fa, skipped_count
