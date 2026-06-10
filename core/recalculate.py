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

ARSITEKTUR: MediaPipe-only — sinyal emosi dihitung dari blendshape MediaPipe
(compute_blendshape_features), tidak ada py-feat atau subprocess eksternal.
Chain: Craig 2008 (AU→emosi) + Turrisi 2026 (BF→AU, κ=0.92).
"""

import os
import json
from .landmark_analyzer import LandmarkResult, compute_emotion_scores


def _safe_name(rel_path: str) -> str:
    """Konversi rel_path ke nama file aman tanpa ekstensi."""
    safe = rel_path.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
    return os.path.splitext(safe)[0]


def _reconstruct_lr(frame_data: dict) -> LandmarkResult:
    """Rekonstruksi LandmarkResult dari data cache.

    Sinyal emosi dihitung oleh compute_emotion_scores() via compute_blendshape_features(blendshapes)
    — tidak ada py-feat, tidak ada subprocess eksternal.
    """
    return LandmarkResult(
        yaw        = frame_data.get("yaw", 0.0),
        pitch      = frame_data.get("pitch", 0.0),
        roll       = frame_data.get("roll", 0.0),
        iris_x     = frame_data.get("iris_x", 0.0),
        iris_y     = frame_data.get("iris_y", 0.0),
        iris_img_x = frame_data.get("iris_img_x", 0.0),
        iris_img_y = frame_data.get("iris_img_y", 0.0),
        blendshapes= frame_data.get("blendshapes", {}),
        face_found = frame_data.get("face_found", False),
        # Count-based hand (Grafsgaard 2013b). Cache lama mungkin pakai key berbeda → default 0.0.
        hand_one   = frame_data.get("hand_one", 0.0),
        hand_two   = frame_data.get("hand_two", 0.0),
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

        # Baseline netral per-orang (Bosch 2023 + FACS). dataset_dir = induk raw_cache_dir.
        # Format MediaPipe AU (AU1, AU2, AU4, ...) — dipakai compute_action_units() untuk
        # override neutral anchor dengan nilai netral pribadi orang ini.
        try:
            from utils.person_neutral import get_person_neutral
            _pn = get_person_neutral(os.path.dirname(raw_cache_dir), rel_path)
        except Exception:
            _pn = None

        # Landmark scores per frame dengan rules baru (MediaPipe blendshape)
        land_scores = []
        for fd in raw_frames:
            lr = _reconstruct_lr(fd)
            if _pn:
                lr.person_neutral = _pn
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

            thr         = thresholds[i]
            frame_preds = [1 if s >= thr else 0 for s in hybrid_scores]

            # Zero-out rejected frames dalam voting
            valid_preds = [p for j, p in enumerate(frame_preds) if j not in rejected_set]
            n_valid     = len(valid_preds)
            vote_pos    = sum(valid_preds)

            avg_score   = round(
                sum(hybrid_scores[f] for f in valid_land_frames) / max(len(valid_land_frames), 1), 4
            )
            siglip_avg  = round(sum(siglip_scores) / n_frames, 4)
            land_avg    = round(
                sum(land_scores[f][i] for f in valid_land_frames) / max(len(valid_land_frames), 1), 4
            )
            # Prediction berdasarkan avg_score vs threshold — konsisten di semua path
            prediction  = 1 if (n_valid > 0 and avg_score >= thr) else 0

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



        # Enforce mutual exclusion pada prediksi FINAL
        # Hanya pasangan yang secara semantik berlawanan langsung.
        # Eng+Conf dan Bore+Frus tetap DIPERBOLEHKAN (rare, dijaga cross-suppression di scoring).
        from ui.constants import LABELS
        for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
            idx_a = str(LABELS.index(lbl_a))
            idx_b = str(LABELS.index(lbl_b))
            if per_label_history[idx_a]["prediction"] == 1 and per_label_history[idx_b]["prediction"] == 1:
                if per_label_history[idx_a]["avg_score"] >= per_label_history[idx_b]["avg_score"]:
                    loser = idx_b
                else:
                    loser = idx_a
                per_label_history[loser]["prediction"] = 0
                per_label_history[loser]["frame_preds"] = [0] * len(per_label_history[loser]["frame_preds"])

        # Update frame_annotations dari frame_preds baru
        for f_idx in range(n_frames):
            if str(f_idx) not in new_fa_vid:
                new_fa_vid[str(f_idx)] = {}
            for li, lbl in enumerate(LABELS):
                new_fa_vid[str(f_idx)][lbl] = per_label_history[str(li)]["frame_preds"][f_idx]
            # Enforce mutual exclusion per frame (sinkron dengan level video di atas)
            for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                idx_a = LABELS.index(lbl_a)
                idx_b = LABELS.index(lbl_b)
                if new_fa_vid[str(f_idx)].get(lbl_a, 0) == 1 and new_fa_vid[str(f_idx)].get(lbl_b, 0) == 1:
                    score_a = per_label_history[str(idx_a)]["frame_scores"][f_idx]
                    score_b = per_label_history[str(idx_b)]["frame_scores"][f_idx]
                    if score_a >= score_b:
                        new_fa_vid[str(f_idx)][lbl_b] = 0
                    else:
                        new_fa_vid[str(f_idx)][lbl_a] = 0

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
