"""
core/inference.py

Inferensi zero-shot label emosi menggunakan SigLIP2.

Untuk setiap video:
    1. Semua prompt (4 label × 6 deskripsi = 24 teks) diproses dalam satu batch
    2. Model menghasilkan logits [n_frames × n_texts]
    3. Logits langsung dilewatkan ke fungsi Sigmoid (karena SigLIP = Sigmoid Loss)
    4. Rata-rata probabilitas prompt per label dihitung untuk setiap frame
    5. Prediksi akhir ditentukan dari avg_score vs threshold
"""

import torch
from .siglip_model import get_siglip, get_device


_LABEL_KEYS   = ["BOREDOM", "ENGAGEMENT", "CONFUSION", "FRUSTRATION"]
_LABEL_DEFAULTS = [
    (0.50, 0.50),   # Boredom
    (0.50, 0.50),   # Engagement
    (0.50, 0.50),   # Confusion
    (0.50, 0.50),   # Frustration
]


def _get_label_weights(label_idx: int) -> tuple:
    """
    Baca bobot hybrid per-label dari env.

    Urutan prioritas:
    1. {LABEL}_SIGLIP_WEIGHT / {LABEL}_LANDMARK_WEIGHT   (per-label)
    2. SIGLIP_WEIGHT / LANDMARK_WEIGHT                   (global fallback)
    3. Hardcoded default dari _LABEL_DEFAULTS
    """
    import os
    sw_def, lw_def = _LABEL_DEFAULTS[label_idx]
    key = _LABEL_KEYS[label_idx]
    try:
        # Per-label env var, fall back to global, then hardcoded
        global_sw = float(os.getenv("SIGLIP_WEIGHT",   str(sw_def)))
        global_lw = float(os.getenv("LANDMARK_WEIGHT", str(lw_def)))
        sw = float(os.getenv(f"{key}_SIGLIP_WEIGHT",   str(global_sw)))
        lw = float(os.getenv(f"{key}_LANDMARK_WEIGHT", str(global_lw)))
        total = sw + lw if (sw + lw) > 0 else 1
        return sw / total, lw / total
    except ValueError:
        return sw_def / (sw_def + lw_def), lw_def / (sw_def + lw_def)


def run_siglip_on_frames(
    pil_images:       list,
    prompt_groups:    list,
    thresholds:       list,
    ambiguity_margin: float = 0.02,
    landmark_results: list  = None,
    cfg:              dict  = None,
    siglip_cache_path: str  = None,
    rel_path:         str   = None,
) -> dict:
    """
    Inferensi SigLIP2 pada 4 frame dari satu video, dengan hybrid scoring
    opsional menggunakan MediaPipe FaceLandmarker.

    Args:
        pil_images:       List[PIL.Image] — 4 frame crop wajah.
        prompt_groups:    List[(pos_lines, _)] per label.
        thresholds:       List[float] — satu threshold per label.
        ambiguity_margin: Tidak digunakan, dipertahankan untuk kompatibilitas API.
        landmark_results: List[LandmarkResult] opsional — hasil analyze_frame per frame.
                          Jika diberikan, skor akhir = α×SigLIP + β×Landmark.

    Returns:
        {"per_label": {i: {prediction, vote_pos, vote_neg, skipped,
                           avg_score, frame_scores, frame_preds,
                           siglip_avg, landmark_avg}},
         "n_frames": int, "thresholds": list}
    """
    from core.landmark_analyzer import compute_emotion_scores
    from core.rules import DEFAULT_RULES
    import datetime

    if cfg is None:
        cfg = DEFAULT_RULES

    device           = get_device()
    model, processor = get_siglip()
    n_labels         = len(prompt_groups)

    # ── PERBAIKAN: split string multi-baris → list prompt individual ──────────
    # pos_lines bisa berupa str (dari constants) atau list (dari UI editor)
    all_texts, group_indices, current_idx = [], [], 0
    for pos_lines, _neg_lines in prompt_groups:
        if isinstance(pos_lines, str):
            lines = [l.strip() for l in pos_lines.strip().split("\n") if l.strip()]
        else:
            lines = [str(l).strip() for l in pos_lines if str(l).strip()]
        all_texts.extend(lines)
        group_indices.append(list(range(current_idx, current_idx + len(lines))))
        current_idx += len(lines)

    inputs = processor(
        text=all_texts, images=pil_images,
        return_tensors="pt", padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_per_image = model(**inputs).logits_per_image  # [n_frames, n_texts]

    n_frames = len(pil_images)

    # ── SIGMOID dengan Empirical Bias ─────────────────────────────────────────
    EMPIRICAL_BIAS = cfg["hybrid"]["empirical_bias"]

    norm_by_label = []
    for i in range(n_labels):
        group_logits = logits_per_image[:, group_indices[i]]       # [n_frames, 6]
        probs        = torch.sigmoid(group_logits + EMPIRICAL_BIAS) # [n_frames, 6]
        norm_by_label.append(probs.mean(dim=1))                    # [n_frames]

    # Pre-compute landmark scores per frame (jika tersedia)
    land_scores_per_frame = None
    if landmark_results and len(landmark_results) == n_frames:
        land_scores_per_frame = [compute_emotion_scores(r, cfg) for r in landmark_results]

    # Simpan siglip scores per frame ke cache (sebelum hybrid)
    if siglip_cache_path:
        try:
            import os
            os.makedirs(os.path.dirname(siglip_cache_path), exist_ok=True)
            siglip_frame_data = []
            for f in range(n_frames):
                per_label_scores = [
                    round(norm_by_label[i][f].item(), 4) for i in range(n_labels)
                ]
                siglip_frame_data.append({
                    "frame_idx": f,
                    "siglip_scores": per_label_scores,
                })
            import json
            with open(siglip_cache_path, "w") as fp:
                json.dump({
                    "video_rel": rel_path or "",
                    "generated_at": datetime.datetime.now().isoformat(),
                    "empirical_bias": EMPIRICAL_BIAS,
                    "frames": siglip_frame_data,
                }, fp, indent=2)
        except Exception as e:
            print(f"[SigLIP Cache] Gagal simpan: {e}")

    per_label_result = {}
    for i in range(n_labels):
        # Bobot per-label dari cfg (fallback ke env)
        hcfg = cfg["hybrid"]
        sw_raw = hcfg["siglip_w"][i] if i < len(hcfg["siglip_w"]) else 0.5
        lw_raw = hcfg["land_w"][i]   if i < len(hcfg["land_w"])   else 0.5
        total  = (sw_raw + lw_raw) or 1.0
        siglip_w = sw_raw / total
        land_w   = lw_raw / total

        # SigLIP score per frame
        siglip_scores = [
            round(norm_by_label[i][f].item(), 4)
            for f in range(n_frames)
        ]

        # Hybrid scoring per frame
        if land_scores_per_frame:
            # Filter: frame yang semua landmark scores ≈ 0 dianggap glitch MediaPipe,
            # tidak ikut rata-rata supaya tidak menurunkan skor frame yang valid.
            DEAD_THRESHOLD = 0.01
            valid_frames = [
                f for f in range(n_frames)
                if sum(land_scores_per_frame[f]) > DEAD_THRESHOLD
            ]
            if not valid_frames:
                valid_frames = list(range(n_frames))  # fallback semua

            hybrid_scores = [
                round(siglip_w * siglip_scores[f] + land_w * land_scores_per_frame[f][i], 4)
                for f in range(n_frames)
            ]
            land_avg = round(
                sum(land_scores_per_frame[f][i] for f in valid_frames) / len(valid_frames), 4
            )

            # ── Temporal Restlessness Bonus (khusus Boredom, i=0) ────────
            if i == 0:
                yaws = [r.yaw for r in landmark_results if r.face_found]
                if len(yaws) >= 2:
                    import numpy as np
                    yaw_std = float(np.std(yaws))
                    std_min   = hcfg["restless_std_min"]
                    std_range = hcfg["restless_std_range"]
                    bonus_max = hcfg["restless_bonus_max"]
                    restless_bonus = min(max((yaw_std - std_min) / std_range, 0.0), 1.0) * bonus_max
                    if restless_bonus > 0.01:
                        hybrid_scores = [
                            round(min(s + restless_bonus, 1.0), 4) for s in hybrid_scores
                        ]
                        print(f"  [RESTLESS] yaw_std={yaw_std:.1f}° → bonus={restless_bonus:.3f}")
        else:
            hybrid_scores = siglip_scores
            land_avg      = None

        avg_score   = round(
            sum(hybrid_scores[f] for f in (valid_frames if land_scores_per_frame else range(n_frames)))
            / max(len(valid_frames) if land_scores_per_frame else n_frames, 1), 4
        )
        siglip_avg  = round(sum(siglip_scores) / n_frames, 4)
        thr         = thresholds[i]
        frame_preds = [1 if s >= thr else 0 for s in hybrid_scores]
        vote_pos    = sum(frame_preds)
        # Majority vote — konsisten dengan recalculate.py
        prediction  = 1 if (n_frames > 0 and vote_pos >= max(1, (n_frames + 1) // 2)) else 0

        per_label_result[i] = {
            "prediction":   prediction,
            "vote_pos":     vote_pos,
            "vote_neg":     n_frames - vote_pos,
            "skipped":      0,
            "avg_score":    avg_score,
            "siglip_avg":   siglip_avg,
            "landmark_avg": land_avg,
            "frame_scores": hybrid_scores,
            "frame_preds":  frame_preds,
        }

    # Cleanup GPU memory — cegah akumulasi tensor antar video dalam batch
    del inputs, logits_per_image, norm_by_label
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Deteksi sampel tanpa label — semua emosi di bawah threshold
    has_any_label = any(
        per_label_result[i]["prediction"] == 1 for i in range(n_labels)
    )

    return {
        "per_label":  per_label_result,
        "n_frames":   n_frames,
        "thresholds": thresholds,
        "no_label":   not has_any_label,
    }

