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
    (0.35, 0.65),   # Boredom     — landmark dominan (head yaw + restlessness paling reliable)
    (0.45, 0.55),   # Engagement  — landmark dominan (forward gate paling reliable)
    (0.75, 0.25),   # Confusion   — SigLIP dominan (blendshapes subtle, hand rare)
    (0.65, 0.35),   # Frustration — SigLIP dominan (ekspresi + hand coverage luas)
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
    pil_images:     list,
    prompt_groups:  list,
    thresholds:     list,
    ambiguity_margin: float = 0.02,
    landmark_results: list  = None,
) -> dict:
    """
    Inferensi SigLIP2 pada 16 frame dari satu video, dengan hybrid scoring
    opsional menggunakan MediaPipe FaceLandmarker.

    Args:
        pil_images:       List[PIL.Image] — 16 frame crop wajah.
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

    # ── MURNI SIGMOID (Sesuai Arsitektur Asli SigLIP) ───────────────────────
    # Logit dari SigLIP secara bawaan sudah didesain sebagai input untuk Sigmoid
    # untuk menghasilkan probabilitas independen (multi-label).
    # Kita tidak boleh melakukan normalisasi max() per frame karena akan merusak
    # keyakinan absolut model. Namun, karena logit zero-shot untuk teks yang spesifik 
    # seringkali berada di rentang negatif (misal -3.0 hingga -6.0), nilai sigmoid murni
    # akan sangat kecil (mendekati 0).
    # Solusinya: Gunakan Bias Kalibrasi Statis (Empirical Bias) untuk menggeser kurva
    # tanpa memanipulasi distribusi antar-frame.
    EMPIRICAL_BIAS = 3.5

    norm_by_label = []
    for i in range(n_labels):
        group_logits = logits_per_image[:, group_indices[i]]       # [n_frames, 6]
        probs        = torch.sigmoid(group_logits + EMPIRICAL_BIAS) # [n_frames, 6]
        norm_by_label.append(probs.mean(dim=1))                    # [n_frames]

    # Pre-compute landmark scores per frame (jika tersedia)
    land_scores_per_frame = None
    if landmark_results and len(landmark_results) == n_frames:
        land_scores_per_frame = [compute_emotion_scores(r) for r in landmark_results]

    per_label_result = {}
    for i in range(n_labels):
        # Bobot per-label dari env
        siglip_w, land_w = _get_label_weights(i)

        # SigLIP score per frame
        siglip_scores = [
            round(norm_by_label[i][f].item(), 4)
            for f in range(n_frames)
        ]

        # Hybrid scoring per frame
        if land_scores_per_frame:
            hybrid_scores = [
                round(siglip_w * siglip_scores[f] + land_w * land_scores_per_frame[f][i], 4)
                for f in range(n_frames)
            ]
            land_avg = round(
                sum(land_scores_per_frame[f][i] for f in range(n_frames)) / n_frames, 4
            )

            # ── Temporal Restlessness Bonus (khusus Boredom, i=0) ────────
            # Jika kepala bergerak bolak-balik (std yaw tinggi), naikkan skor boredom.
            # Ini menangkap pola 'tolah-toleh' yang tidak bisa dideteksi per-frame.
            if i == 0:
                yaws = [r.yaw for r in landmark_results if r.face_found]
                if len(yaws) >= 4:
                    import numpy as np
                    yaw_std = float(np.std(yaws))
                    # std >= 3° mulai bonus, >= 10° = bonus penuh (0.15)
                    restless_bonus = min(max((yaw_std - 3.0) / 7.0, 0.0), 1.0) * 0.15
                    if restless_bonus > 0.01:
                        hybrid_scores = [
                            round(min(s + restless_bonus, 1.0), 4) for s in hybrid_scores
                        ]
                        print(f"  [RESTLESS] yaw_std={yaw_std:.1f}° → bonus={restless_bonus:.3f}")
        else:
            hybrid_scores = siglip_scores
            land_avg      = None

        avg_score   = round(sum(hybrid_scores) / n_frames, 4)
        siglip_avg  = round(sum(siglip_scores) / n_frames, 4)
        thr         = thresholds[i]
        vote_pos    = sum(1 for s in hybrid_scores if s >= thr)
        frame_preds = [1 if s >= thr else 0 for s in hybrid_scores]

        per_label_result[i] = {
            "prediction":   1 if avg_score >= thr else 0,
            "vote_pos":     vote_pos,
            "vote_neg":     n_frames - vote_pos,
            "skipped":      0,
            "avg_score":    avg_score,
            "siglip_avg":   siglip_avg,
            "landmark_avg": land_avg,
            "frame_scores": hybrid_scores,
            "frame_preds":  frame_preds,
        }

    return {
        "per_label":  per_label_result,
        "n_frames":   n_frames,
        "thresholds": thresholds,
    }

