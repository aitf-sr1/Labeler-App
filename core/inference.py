"""
core/inference.py

Inferensi zero-shot label emosi menggunakan SigLIP2.

Untuk setiap video:
    1. Semua prompt (4 label × 6 deskripsi = 24 teks) diproses dalam satu batch
    2. Model menghasilkan logits [n_frames × n_texts]
    3. Logits dinormalisasi min-max per frame agar skor antar prompt sebanding
    4. Rata-rata skor positif prompt per label dihitung untuk setiap frame
    5. Prediksi akhir ditentukan dari avg_score vs threshold
"""

import torch
from .siglip_model import get_siglip, get_device


def run_siglip_on_frames(
    pil_images: list,
    prompt_groups: list,
    thresholds: list,
    ambiguity_margin: float = 0.02,
) -> dict:
    """
    Inferensi SigLIP2 pada 16 frame dari satu video.

    Args:
        pil_images:    List[PIL.Image] — 16 frame crop wajah.
        prompt_groups: List[(pos_lines, _)] per label. neg_lines diabaikan.
        thresholds:    List[float] — satu threshold per label.
        ambiguity_margin: Tidak digunakan, dipertahankan untuk kompatibilitas API.

    Returns:
        {"per_label": {i: {prediction, vote_pos, vote_neg, skipped,
                           avg_score, frame_scores, frame_preds}},
         "n_frames": int, "thresholds": list}
    """
    device           = get_device()
    model, processor = get_siglip()
    n_labels         = len(prompt_groups)

    all_texts, group_indices, current_idx = [], [], 0
    for pos_lines, _neg_lines in prompt_groups:
        all_texts.extend(pos_lines)
        group_indices.append(list(range(current_idx, current_idx + len(pos_lines))))
        current_idx += len(pos_lines)

    inputs = processor(
        text=all_texts, images=pil_images,
        return_tensors="pt", padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_per_image = model(**inputs).logits_per_image  # [n_frames, n_texts]

    n_frames = len(pil_images)

    # Normalisasi min-max per frame agar skor antar prompt bisa dibandingkan secara relatif.
    # Sigmoid langsung tidak digunakan karena skala logit tidak konsisten antar-video.
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

    return {
        "per_label":  per_label_result,
        "n_frames":   n_frames,
        "thresholds": thresholds,
    }
