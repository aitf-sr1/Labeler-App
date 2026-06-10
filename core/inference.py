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


# ── Mutual exclusion pasangan emosi berlawanan (PENYEDERHANAAN DESAIN) ────────
# Sinkron dengan core/recalculate.py agar preview galeri == hasil final tersimpan.
# Index LABELS: 0=Boredom 1=Engagement 2=Confusion 3=Frustration.
#
# CATATAN KEJUJURAN (hard XOR ini BUKAN verbatim paper — lihat DESIGN_RATIONALE §11/§15):
#   Bore↔Eng : yang paper-grounded adalah hubungan KOMPLEMENTER (DAiSEE 2016 §5
#              "when engagement is low, boredom is generally high and vice-versa";
#              D'Mello 2012) → sudah ditangani SOFT suppress `bore_eng_suppress=0.40`
#              di skor landmark. Hard XOR di sini = versi keras (penyederhanaan).
#   Conf↔Frus: basis LEMAH — paper (D'Mello 2012) hanya menunjukkan TRANSISI temporal
#              confusion→frustration, BUKAN mutual exclusion per-frame. DAiSEE multi-label
#              malah membolehkan co-occur. Hard XOR di sini = penyederhanaan desain
#              (alasan praktis: kirim label bersih ke LLM hilir), bukan dari paper.
# Aturan: jika kedua pasangan diprediksi 1, ambil yang skor DOMINAN.
_MUTEX_PAIRS = [(0, 1), (2, 3)]


def _apply_mutual_exclusion(per_label_result: dict, n_frames: int) -> None:
    """Terapkan mutual exclusion in-place pada hasil per-label (struktur inference)."""
    # Video-level: jika kedua emosi pasangan diprediksi 1, yang skor-avg lebih rendah → 0
    # (sekaligus nol-kan semua frame_preds-nya — konsisten dengan recalculate.py).
    for a, b in _MUTEX_PAIRS:
        if per_label_result[a]["prediction"] == 1 and per_label_result[b]["prediction"] == 1:
            loser = b if per_label_result[a]["avg_score"] >= per_label_result[b]["avg_score"] else a
            per_label_result[loser]["prediction"] = 0
            per_label_result[loser]["frame_preds"] = [0] * n_frames
    # Per-frame: untuk frame yang masih dobel, ambil yang skor-frame lebih tinggi
    for f in range(n_frames):
        for a, b in _MUTEX_PAIRS:
            if per_label_result[a]["frame_preds"][f] == 1 and per_label_result[b]["frame_preds"][f] == 1:
                sa = per_label_result[a]["frame_scores"][f]
                sb = per_label_result[b]["frame_scores"][f]
                loser = b if sa >= sb else a
                per_label_result[loser]["frame_preds"][f] = 0
    # Resync vote counts
    for i in per_label_result:
        per_label_result[i]["vote_pos"] = sum(per_label_result[i]["frame_preds"])
        per_label_result[i]["vote_neg"] = n_frames - per_label_result[i]["vote_pos"]


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

            # CATATAN: eng_gaze_gate eksternal DIHAPUS. Prinsip Whitehill 2014 ("looking away
            # = not engaged") sudah ditegakkan DI DALAM skor landmark Engagement (gate via
            # gaze_dev_eng + yaw_gate + pitch_gate, Whitehill/Sümer-grounded) yang kini berbobot
            # land_w[Eng]=0.75. Gate eksternal hanya aplikasi-ganda redundan pada porsi SigLIP
            # dan menimbulkan divergensi inference vs recalculate → dihapus (tanpa kehilangan
            # mekanisme paper). Lihat DESIGN_RATIONALE.
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
        # Prediction berdasarkan avg_score vs threshold — konsisten di semua path
        prediction  = 1 if (n_frames > 0 and avg_score >= thr) else 0

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
            # per-frame breakdown untuk debug
            "siglip_scores": siglip_scores,
            "land_scores": (
                [round(land_scores_per_frame[f][i], 4) for f in range(n_frames)]
                if land_scores_per_frame else None
            ),
            "threshold":    thr,
        }

    # Cleanup GPU memory — cegah akumulasi tensor antar video dalam batch
    del inputs, logits_per_image, norm_by_label
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # CATATAN: strict_rules_bias & dominance_gap DIHAPUS (keduanya "spec", bukan dari paper;
    # dominance_gap juga bertentangan dengan multi-label DAiSEE). Skor multi-label dibiarkan
    # apa adanya (tiap emosi independen). Lihat DESIGN_RATIONALE §15.

    # Recompute predictions after adjustments
    for i in range(n_labels):
        r = per_label_result[i]
        r["frame_preds"] = [1 if s >= thresholds[i] else 0 for s in r["frame_scores"]]
        r["vote_pos"] = sum(r["frame_preds"])
        r["vote_neg"] = n_frames - r["vote_pos"]
        vf = (valid_frames if land_scores_per_frame else list(range(n_frames)))
        r["avg_score"] = round(
            sum(r["frame_scores"][f] for f in vf) / max(len(vf), 1), 4
        )
        r["prediction"] = 1 if (n_frames > 0 and r["avg_score"] >= thresholds[i]) else 0

    # Mutual exclusion (Bore↔Eng, Conf↔Frus) — sinkron dgn recalculate.py (galeri == final)
    if n_labels == 4:
        _apply_mutual_exclusion(per_label_result, n_frames)

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


def run_siglip_batch(
    batch_items:   list,
    prompt_groups: list,
    thresholds:    list,
    cfg:           dict = None,
) -> list:
    """
    Proses multiple video dalam SATU GPU forward pass.

    batch_items: list of dict dengan keys:
        - pil_images:        List[PIL.Image]  — crop wajah per frame
        - landmark_results:  List[LandmarkResult] atau None
        - rel_path:          str
        - siglip_cache_path: str atau None

    Returns: list of result dict, urutan sama dengan batch_items.
    Format tiap result sama dengan run_siglip_on_frames().
    """
    from core.landmark_analyzer import compute_emotion_scores
    from core.rules import DEFAULT_RULES
    import datetime, json, os

    if not batch_items:
        return []
    if cfg is None:
        cfg = DEFAULT_RULES

    device           = get_device()
    model, processor = get_siglip()
    n_labels         = len(prompt_groups)

    # ── Build text prompts (sama untuk semua video) ───────────────────────────
    all_texts, group_indices, current_idx = [], [], 0
    for pos_lines, _neg in prompt_groups:
        if isinstance(pos_lines, str):
            lines = [l.strip() for l in pos_lines.strip().split("\n") if l.strip()]
        else:
            lines = [str(l).strip() for l in pos_lines if str(l).strip()]
        all_texts.extend(lines)
        group_indices.append(list(range(current_idx, current_idx + len(lines))))
        current_idx += len(lines)

    # ── Stack semua frame dari semua video ────────────────────────────────────
    all_pil_images = []
    n_frames_per_video = []
    for item in batch_items:
        imgs = item["pil_images"]
        all_pil_images.extend(imgs)
        n_frames_per_video.append(len(imgs))

    # ── Satu GPU forward pass untuk seluruh batch ─────────────────────────────
    inputs = processor(
        text=all_texts, images=all_pil_images,
        return_tensors="pt", padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_all = model(**inputs).logits_per_image  # [total_frames, n_texts]

    EMPIRICAL_BIAS = cfg["hybrid"]["empirical_bias"]

    # sigmoid untuk semua frame × semua label sekaligus
    norm_all = []
    for i in range(n_labels):
        group_logits = logits_all[:, group_indices[i]]
        probs        = torch.sigmoid(group_logits + EMPIRICAL_BIAS)
        norm_all.append(probs.mean(dim=1))   # [total_frames]

    # ── Post-process per video ────────────────────────────────────────────────
    results      = []
    frame_offset = 0

    for item, n_frames in zip(batch_items, n_frames_per_video):
        landmark_results  = item.get("landmark_results")
        rel_path          = item["rel_path"]
        siglip_cache_path = item.get("siglip_cache_path")

        norm_by_label = [norm_all[i][frame_offset:frame_offset + n_frames] for i in range(n_labels)]

        # Landmark scores
        land_scores_per_frame = None
        if landmark_results and len(landmark_results) == n_frames:
            land_scores_per_frame = [compute_emotion_scores(r, cfg) for r in landmark_results]

        # Simpan siglip cache
        if siglip_cache_path:
            try:
                os.makedirs(os.path.dirname(siglip_cache_path), exist_ok=True)
                siglip_frame_data = [
                    {"frame_idx": f,
                     "siglip_scores": [round(norm_by_label[i][f].item(), 4) for i in range(n_labels)]}
                    for f in range(n_frames)
                ]
                with open(siglip_cache_path, "w") as fp:
                    json.dump({
                        "video_rel":      rel_path or "",
                        "generated_at":   datetime.datetime.now().isoformat(),
                        "empirical_bias": EMPIRICAL_BIAS,
                        "frames":         siglip_frame_data,
                    }, fp, indent=2)
            except Exception as e:
                print(f"[SigLIP Cache] Gagal simpan {rel_path}: {e}")

        # Per-label hybrid scoring
        per_label_result = {}
        hcfg = cfg["hybrid"]
        for i in range(n_labels):
            sw_raw   = hcfg["siglip_w"][i] if i < len(hcfg["siglip_w"]) else 0.5
            lw_raw   = hcfg["land_w"][i]   if i < len(hcfg["land_w"])   else 0.5
            total    = (sw_raw + lw_raw) or 1.0
            siglip_w = sw_raw / total
            land_w   = lw_raw / total

            siglip_scores = [round(norm_by_label[i][f].item(), 4) for f in range(n_frames)]

            if land_scores_per_frame:
                DEAD_THRESHOLD = 0.01
                valid_frames = [f for f in range(n_frames) if sum(land_scores_per_frame[f]) > DEAD_THRESHOLD] or list(range(n_frames))
                hybrid_scores = [
                    round(siglip_w * siglip_scores[f] + land_w * land_scores_per_frame[f][i], 4)
                    for f in range(n_frames)
                ]
                land_avg = round(sum(land_scores_per_frame[f][i] for f in valid_frames) / len(valid_frames), 4)

            else:
                hybrid_scores = siglip_scores
                land_avg      = None
                valid_frames  = list(range(n_frames))

            avg_score   = round(sum(hybrid_scores[f] for f in valid_frames) / max(len(valid_frames), 1), 4)
            siglip_avg  = round(sum(siglip_scores) / n_frames, 4)
            thr         = thresholds[i]
            frame_preds = [1 if s >= thr else 0 for s in hybrid_scores]
            vote_pos    = sum(frame_preds)
            # Prediction berdasarkan avg_score vs threshold — konsisten di semua path
            prediction  = 1 if (n_frames > 0 and avg_score >= thr) else 0

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
                "siglip_scores": siglip_scores,
                "land_scores": (
                    [round(land_scores_per_frame[f][i], 4) for f in range(n_frames)]
                    if land_scores_per_frame else None
                ),
                "threshold":    thr,
            }

        # CATATAN: strict_rules_bias & dominance_gap DIHAPUS (spec, bukan paper; dominance_gap
        # bertentangan dengan multi-label DAiSEE). Skor multi-label dibiarkan apa adanya.

        # Recompute predictions after adjustments
        for i in range(n_labels):
            r_lbl = per_label_result[i]
            r_lbl["frame_preds"] = [1 if s >= thresholds[i] else 0 for s in r_lbl["frame_scores"]]
            r_lbl["vote_pos"] = sum(r_lbl["frame_preds"])
            r_lbl["vote_neg"] = n_frames - r_lbl["vote_pos"]
            vf = valid_frames if land_scores_per_frame else list(range(n_frames))
            r_lbl["avg_score"] = round(
                sum(r_lbl["frame_scores"][f] for f in vf) / max(len(vf), 1), 4
            )
            r_lbl["prediction"] = 1 if (n_frames > 0 and r_lbl["avg_score"] >= thresholds[i]) else 0

        # Mutual exclusion (Bore↔Eng, Conf↔Frus) — sinkron dgn recalculate.py (galeri == final)
        if n_labels == 4:
            _apply_mutual_exclusion(per_label_result, n_frames)

        has_any_label = any(per_label_result[i]["prediction"] == 1 for i in range(n_labels))
        results.append({
            "per_label":  per_label_result,
            "n_frames":   n_frames,
            "thresholds": thresholds,
            "no_label":   not has_any_label,
        })
        frame_offset += n_frames

    # Cleanup GPU memory
    del inputs, logits_all, norm_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

