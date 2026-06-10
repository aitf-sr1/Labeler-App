"""
core/rules.py

Konfigurasi parameter kalkulasi landmark emosi.
Semua konstanta di compute_emotion_scores() dapat diubah dari sini.

ARSITEKTUR PIPELINE (branch feature/mediapipe-only-accurate):
  MediaPipe FaceLandmarker → blendshape → AU normalisasi → skor landmark
  MediaPipe HandLandmarker → jumlah tangan di wajah → cue tangan
  SigLIP2 → skor visual holistik
  Hybrid = α × SigLIP + β × Landmark

Tidak ada py-feat, tidak ada subprocess eksternal. Semua inferensi sinkron
dalam satu proses, satu virtual environment.
"""

import os
import json

DEFAULT_RULES = {
    "gaze": {
        "scale_h": 35.0,              # iris_x × ini = horizontal gaze contribution (°)
        "scale_v": 25.0,              # iris_y × ini = vertical gaze contribution (°)
        "iris_side_mult": 2.0,        # floor iris side = iris_x × scale_h × ini
        "v_dead_zone": 15.0,          # downward gaze_v dead zone — besar agar nunduk/ngetik tidak kena boredom
        "v_dead_zone_up": 5.0,        # upward gaze_v dead zone — kecil karena tatapan ke atas = tidak fokus ke layar
        "roll_dz": 5.0,               # dead zone roll sebelum masuk gaze_dev — kepala miring (roll) ikut geser arah pandang dari layar
        "iris_blink_suppress_th": 0.60,  # blink_corrected di atas ini → iris_y mulai di-suppress (artifact mata menutup)
        "iris_blink_zero_th":     0.0,   # 0 = disabled; > 0 → iris_y = 0 sepenuhnya jika blink_corr >= nilai ini (squint fokus ekstrem)
    },
    "boredom": {
        # DIPERKETAT (lebih sensitif untuk atas/samping; bawah TETAP dikecualikan via gaze_dev_bore):
        "gaze_dead_zone": 5.0,       # gaze_dev < ini = tidak bosan dari gaze
        "gaze_range": 8.0,           # boredom penuh di dev=13° (dulu 10 → 15°)
        # AU43 blendshape eyeBlink — ter-normalisasi 0–1 via au["AU43"]
        "blink_dead_zone": 0.20,     # AU43 < ini = tidak dihitung (dead zone normalisasi)
        "blink_range": 0.50,         # range AU43 di atas dead zone
        # Craig et al. (2008): AU43 (eye closure) = primary boredom signal
        "sig_expr_weight": 0.70,     # bobot AU43 dalam expr path
        "blend_a": 0.85,
        "blend_b": 0.15,
        "expr_gaze_gate_th": 0.35,   # bore_gaze min untuk AU43-gated aktif
        "fwd_yaw_th":    4.0,        # gaze menjauh >4° mulai picu boredom (dulu 5)
        "fwd_yaw_range": 4.0,
        # Craig et al. (2008): AU43 = primary signal, independent of gaze
        "blink_direct_th": 0.45,
        "blink_direct_w":  0.45,
    },
    "engagement": {
        # Whitehill et al. (2014): engagement = forward gaze + eye openness (holistic appearance)
        # Level 1: "looking away from computer, eyes completely closed" = NOT engaged
        # Level 2: "eyes barely open, clearly not 'into' the task" = NOT engaged
        # DIPERKETAT: hadap atas/samping (bukan ke layar) lebih cepat kurangi engagement
        # (Whitehill 2014 "looking away from computer" = NOT engaged). Nunduk TIDAK dihukum (Sümer 2021).
        "tegak_dead_zone": 5.0,      # gaze_dev dead zone (°) — dulu 8
        "tegak_range": 12.0,         # dulu 14
        "yaw_gate_th": 14.0,         # menoleh samping >14° mulai kurangi engagement — dulu 22
        "yaw_gate_range": 12.0,      # engagement nol di yaw 26°
        "roll_gate_th": 15.0,
        "roll_gate_range": 10.0,
        # AU43 via au[] — ter-normalisasi 0–1 dari blendshape eyeBlink
        "blink_heavy_th": 0.50,      # AU43 threshold droopy eyes (Whitehill level 1-2)
        "blink_heavy_min": 0.30,
        "pitch_gate_th": 10.0,       # mendongak >10° mulai kurangi engagement (Whitehill) — dulu 15
        "pitch_gate_range": 15.0,
        # D'Mello & Graesser (2012) + Gupta et al. (2016) DAiSEE: Bore↔Eng near-mutually exclusive
        "bore_suppress_th": 0.45,
        "bore_eng_suppress": 0.40,
        # Gupta et al. (2016) DAiSEE: default nonaktif — lihat komentar kode
        "bore_eng_low_boost": 0.0,
        "bore_eng_low_th":    0.25,
    },
    "action_units": {
        # Normalisasi baseline-relative blendshape MediaPipe → intensitas sinyal emosi.
        # intensity = clamp((raw - neutral)/(active - neutral), 0, 1). Lihat core/blendshape_features.py.
        # Chain: Craig 2008 (AU→emosi) + Turrisi 2026 (BF→AU, κ=0.92) → blendshape langsung.
        # Anchor = kalibrasi empiris dari 21.204 frame raw_cache dataset ini.
        # neutral ≈ median populasi (otot diam), active ≈ p90–p99 (AU aktif penuh).
        # AU1/AU2 active diturunkan 0.88→0.84 / 0.86→0.82 → alis-naik lebih sensitif (intensity penuh
        # tercapai lebih cepat) memperkuat deteksi Frustration. Neutral tetap (otot diam ≈0.46 = 0 intensity).
        "AU1_neutral": 0.46, "AU1_active": 0.84,    # browInnerUp (inner brow raise — Frustration AU1)
        "AU2_neutral": 0.47, "AU2_active": 0.82,    # browOuterUp (outer brow raise — Frustration AU2)
        # browDown median 0.001 → stretch AGRESIF ke active=0.05 agar AU4 terdeteksi.
        # Basis: Aldenhoven 2026 Table 1 (browDownL/R → AU4) + Turrisi 2026 (κ=1.00).
        # TIDAK ada noseSneer booster — noseSneer = AU9 per Aldenhoven, bukan AU4.
        "AU4_neutral": 0.001, "AU4_active": 0.05,   # browDown (Confusion AU4) — stretch agresif
        "AU7_neutral": 0.30, "AU7_active": 0.52,    # eyeSquint   (lid tightener — Confusion AU7)
        "AU12_neutral": 0.05, "AU12_active": 0.55,  # mouthSmile  (questioning smile — gate Confusion)
        "AU14_neutral": 0.002, "AU14_active": 0.08, # mouthDimple (dimpler — Grafsgaard2013 frustration)
        # AU25+AU26: Namba 2024 — "most significant component" thinking face (Component 2).
        # MediaPipe menyediakan mouthOpen (≈AU25) dan jawOpen (≈AU26) secara langsung.
        "AU25_neutral": 0.05, "AU25_active": 0.50,  # mouthOpen  (Lips Part — mulut terbuka sedikit)
        "AU26_neutral": 0.05, "AU26_active": 0.60,  # jawOpen    (Jaw Drop  — rahang turun)
        "AU43_neutral": 0.12, "AU43_active": 0.55,  # eyeBlink   (eye closure — Boredom AU43)
        # eyeLookDown gating (Turrisi 2026: eyeLookDownL/R = AU64 gaze direction, BUKAN AU7/AU43).
        # Saat lihat bawah, eyeSquint+eyeBlink naik secara mekanik (bukan ekspresi) → suppress.
        # Gate linear dari gate_start (mulai suppress) ke gate_full (full suppress = 100%).
        "eyeLookDown_gate_start": 0.25,  # eyeLookDown > 0.25 → mulai suppress AU7 & AU43
        "eyeLookDown_gate_full":  0.60,  # eyeLookDown > 0.60 → full suppress (AU7=0, AU43=0)
    },
    "confusion": {
        # Craig et al. (2008) Table 2: AU4 (brow lowerer) 95%, AU7 (lid tightener) 78%,
        # AU4+AU7 co-occurrence 73%. Grafsgaard (2011): AU4 via HMM as primary predictor.
        "au4_au7_co_w": 0.50,
        "au7_alone_w": 0.78,
        # Hand-over-face (any hand) → cue Confusion. Dua paper saling menguatkan:
        #   Mahmoud 2011: index-finger touching face = 12/15 "thinking" + 2 "unsure" (≈Confusion), KUANTITATIF;
        #   Dong 2026 (ConfusionBench): hand-to-face (touch chin / press forehead) → "thinking, frustration, hesitation";
        #   Behera 2020: HoF naik saat difficulty ↑; Mahmoud 2016: HoF = "cognitive mental states".
        # max(hand_one, hand_two) karena paper tidak bedakan jumlah untuk Confusion.
        # Dinaikkan 0.40→0.50→0.78 = SETARA au7_alone_w (sinyal AU diskrit terkuat). Basis: Mahmoud 2011
        # coverage 14/15 ≈ 93% bahkan LEBIH tinggi dari AU7 (78%) → hand layak jadi cue KUAT, bukan lemah.
        "hand_conf_w": 0.95,         # cue Confusion KUAT — Mahmoud 2011 14/15=93%; dinaikkan agar override smile gate
        # Namba et al. (2024): mulut terbuka (AU25+AU26) = "most significant component" thinking face.
        # Chain: thinking face (Namba) + thinking = Confusion (D'Mello 2012).
        # Dinaikkan 0.25→0.35→0.78 = SETARA au7_alone_w. Basis: Namba "most significant" → cue KUAT, bukan lemah.
        # Dilindungi geometric-mean AU25·AU26 (butuh lips-part + jaw-drop) → tidak fire dari mulut sedikit gerak.
        "mouth_open_conf_w": 0.78,   # cue Confusion KUAT (≈ AU7) — Namba 2024 "most significant component"
        # AU12 questioning smile sebagai gate (bukan sinyal positif) — floor cegah zeroing confusion
        "smile_conf_gate_th": 0.65,       # dinaikkan: smile lemah (AU12<0.65) tidak agresif suppress Confusion
        "smile_conf_gate_floor": 0.30,
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "frustration": {
        # Craig et al. (2008): AU1 (inner brow raise) + AU2 (outer brow raise) = PRIMARY frustration signals (100% coverage)
        # Note: Craig 2008 Table 2 uses non-standard numbering; Grafsgaard 2013 confirms standard FACS: AU1=inner, AU2=outer.
        # MediaPipe: browInnerUp = AU1 (inner), browOuterUpLeft/Right = AU2 (outer).
        # Intensitas AU1/AU2 dihitung baseline-normalized (core/blendshape_features.py): alis "diam"
        # MediaPipe ~0.46 kini = 0 intensity, sehingga frustration TIDAK lagi over-fire saat netral.
        # Dinaikkan 0.65→0.70: kompensasi hilangnya py-feat AU4 range (sebelumnya 0.31–0.61),
        # menguatkan sinyal primer AU1+AU2 sesuai Craig 2008 "100% coverage".
        # Dinaikkan 0.65→0.70→0.85: AU1+AU2 = sinyal PRIMER dgn coverage 100% (Craig 2008 — basis
        # terkuat yg mungkin) → layak jadi bobot tertinggi. Kompensasi MediaPipe-only + memperjelas
        # frustrasi yg sebelumnya kurang terdeteksi (alis-naik kini jadi pemicu kuat).
        # DIKEMBALIKAN ke 0.85 (2026-06): grid search pada 18.618 frame vs label manual menunjukkan
        # 0.85 memberi F1 Frustration terbaik (~0.57); penurunan ke 0.55 (anekdot 1 video) menjatuhkan
        # recall Frustration ke ~0.05. Aggregate ground-truth menang atas kasus tunggal.
        "brow_raise_direct_w": 0.85,
        # Grafsgaard et al. (2013): "Action Unit 4 (brow lowering) was POSITIVELY correlated with
        # frustration" + AU14 (dimpling) juga positif.
        # Dinaikkan 0.60→0.65: kompensasi hilangnya py-feat — lebih mengandalkan AU4+AU14 MediaPipe.
        "face_weight": 0.65,         # AU4/AU14 pendukung (Grafsgaard 2013)
        # Frustration → HANYA 2-tangan (hand_two). Grafsgaard 2013b: two-hands-to-face ↔ self-efficacy
        # RENDAH (signifikan). Dikuatkan: Nojavanasghari 2017 (Hand2Face) & Dong 2026 (ConfusionBench).
        # Dinaikkan 0.30→0.40: konsisten dgn penguatan cue tangan (sinyal pendukung frustrasi).
        "hand_frus_w": 0.40,
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "hybrid": {
        "empirical_bias": 3.5,
        # Prinsip: landmark MediaPipe (AU-diskrit, Craig 2008) = sumber primer untuk SEMUA emosi;
        # SigLIP = cross-check holistik independen (Zhai 2023) — berguna saat landmark sulit
        # (oklusi tangan, wajah miring). Bobotnya = kalibrasi empiris yang SAH (seperti threshold).
        # Bobot DIVALIDASI grid search vs label manual (18.618 frame), 2026-06:
        # - Boredom   (land 0.75): gaze+AU43 primer + SigLIP 0.25.
        # - Engagement(land 0.50): 0.5/0.5 memberi F1 0.905 (sudah sangat baik) — TIDAK diubah
        #   (perubahan ke 0.25/0.75 anekdotal tidak terbukti membaik & berisiko ubah Eng).
        # - Confusion (land 0.70): siglip 0.30 terbukti F1 terbaik (~0.83); siglip 0.10 menurunkannya.
        # - Frustration(land 0.70): siglip 0.30. SigLIP lemah utk Frus (F1 0.26) → landmark dominan.
        "siglip_w": [0.25, 0.5, 0.30, 0.30],   # [Bore, Eng, Conf, Frus]
        "land_w":   [0.75, 0.5, 0.70, 0.70],
        # CATATAN: eng_gaze_gate eksternal DIHAPUS (lihat core/inference.py). Prinsip Whitehill
        # "looking away = not engaged" sudah ditegakkan di dalam skor landmark Engagement
        # (gaze_dev_eng + yaw_gate + pitch_gate) yang berbobot land_w[Eng]=0.75 → gate eksternal
        # redundan & menimbulkan divergensi galeri vs final.
        # Mutual exclusion hard-XOR (Bore↔Eng, Conf↔Frus) diterapkan konsisten di inference.py
        # & recalculate.py — tapi itu PENYEDERHANAAN DESAIN, bukan verbatim paper. Yang
        # paper-grounded: komplementer Bore↔Eng via soft `bore_eng_suppress` (DAiSEE/D'Mello).
        # Conf↔Frus XOR = basis lemah (paper hanya transisi temporal). Lihat DESIGN_RATIONALE §11.
    },
}


def load_rules(path: str) -> dict:
    """Baca rules.json, merge dengan DEFAULT_RULES untuk key yang hilang."""
    if not os.path.exists(path):
        return _deep_copy(DEFAULT_RULES)
    try:
        with open(path) as f:
            data = json.load(f)
        return _deep_merge(DEFAULT_RULES, data)
    except Exception:
        return _deep_copy(DEFAULT_RULES)


def save_rules(path: str, rules: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def _deep_copy(d):
    return json.loads(json.dumps(d))


def _deep_merge(base: dict, override: dict) -> dict:
    result = _deep_copy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
