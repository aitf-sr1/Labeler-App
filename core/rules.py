"""
core/rules.py

Konfigurasi parameter kalkulasi landmark emosi.
Semua konstanta di compute_emotion_scores() dapat diubah dari sini.
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
        "gaze_dead_zone": 5.0,       # gaze_dev < ini = tidak bosan dari gaze
        "gaze_range": 10.0,          # range gaze di atas dead zone — jenuh di dev=18°
        "blink_dead_zone": 0.20,     # eyeBlink < ini = tidak dihitung
        "blink_range": 0.50,         # range blink di atas dead zone
        # Craig et al. (2008): AU43 (eye closure) = primary boredom signal
        "sig_expr_weight": 0.70,     # bobot blink (AU43) dalam expr path
        "blend_a": 0.85,             # koefisien campuran utama
        "blend_b": 0.15,             # koefisien campuran sekunder
        "expr_gaze_gate_th": 0.35,   # bore_gaze min untuk blink gated aktif
        # Craig et al. (2008): tidak ada AU yang memvalidasi suppressor untuk boredom — hanya AU43
        "squint_blink_correction": 0.50,  # koreksi teknis: squint sedikit menutup mata → koreksi AU43 reading
        "frus_bore_suppress_th": 0.40,  # frus > ini mulai suppress boredom (tegang ≠ bosan)
        "frus_bore_suppress":    0.45,  # max reduksi boredom oleh frustration — D'Mello 2012: Frus→Bore significant
        # Craig et al. (2008): AU43 (eye closure) = primary boredom signal, independent of gaze
        "blink_direct_th": 0.45,    # eyeBlink_corrected > ini → kontribusi langsung ke boredom (AU43, tanpa gaze gate)
        "blink_direct_w":  0.45,    # bobot kontribusi langsung blink ke boredom (Craig2008: AU43 primary signal)
    },
    "engagement": {
        # Whitehill et al. (2014): engagement = forward gaze + eye openness (holistic appearance)
        # Level 1: "looking away from computer, eyes completely closed" = NOT engaged
        # Level 2: "eyes barely open, clearly not 'into' the task" = NOT engaged
        "tegak_dead_zone": 8.0,      # gaze_dev dead zone (°)
        "tegak_range": 14.0,         # range setelah dead zone
        "yaw_gate_th": 22.0,         # abs(yaw) mulai suppress engagement (°)
        "yaw_gate_range": 12.0,      # engagement nol di abs(yaw) ≥ 34°
        "roll_gate_th": 15.0,        # abs(roll) mulai suppress engagement (°)
        "roll_gate_range": 10.0,     # engagement nol di abs(roll) ≥ 25°
        "blink_heavy_th": 0.50,      # AU43 eye closure (Whitehill level 1-2): threshold droopy eyes
        "blink_heavy_min": 0.30,     # engagement minimum jika droopy penuh
        "eye_wide_boost": 0.20,      # inverse of "eyes barely open" (Whitehill level 2)
        "pitch_gate_th": 15.0,       # pitch > ini mulai suppress (Whitehill: "looking away from computer")
        "pitch_gate_range": 15.0,    # engagement nol di pitch >= th+range
        # D'Mello & Graesser (2012): Boredom dan Engagement near-mutually exclusive
        "bore_suppress_th": 0.45,
        "bore_eng_suppress": 0.40,
        # D'Mello & Graesser (2012): "Confusion → Engagement/Flow transition significant" (productive struggle)
        "conf_eng_suppress_th": 0.50,
        "conf_eng_suppress":    0.35,
    },
    "confusion": {
        # Craig et al. (2008) Table 2: AU4 (brow lowerer) 95%, AU7 (lid tightener) 78%,
        # AU4+AU7 co-occurrence 73%, AU12 (questioning smile) 95% secondary.
        # Grafsgaard et al. (2011): AU4 validated via HMM as primary confusion predictor.
        "brow_dn_th": 0.35,          # browDown avg / ini = brow_dn_v (AU4 brow lowerer, Craig2008 95%)
        "au7_th": 0.15,              # eyeSquint avg min untuk AU7 (lid tightener) co-signal (Craig2008 78%)
        "au4_au7_co_w": 0.50,        # weight co-occurrence AU4+AU7 (Craig2008 73%)
        # Craig et al. (2008): AU12 questioning smile co-occurs 95% — gate floor prevents zeroing confusion
        "smile_conf_gate_th": 0.35,  # mouthSmile >= ini → mulai gate confusion
        "smile_conf_gate_floor": 0.30,  # confusion tetap ≥30% meski senyum penuh (AU12 co-occurs)
        "blend_a": 0.85,
        "blend_b": 0.15,
        # D'Mello & Graesser (2012): "Confusion→Boredom occurred at chance" — boredom suppresses confusion
        "bore_conf_suppress_bore": 0.40,
    },
    "frustration": {
        # Craig et al. (2008): AU1 (outer brow raise) + AU2 (inner brow raise) = PRIMARY frustration signals (100% coverage)
        "brow_outer_up_th": 0.20,    # browOuterUp avg / ini = bou_fr (AU1, Craig2008 primary)
        "brow_inner_up_th": 0.20,    # browInnerUp / ini = biu_fr (AU2, Craig2008 primary)
        # brow_raise_direct_w: Craig2008 AU1+AU2 mendapat face weight lebih tinggi dari secondary signals.
        "brow_raise_direct_w": 0.65, # direct weight untuk AU1+AU2 primary (Craig2008: 100% coverage)
        # Grafsgaard et al. (2013): AU4 (brow lowering) positively correlated with frustration
        "brow_dn_th": 0.40,          # browDown avg / ini = br_fr (AU4 — secondary, Grafsgaard 2013)
        "face_weight": 0.45,         # skala AU4 secondary signal
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "hybrid": {
        "empirical_bias": 3.5,
        # Whitehill et al.: engagement detection lebih akurat dengan holistic appearance daripada AU individual.
        # SigLIP (vision-language model) lebih baik untuk Engagement → naikkan siglip_w 0.35→0.40.
        # Bartlett (2006): AU-based detection reliable untuk Boredom/Frustration → pertahankan land_w tinggi.
        "siglip_w": [0.30, 0.40, 0.45, 0.30],   # per label [Bore, Eng, Conf, Frus] — Eng: 0.35→0.40 (Whitehill)
        "land_w":   [0.70, 0.60, 0.55, 0.70],   # Bore↑0.70 Frus↑0.70: tangan/gaze sangat reliable; Eng: 0.65→0.60
        "dual_label_gap": 0.12,       # Min gap antara top vs secondary label untuk mempertahankan dual-label (spec: most images = 1 label)
        "strict_rules_strength": 0.20, # Kekuatan soft bias dari strict labeling rules (0 = off, 0.20 = moderate)
        "restless_bonus_max": 0.0,   # disabled — heuristik tanpa basis definisi semantik
        "restless_std_min": 3.0,
        "restless_std_range": 7.0,
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
