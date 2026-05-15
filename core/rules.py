"""
core/rules.py

Konfigurasi parameter kalkulasi landmark emosi.
Semua konstanta di compute_emotion_scores() dapat diubah dari sini.
"""

import os
import json

DEFAULT_RULES = {
    "gaze": {
        "scale_h": 35.0,        # iris_x × ini = horizontal gaze contribution (°)
        "scale_v": 25.0,        # iris_y × ini = vertical gaze contribution (°)
        "iris_side_mult": 2.0,  # floor iris side = iris_x × scale_h × ini
        "v_dead_zone": 15.0,    # gaze_v < ini = tidak berkontribusi ke gaze_dev
    },
    "boredom": {
        "gaze_dead_zone": 8.0,       # gaze_dev < ini = tidak bosan dari gaze
        "gaze_range": 12.0,          # range gaze di atas dead zone
        "pitch_nunduk_th": 12.0,     # pitch < -ini = kepala nunduk
        "pitch_nunduk_range": 8.0,   # range bore_pitch_down setelah nunduk_th
        "pitch_up_th": 8.0,          # pitch > ini = kepala mendongak
        "pitch_up_range": 10.0,      # range bore_pitch_up setelah pitch_up_th
        "blink_dead_zone": 0.20,     # eyeBlink < ini = tidak dihitung
        "blink_range": 0.50,         # range blink di atas dead zone
        "yawn_threshold": 0.35,      # jawOpen / ini = yawn_v (max 1)
        "sig_expr_weight": 0.70,     # bobot max(blink, yawn, pitch_up)
        "blend_a": 0.85,             # koefisien campuran utama
        "blend_b": 0.15,             # koefisien campuran sekunder
        "frus_suppress": 0.70,       # seberapa kuat frus_restless tekan bore_gaze
    },
    "engagement": {
        "nunduk_gate_range": 5.0,    # saat nunduk: gate = 1 - gaze_h_eff / ini
        "tegak_dead_zone": 3.0,      # saat tegak: gaze_dev dead zone (°)
        "tegak_range": 12.0,         # saat tegak: range setelah dead zone
        "blink_heavy_th": 0.50,      # eyeBlink > ini = droopy parah
        "blink_heavy_min": 0.30,     # engagement minimum jika droopy penuh
    },
    "confusion": {
        "iris_up_dead_zone": 0.15,   # -iris_y < ini = tidak terhitung iris_up_v
        "iris_up_range": 0.30,       # range iris_up_v di atas dead zone
        "look_up_threshold": 0.35,   # eyeLookUp / ini = look_up_v
        "pitch_start": 5.0,          # pitch > ini = mulai pitch_cu
        "pitch_range": 15.0,         # range pitch_cu
        "brow_dn_th": 0.23,          # browDown avg / ini = brow_dn_v
        "brow_in_th": 0.30,          # browInnerUp / ini = brow_in_raw
        "brow_in_co_gate": 0.25,     # co_signal / ini = gate browInnerUp
        "smile_penalty_th": 0.15,    # mouthSmile > ini = mulai penalty
        "jaw_start": 0.05,           # jawOpen < ini = jaw_val_conf = 0
        "jaw_peak": 0.25,            # titik puncak jaw_val_conf = 1
        "jaw_end": 0.40,             # jawOpen > ini = jaw_val_conf = 0
        "pucker_th": 0.30,           # mouthPucker / ini = pucker_raw
        "pucker_gate_th": 0.30,      # max(jaw,brow) / ini = pucker_gate
        "gaze_gate_dead": 8.0,       # gaze_h_eff < ini = tidak ada gate
        "gaze_gate_range": 12.0,     # range conf_gaze_gate di atas dead zone
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "frustration": {
        "brow_dn_th": 0.40,          # browDown avg / ini = br_fr
        "nose_sneer_th": 0.20,       # noseSneer / ini = ns_fr
        "cheek_squint_th": 0.40,     # cheekSquint avg / ini = ck_fr
        "mouth_press_th": 0.40,      # mouthPress avg / ini = lp_fr
        "eye_squint_th": 0.40,       # eyeSquint avg / ini = ey_fr
        "jaw_start": 0.10,           # jawOpen < ini = kontribusi rahang = 0
        "jaw_range": 0.20,           # range jw_fr di atas jaw_start
        "restless_dead": 5.0,        # yaw-eye dissonance < ini = tidak restless
        "restless_range": 10.0,      # range frus_restless
        "restless_w": 0.50,          # bobot frus_restless dalam base_frus
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "hybrid": {
        "empirical_bias": 3.5,
        "siglip_w": [0.50, 0.50, 0.60, 0.50],   # per label [Bore, Eng, Conf, Frus]
        "land_w":   [0.50, 0.50, 0.40, 0.50],
        "restless_bonus_max": 0.15,
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
