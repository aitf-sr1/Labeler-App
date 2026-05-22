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
        "v_dead_zone": 15.0,    # downward gaze_v dead zone — besar agar nunduk/ngetik tidak kena boredom
        "v_dead_zone_up": 5.0,  # upward gaze_v dead zone — kecil karena tatapan ke atas = tidak fokus ke layar
        "roll_dz": 5.0,         # dead zone roll sebelum masuk gaze_dev — kepala miring (roll) ikut geser arah pandang dari layar
    },
    "boredom": {
        "gaze_dead_zone": 5.0,       # gaze_dev < ini = tidak bosan dari gaze. Nunduk dilindungi v_dead_zone=15, bukan dead_zone ini.
        "gaze_range": 10.0,          # range gaze di atas dead zone — jenuh di dev=18°
        "blink_dead_zone": 0.20,     # eyeBlink < ini = tidak dihitung
        "blink_range": 0.50,         # range blink di atas dead zone
        "yawn_dead_zone": 0.20,      # jawOpen < ini = yawn_raw = 0. Mencegah mulut sedikit terbuka (istirahat/napas) dikuatkan expr_gate
        "yawn_threshold": 0.55,      # jawOpen di atas dead zone / ini = yawn_raw → 1.0. Butuh bukaan nyata untuk yawn penuh.
        "pitch_up_th": 20.0,         # pitch > ini = mulai pitch_up_v (kepala mendongak)
        "pitch_up_range": 25.0,      # range pitch_up_v di atas threshold
        "sig_expr_weight": 0.70,     # bobot max(blink, yawn, pitch_up)
        "blend_a": 0.85,             # koefisien campuran utama
        "blend_b": 0.15,             # koefisien campuran sekunder
        "expr_gaze_gate_th": 0.35,   # bore_gaze min untuk ekspresi boredom aktif
        "eye_wide_suppress": 0.30,   # mata lebar → kurangi skor boredom (attentif ≠ bosan)
        "squint_suppress": 0.30,     # mata sipit → kurangi skor boredom (sipit = konsentrasi ≠ bosan)
        "squint_blink_correction": 0.50,  # koreksi blink_avg dari kontribusi squint sebelum dihitung blink_v
        "teeth_gate_th": 0.20,       # gigi terlihat > ini = jawOpen BUKAN menguap (senyum/bicara)
        "smile_suppress": 0.40,      # senyum/gigi kelihatan → kurangi skor boredom
        "smile_gaze_max": 15.0,      # gaze_dev > ini → smile suppress nonaktif (noleh sambil ketawa = bosan)
        "chin_bore_th":    0.30,     # hand_chin minimum sebelum boost boredom aktif (menopang dagu)
        "chin_bore_range": 0.40,     # range di atas th → saturasi di hand_chin=0.70
        "chin_bore_max":   0.65,     # kontribusi boredom maksimal dari menopang dagu
        "yawn_bore_w": 0.75,         # bobot langsung yawn ke boredom. Menguap = bosan meski tatap layar.
        "yawn_strong_th": 0.50,      # jawOpen mentah minimum untuk bypass gaze gate (genuine yawn, bukan sekadar napas)
        "yawn_strong_range": 0.25,   # range di atas threshold; jawOpen < th tetap butuh gaze gate
    },
    "engagement": {
        "tegak_dead_zone": 8.0,      # gaze_dev dead zone (°) — gaze_dev tidak include roll (roll ditangani roll_gate)
        "tegak_range": 14.0,         # range setelah dead zone — gate=0 di gaze_dev≥22°
        "yaw_gate_th": 22.0,         # abs(yaw) mulai suppress engagement (°)
        "yaw_gate_range": 12.0,      # engagement nol di abs(yaw) ≥ 34°
        "roll_gate_th": 15.0,        # abs(roll) mulai suppress engagement (°) — naikkan dari 10° ke 15° (baca buku/layar miring itu alami)
        "roll_gate_range": 10.0,     # engagement nol di abs(roll) ≥ 25° — lebih toleran dari sebelumnya (16°)
        "blink_heavy_th": 0.50,      # eyeBlink > ini = droopy parah
        "blink_heavy_min": 0.30,     # engagement minimum jika droopy penuh
        "eye_wide_boost": 0.20,      # eyeWide → naikkan skor engagement (mata terbuka = fokus)
        "eye_squint_boost": 0.15,    # eyeSquint → sedikit naikkan engagement (sipit = konsentrasi aktif)
        "smile_boost": 0.30,         # senyum/gigi kelihatan → naikkan skor engagement
        "smile_gaze_max": 15.0,      # gaze_dev > ini → smile boost nonaktif
        "yawn_eng_th": 0.55,         # jawOpen >= ini mulai penalti engagement (threshold independen dari boredom)
        "yawn_eng_pen_w": 0.35,      # besar penalti engagement dari yawn (0.35 = lebih gentle)
        "pitch_gate_th": 15.0,       # pitch > ini mulai suppress engagement (mendongak ke atas)
        "pitch_gate_range": 15.0,    # engagement nol di pitch >= th+range (default: 30°)
        "gaze_fwd_bonus": 0.15,      # bonus engagement ketika gaze benar-benar ke depan (dalam dead zone)
        "bore_suppress_th": 0.45,    # bore < ini tidak suppress engagement (dead zone besar, abaikan noise)
        "bore_eng_suppress": 0.40,   # boredom landmark > bore_suppress_th → suppress engagement (dikurangi dari 0.5)
        "fwd_eng_min": 0.55,         # minimum engagement saat hadap depan — floor mutlak, tidak bisa ditekan
        "fwd_eng_gaze_max": 10.0,    # gaze_dev_eng > ini → fwd_eng floor mulai turun ke 0
        "look_dn_eng_th": 0.20,      # lookDown > ini = mulai boost engagement (diturunkan dari 0.25 supaya lebih sensitif)
        "look_dn_eng_boost": 0.30,   # boost engagement dari lihat bawah (naik dari 0.20 — nunduk baca = engaged)
        "look_dn_eng_yaw_max": 20.0, # yaw > ini → look_dn boost nonaktif
        "hand_chin_eng_pen": 0.50,   # menopang dagu → kurangi engagement (postur pasif, bukan aktif)
    },
    "confusion": {
        "iris_up_dead_zone": 0.20,   # -iris_y < ini = tidak terhitung iris_up_v
        "iris_up_range": 0.30,       # range iris_up_v di atas dead zone
        "look_up_threshold": 0.40,   # eyeLookUp / ini = look_up_v
        "look_dn_th": 0.30,          # lookDown > ini = mulai look_dn_v (diturunkan 0.40→0.30 supaya lebih sensitif)
        "look_dn_range": 0.30,       # range look_dn_v di atas threshold
        "look_dn_yaw_max": 15.0,     # lihat bawah + yaw > ini = bukan confusion (noleh sambil nunduk)
        "pitch_start": 10.0,         # pitch > ini = mulai pitch_cu
        "pitch_range": 15.0,         # range pitch_cu
        "brow_dn_th": 0.35,          # browDown avg / ini = brow_dn_v
        "brow_in_th": 0.30,          # browInnerUp / ini = brow_in_raw
        "brow_in_co_gate": 0.25,     # co_signal / ini = gate browInnerUp
        "smile_penalty_th": 0.15,    # mouthSmile > ini = mulai penalty jaw
        "smile_conf_gate_th": 0.20,  # mouthSmile >= ini → conf disupress ke 0 (senyum ≠ bingung)
        "jaw_start": 0.05,           # jawOpen < ini = jaw_val_conf = 0
        "jaw_peak": 0.25,            # titik puncak jaw_val_conf = 1
        "jaw_end": 0.40,             # jawOpen > ini = jaw_val_conf = 0
        "pucker_th": 0.30,           # mouthPucker / ini = pucker_co
        "roll_dead_zone": 8.0,       # abs(roll) < ini = tidak terhitung roll_v
        "roll_range": 15.0,          # range roll_v di atas dead zone
        "blend_a": 0.85,
        "blend_b": 0.15,
        "attentive_dead": 8.0,       # gaze_dev < ini = full attentive gate (1.0)
        "attentive_range": 20.0,     # gaze_dev > dead+range → gate turun ke floor
        "attentive_floor": 0.3,      # floor gate — bingung sebentar boleh lihat sekeliling
        "squint_conf_th": 0.15,           # eyeSquint avg >= ini = mulai aktif sebagai co_signal browInnerUp dan sinyal confusion langsung
        "squint_conf_range": 0.25,        # range squint_conf_v di atas threshold (saturasi di 0.40)
        "jaw_closed_th": 0.10,            # jawOpen < ini = jaw nyaris tertutup → aktifkan mouthUpperUp gate
        "mu_conf_th": 0.40,               # mouthUpperUpAvg >= ini mulai jadi sinyal konfusi (ketegangan bibir atas)
        "mu_conf_range": 0.30,            # range mu_conf_v di atas threshold (saturasi di 0.70)
        "bore_conf_suppress_bore": 0.40,  # boredom tinggi → conf ditekan (bosan = checked out, bukan aktif bingung)
        "frus_conf_suppress": 0.5,   # frustrasi tinggi → conf ditekan (mutual exclusive)
        "look_dn_th": 0.40,          # lookDown > ini = mulai look_dn_v (lihat bawah = bisa confusion)
        "look_dn_range": 0.30,       # range look_dn_v di atas threshold
        "look_dn_yaw_max": 15.0,     # lihat bawah + yaw > ini = bukan confusion
        "look_dn_boost": 0.35,       # boost langsung confusion saat lihat bawah + hadap depan
    },
    "frustration": {
        "brow_dn_th": 0.40,          # browDown avg / ini = br_fr
        "nose_sneer_th": 0.20,       # noseSneer / ini = ns_fr
        "cheek_squint_th": 0.40,     # cheekSquint avg / ini = ck_fr
        "mouth_press_th": 0.40,      # mouthPress avg / ini = lp_fr
        "eye_squint_th": 0.40,       # eyeSquint avg / ini = ey_fr
        "jaw_start": 0.10,           # jawOpen < ini = kontribusi rahang = 0
        "jaw_range": 0.20,           # range jw_fr di atas jaw_start
        "mouth_frown_th": 0.25,      # mouthFrown avg / ini = mf_fr (sudut mulut turun = frustrasi)
        "face_weight": 0.45,         # skala max sinyal wajah — dinaikkan 0.35→0.45 agar frustrasi tanpa tangan terdeteksi
        "hand_weight": 0.65,         # weighted-max: hand × 0.65 + face × 0.45, lalu max(weighted, hand, face)
        "blend_a": 0.85,
        "blend_b": 0.15,
    },
    "hybrid": {
        "empirical_bias": 3.5,
        "siglip_w": [0.40, 0.30, 0.40, 0.50],   # per label [Bore, Eng, Conf, Frus] — landmark dominan
        "land_w":   [0.60, 0.70, 0.60, 0.50],   # Eng/Conf: land 70/60% — landmark rules sudah dituning spesifik
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
