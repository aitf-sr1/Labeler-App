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
        "eye_wide_suppress": 0.30,   # mata lebar → kurangi skor boredom (attentif ≠ bosan)
        "squint_suppress": 0.30,     # mata sipit → kurangi skor boredom (sipit = konsentrasi ≠ bosan)
        "brow_inner_suppress_th": 0.45,  # browInnerUp > ini → mulai suppress boredom (waspada/fokus ≠ bosan)
        "brow_inner_suppress":    0.55,  # max reduksi boredom dari browInnerUp tinggi
        "squint_blink_correction": 0.50,  # koreksi blink_avg dari kontribusi squint
        "smile_suppress": 0.40,      # senyum → kurangi skor boredom
        "smile_gaze_max": 15.0,      # gaze_dev > ini → smile suppress nonaktif
        "frus_bore_suppress_th": 0.40,  # frus > ini mulai suppress boredom (tegang ≠ bosan)
        "frus_bore_suppress":    0.45,  # max reduksi boredom oleh frustration — D'Mello 2012: Frus→Bore significant
        # Craig et al. (2008): AU43 (eye closure) = primary boredom signal, independent of gaze
        "blink_direct_th": 0.45,    # eyeBlink_corrected > ini → kontribusi langsung ke boredom (AU43, tanpa gaze gate)
        "blink_direct_w":  0.45,    # bobot kontribusi langsung blink ke boredom (Craig2008: AU43 primary signal)
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
        "pitch_gate_th": 15.0,       # pitch > ini mulai suppress engagement (mendongak ke atas)
        "pitch_gate_range": 15.0,    # engagement nol di pitch >= th+range (default: 30°)
        "gaze_fwd_bonus": 0.15,      # bonus engagement ketika gaze benar-benar ke depan (dalam dead zone)
        "bore_suppress_th": 0.45,    # bore < ini tidak suppress engagement (dead zone besar, abaikan noise)
        "bore_eng_suppress": 0.40,   # boredom landmark > bore_suppress_th → suppress engagement (dikurangi dari 0.5)
        "fwd_eng_min": 0.35,         # minimum engagement saat hadap depan — diturunkan 0.55→0.35 agar confusion bisa menang
        "fwd_eng_gaze_max": 10.0,    # gaze_dev_eng > ini → fwd_eng floor mulai turun ke 0
        "look_dn_eng_th": 0.20,      # lookDown > ini = mulai boost engagement (diturunkan dari 0.25 supaya lebih sensitif)
        "look_dn_eng_boost": 0.30,   # boost engagement dari lihat bawah (naik dari 0.20 — nunduk baca = engaged)
        "look_dn_eng_yaw_max": 20.0, # yaw > ini → look_dn boost nonaktif
        # D'Mello & Graesser (2012): Confusion dan Engagement dapat co-exist dalam "productive struggle".
        # Mahasiswa bingung tapi masih actively engaged dengan konten = valid state.
        # Turunkan suppression agar confusion tidak terlalu agresif membunuh engagement.
        "conf_eng_suppress_th": 0.50,   # raised 0.40→0.50: D'Mello2012 productive struggle = conf+eng co-occur
        "conf_eng_suppress":    0.35,   # reduced 0.55→0.35: confusion tidak harus membunuh engagement
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
        "brow_dn_th": 0.35,          # browDown avg / ini = brow_dn_v (AU4 brow lowerer)
        "brow_in_th": 0.30,          # browInnerUp / ini = brow_in_raw
        "brow_in_co_gate": 0.25,     # co_signal / ini = gate browInnerUp
        # Craig et al. (2008): AU12 (lip corner puller = mouthSmile) co-occurs with confusion 95% of episodes.
        "smile_conf_gate_th": 0.35,  # mouthSmile >= ini → conf disupress (raised: AU12 can co-occur per Craig2008)
        "blend_a": 0.85,
        "blend_b": 0.15,
        "attentive_dead": 8.0,       # gaze_dev < ini = full attentive gate (1.0)
        "attentive_range": 20.0,     # gaze_dev > dead+range → gate turun ke floor
        "attentive_floor": 0.3,      # floor gate — bingung sebentar boleh lihat sekeliling
        "squint_conf_th": 0.15,           # eyeSquint avg >= ini = mulai aktif sebagai co_signal browInnerUp dan sinyal confusion langsung
        "squint_conf_range": 0.25,        # range squint_conf_v di atas threshold (saturasi di 0.40)
        "bore_conf_suppress_bore": 0.40,  # boredom tinggi → conf ditekan (bosan = checked out, bukan aktif bingung)
        "look_dn_boost": 0.15,       # boost confusion saat lihat bawah — dikurangi dari 0.50 (spec Rule 3: nunduk = engagement)
        # Craig et al. (2008): AU4 (browDown) + AU7 (eyeSquint/lid tightener) co-occurrence = 73% confusion coverage
        "au4_au7_co_w": 0.50,        # weight co-occurrence AU4+AU7 sebagai sinyal confusion eksplisit
        "au7_th": 0.15,              # eyeSquint avg min untuk dihitung sebagai AU7 (lid tightener) co-signal
        # Craig et al. (2008): AU2 (browInnerUp) = frustration signal, bukan confusion.
        # Suppress brow_in_v untuk confusion ketika AU1 (browOuterUp) juga aktif (= pola frustration).
        "biu_au1_check_th": 0.25,    # browOuterUp threshold untuk mendeteksi pola frustration AU1+AU2
        "biu_au1_suppress": 0.80,    # seberapa besar browInnerUp disuppress untuk confusion saat AU1 aktif
        # Craig et al. (2008): AU12 (mouthSmile) co-occurs dengan confusion 95% episodes (questioning smile).
        "smile_conf_gate_floor": 0.30,  # floor gate senyum — confusion tetap ≥30% meski senyum penuh
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
