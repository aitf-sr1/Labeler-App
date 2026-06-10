"""
core/blendshape_features.py

Konversi blendshape MediaPipe FaceLandmarker → intensitas sinyal FACS
untuk deteksi emosi belajar (Boredom, Engagement, Confusion, Frustration).

CHAIN RULE — LANDASAN PAPER
────────────────────────────
Emosi belajar didefinisikan dalam FACS Action Units oleh Craig et al. (2008).
Turrisi et al. (2026) memvalidasi mapping blendshape MediaPipe → AU secara
expert-validated (10 psikolog klinis bersertifikat, κ=0.92, 88% unanimous):

    Mapping dikonfirmasi oleh Aldenhoven 2026 Table 1 + Turrisi 2026 (κ=0.92):

    eyeSquintLeft/Right  → AU7  (Lid Tightener)    Aldenhoven Table1 + Turrisi κ≈0.93
    browDownLeft/Right   → AU4  (Brow Lowerer)      Aldenhoven Table1 + Turrisi κ=1.00
    browInnerUp          → AU1  (Inner Brow Raiser) Aldenhoven Table1 + Turrisi κ=1.00
    browOuterUpL/R       → AU2  (Outer Brow Raiser) Aldenhoven Table1 + Turrisi κ=1.00
    eyeBlinkLeft/Right   → AU43 (Eyes Closed)       Turrisi: "codify both AU43 and AU45"
    mouthSmileL/R        → AU12 (Lip Corner Puller) Aldenhoven Table1 + Turrisi
    mouthDimpleL/R       → AU14 (Dimpler)           Aldenhoven Table1 (contempt AU14)
    jawOpen              → AU26 (Jaw Drop)           Aldenhoven Table1 + Turrisi
    jawOpen (proxy AU25) → AU25 (Lips Part)         Kompromi — ARKit tidak punya
                                                     "mouthOpen" terpisah; Namba 2024
                                                     justifikasi konsep mulut-terbuka
    eyeLookDownL/R       → AU64 (Eyes Turn Down)    Turrisi: "jointly codify AU64"
                                                     (gaze direction ≠ AU7/AU43)

    TIDAK dipakai: noseSneer → AU9 (Nose Wrinkler) per Aldenhoven Table 1 &
    Turrisi (nose region κ=−0.111, tidak ada agreement). noseSneer BUKAN AU4.

Chain: Craig (AU→emosi belajar) + Turrisi (BF→AU, κ=0.92) → BF langsung
merepresentasikan sinyal emosi belajar dengan justifikasi paper ganda.

Aldenhoven et al. (2026, Sensors) memvalidasi bahwa native MediaPipe/ARKit
blendshapes dipakai langsung (tanpa konversi AU intermediate) untuk klasifikasi
emosi via cosine similarity dan menghasilkan akurasi 68.3% — melebihi human
rater (58.9%). Ini menguatkan penggunaan native MediaPipe blendshapes sebagai
sumber primer (bukan model aproksimasi mp_blendshapes).

GATING eyeLookDown (AU64) — TURRISI 2026
─────────────────────────────────────────
Ketika siswa melihat ke bawah (membaca buku/keyboard), MediaPipe melaporkan
eyeLookDownLeft/Right tinggi (AU64 = gaze direction). Dalam kondisi ini,
kelopak mata sedikit menutup secara mekanik mengikuti arah pandang, sehingga:
  - eyeSquint (→AU7) naik → falsely trigger Confusion
  - eyeBlink  (→AU43) naik → falsely trigger Boredom
Turrisi 2026 menegaskan AU64 adalah gaze direction, BUKAN eyelid action (AU7
= lid tightening dari usaha kognitif, AU43 = eye closure dari mengantuk).
Gate: kurangi AU7 dan AU43 secara proporsional saat eyeLookDown tinggi.

NORMALISASI BASELINE-RELATIVE
──────────────────────────────
Pada distribusi 21.204 frame nyata: browInnerUp median 0.46, eyeSquint median
0.31 — blendshape "diam" tidak bernilai 0. Tanpa normalisasi, AU1/AU2 over-fire
dan AU4 tidak pernah terdeteksi. Normalisasi: clamp((raw−neutral)/(active−neutral), 0, 1).

CATATAN AU25/AU26: ARKit 52 blendshape tidak punya "mouthOpen" — sebelumnya
g("mouthOpen") selalu 0. Diperbaiki ke jawOpen sebagai proxy keduanya (Namba 2024).

Referensi:
  Craig et al. (2008). Affect and learning: An exploratory look into the role of
    affect in learning with AutoTutor. J Educ Media, 33(3).
  Turrisi et al. (2026). Blendshape features meet action units: a clinical mapping
    for enhancing facial expression analysis. Computers in Human Behavior Reports, 22.
  Aldenhoven et al. (2026). Real-time emotion recognition performance of mobile devices.
    Sensors, 26, 1060.
  Grafsgaard et al. (2013). Automatically recognizing facial expression:
    Predicting engagement and frustration. IEDM.
  Namba et al. (2024). Components of the thinking face. Acta Psychologica, 244.
"""

# Nama sinyal → deskripsi FACS (untuk viz / debug)
# Label tetap pakai AU (AU1, AU4, …) karena Turrisi 2026 memvalidasi korespondensnya.
AU_NAMES = {
    "AU1":  "Inner Brow Raiser",
    "AU2":  "Outer Brow Raiser",
    "AU4":  "Brow Lowerer",
    "AU7":  "Lid Tightener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU25": "Lips Part",   # Namba 2024: thinking face Component 2
    "AU26": "Jaw Drop",    # Namba 2024: thinking face Component 2
    "AU43": "Eyes Closed",  # eyeBlink: Turrisi 2026 "codify both AU43 and AU45"
}

# Anchor kalibrasi default (neutral, active) — empiris dari 21.204 frame raw_cache.
# active ≈ p90–p99 agar hanya aktivasi nyata yang → 1.0.
DEFAULT_AU_CALIB = {
    "AU1_neutral": 0.46, "AU1_active": 0.84,   # browInnerUp
    "AU2_neutral": 0.47, "AU2_active": 0.82,   # browOuterUp
    "AU4_neutral": 0.001, "AU4_active": 0.05,  # browDown (stretch agresif, median=0.001 p99=0.18)
    "AU7_neutral": 0.30, "AU7_active": 0.52,   # eyeSquint (setelah eyeLookDown gating)
    "AU12_neutral": 0.05, "AU12_active": 0.55, # mouthSmile
    "AU14_neutral": 0.002, "AU14_active": 0.08,# mouthDimple
    "AU25_neutral": 0.05, "AU25_active": 0.50, # jawOpen proxy AU25
    "AU26_neutral": 0.05, "AU26_active": 0.60, # jawOpen proxy AU26
    "AU43_neutral": 0.12, "AU43_active": 0.55, # eyeBlink (setelah eyeLookDown gating)
    # Threshold gating eyeLookDown (Turrisi 2026: AU64 = gaze direction ≠ eyelid)
    "eyeLookDown_gate_start": 0.25,  # eyeLookDown > nilai ini → mulai suppress AU7+AU43
    "eyeLookDown_gate_full":  0.60,  # eyeLookDown > nilai ini → full suppress
}


def _raw_blendshape_signals(blendshapes: dict, calib: dict = None) -> dict:
    """
    Blendshape mentah → nilai sinyal raw (pemetaan nama + gating, belum dinormalisasi).

    Menerapkan eyeLookDown gating (Turrisi 2026) untuk AU7 dan AU43:
    ketika eyeLookDown tinggi, aktivasi eyeSquint/eyeBlink adalah artifak
    arah pandang (AU64), bukan lid tightening (AU7) atau eye closure (AU43).
    """
    if calib is None:
        calib = DEFAULT_AU_CALIB
    g = lambda k: blendshapes.get(k, 0.0)

    # ── eyeLookDown gating (Turrisi 2026: AU64 gaze ≠ AU7/AU43 eyelid) ──────
    eye_look_down = (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2
    _gate_start = calib.get("eyeLookDown_gate_start", 0.25)
    _gate_full  = calib.get("eyeLookDown_gate_full", 0.60)
    _gate_range = max(_gate_full - _gate_start, 1e-6)
    # Linear ramp: 0 suppress saat eyeLookDown=gate_start, 1.0 suppress saat =gate_full
    _ld_suppress = max(0.0, min(1.0, (eye_look_down - _gate_start) / _gate_range))
    _ld_gate = 1.0 - _ld_suppress  # 1.0 = tidak ada suppress, 0.0 = full suppress

    # ── AU4: browDown (Brow Lowerer) ──────────────────────────────────────────
    # Aldenhoven 2026 Table 1: browDownLeft/Right → AU4 (Anger, Disgust, Fear).
    # Turrisi 2026: brow region κ=1.00 — unanimously confirmed.
    # CATATAN: noseSneer → AU9 (Nose Wrinkler) per Aldenhoven Table 1 & Turrisi;
    # bukan AU4 booster — dihapus dari pipeline agar sesuai paper.
    au4_raw = (g("browDownLeft") + g("browDownRight")) / 2

    return {
        "AU1":  g("browInnerUp"),
        "AU2":  (g("browOuterUpLeft") + g("browOuterUpRight")) / 2,
        "AU4":  au4_raw,
        # AU7 di-gate saat eyeLookDown tinggi (Turrisi 2026: AU64 ≠ AU7)
        "AU7":  ((g("eyeSquintLeft") + g("eyeSquintRight")) / 2) * _ld_gate,
        "AU12": max(g("mouthSmileLeft"), g("mouthSmileRight")),
        "AU14": (g("mouthDimpleLeft") + g("mouthDimpleRight")) / 2,
        # jawOpen sebagai proxy AU25+AU26 (ARKit tidak punya "mouthOpen" terpisah)
        "AU25": g("jawOpen"),
        "AU26": g("jawOpen"),
        # AU43 di-gate saat eyeLookDown tinggi (Turrisi 2026: AU64 ≠ AU43)
        "AU43": ((g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2) * _ld_gate,
        # Simpan nilai gating untuk debug
        "_eye_look_down": eye_look_down,
        "_ld_gate": _ld_gate,
    }


# Alias lama untuk kompatibilitas backward (dipanggil dari landmark_analyzer + app.py)
def _raw_action_units(blendshapes: dict) -> dict:
    """Alias backward-compat ke _raw_blendshape_signals (tanpa gating calib kustom)."""
    raw = _raw_blendshape_signals(blendshapes)
    # Hapus debug keys sebelum dikembalikan (interface lama tidak punya)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def compute_blendshape_features(blendshapes: dict, cfg: dict = None,
                                person_neutral: dict = None) -> dict:
    """
    Hitung intensitas sinyal emosi 0.0–1.0 dari blendshape MediaPipe FaceLandmarker,
    dengan normalisasi baseline-relative dan eyeLookDown gating.

    Chain paper:
      Craig 2008 (AU→emosi belajar) + Turrisi 2026 (BF→AU, κ=0.92) →
      blendshape sebagai sinyal langsung untuk Boredom/Confusion/Frustration/Engagement.

    Mendukung kalibrasi per-orang (Bosch 2023): jika person_neutral disediakan,
    neutral anchor di-override dengan nilai netral pribadi yang direkam.

    Args:
        blendshapes:    dict {nama_blendshape: skor 0–1} dari analyze_frame().
        cfg:            DEFAULT_RULES. Anchor dibaca dari cfg["action_units"].
        person_neutral: dict {AU_key: nilai_netral} dari person_neutrals.json.
                        Jika None → pakai baseline populasi DEFAULT_AU_CALIB.

    Returns:
        dict intensitas 0–1, mis. {"AU1": 0.0, "AU7": 0.4, "AU43": 0.0, ...}.
        Kunci "_raw" berisi nilai sebelum normalisasi (untuk debug).
        Kunci "_eye_look_down" dan "_ld_gate" untuk debug gating.
    """
    calib = DEFAULT_AU_CALIB.copy()
    if cfg is not None and "action_units" in cfg:
        calib.update(cfg["action_units"])

    raw_full = _raw_blendshape_signals(blendshapes, calib)
    # Pisahkan debug keys dari sinyal AU
    debug_keys = {k: v for k, v in raw_full.items() if k.startswith("_")}
    raw = {k: v for k, v in raw_full.items() if not k.startswith("_")}

    out = {}
    for au, val in raw.items():
        neutral = calib.get(f"{au}_neutral", 0.0)
        active  = calib.get(f"{au}_active", 1.0)
        # Per-person calibration (Bosch 2023 + FACS: intensitas = deviasi dari netral pribadi).
        if person_neutral and au in person_neutral:
            _pn = person_neutral[au]
            if 0 <= _pn < 1.0 - 1e-3:  # valid MediaPipe blendshape range
                _pop_range = max(active - neutral, 1e-6)
                neutral = _pn
                if _pn >= active:
                    # Person netral di atas population active (alis structural tinggi, dsb.).
                    # Geser active ke atas dengan range yang sama agar at-rest = 0 intensity,
                    # bukan blokir dan fall back ke populasi (yang menyebabkan always-1.0).
                    active = min(1.0, _pn + _pop_range)
        denom  = max(active - neutral, 1e-6)
        out[au] = max(0.0, min(1.0, (val - neutral) / denom))

    out["_raw"] = raw
    out.update(debug_keys)  # _eye_look_down, _ld_gate untuk debug
    return out


# ── Backward-compatibility alias ─────────────────────────────────────────────
# Semua caller lama (landmark_analyzer.py, app.py, recalculate.py) pakai
# compute_action_units — alias ini memastikan tidak ada yang perlu diubah.
compute_action_units = compute_blendshape_features
