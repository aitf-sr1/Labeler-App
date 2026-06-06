"""
core/action_units.py

Konversi blendshape MediaPipe FaceLandmarker → intensitas Action Unit FACS.

LATAR BELAKANG (kenapa modul ini ada)
-------------------------------------
Paper rujukan (Craig et al. 2008; Grafsgaard et al. 2011/2013) mengukur emosi
belajar lewat INTENSITAS Action Unit FACS yang dikode manusia bersertifikat.
MediaPipe meng-output 52 blendshape gaya ARKit — itu BUKAN AU FACS. Ada dua
perbedaan yang membuat blendshape mentah TIDAK boleh dipakai langsung sebagai AU:

1) Korespondensi nama (ARKit ↔ FACS), pemetaan many-to-one:
     AU1  Inner Brow Raiser  ← browInnerUp
     AU2  Outer Brow Raiser  ← mean(browOuterUpLeft, browOuterUpRight)
     AU4  Brow Lowerer       ← mean(browDownLeft, browDownRight)
                               + max(noseSneerLeft, noseSneerRight)*0.3  [booster lemah]
     AU7  Lid Tightener      ← mean(eyeSquintLeft, eyeSquintRight)
     AU12 Lip Corner Puller  ← max(mouthSmileLeft, mouthSmileRight)
     AU14 Dimpler            ← mean(mouthDimpleLeft, mouthDimpleRight)
     AU25 Lips Part          ← mouthOpen   (mulut terbuka sedikit)
     AU26 Jaw Drop           ← jawOpen     (rahang turun)
     AU43 Eyes Closed        ← mean(eyeBlinkLeft, eyeBlinkRight)

2) Skala blendshape ≠ intensitas FACS. Intensitas FACS = seberapa jauh otot
   bergerak DARI NETRAL. Pada 21.204 frame nyata dataset ini, blendshape "diam"
   ternyata TIDAK bernilai 0:
       browInnerUp  median 0.46   → AU1 seolah selalu aktif  (frustration over-fire)
       browOuterUp  median 0.47   → AU2 seolah selalu aktif
       browDown     median 0.001  → AU4 nyaris mati, rentang sempit (p99=0.18)
       eyeSquint    median 0.31   → AU7 baseline tinggi
       eyeBlink     median 0.11
   Memakai nilai mentah membuat AU1/AU2 (frustration) over-fire dan AU4
   (confusion) tidak pernah terdeteksi — persis bias yang teramati di data.

SOLUSI: normalisasi baseline-relative.
    intensity = clamp( (raw - neutral) / (active - neutral), 0, 1 )
  neutral = nilai blendshape saat wajah netral (≈ median populasi)
  active  = nilai saat AU benar-benar aktif (≈ p90–p99 / anchor FACS level-E)
Hasilnya intensitas 0–1 yang setara makna dengan intensitas AU FACS di paper:
0 = otot diam, 1 = AU aktif penuh.

CATATAN AU4 (browDown): MediaPipe browDown punya range SANGAT sempit (median
0.001, p99 hanya 0.18). Oleh karena itu kalibrasi AU4_active diset 0.05
(stretch agresif) agar deviasi kecil tetap terdeteksi. Selain itu, noseSneer
(sneer) ditambahkan sebagai sinyal co-occur lemah (×0.3) karena sneer sering
muncul bersamaan dengan brow lowering pada confusion/frustration, memperkuat
sinyal AU4 secara implisit.

CATATAN AU25/AU26 (mulut terbuka): Namba et al. (2024) menyebut opening the
mouth (AU25+AU26) sebagai "most significant component" thinking face. MediaPipe
menyediakan mouthOpen (≈AU25) dan jawOpen (≈AU26) sebagai blendshape terpisah
yang dapat dipakai langsung — sebelumnya hanya tersedia via py-feat.

Angka anchor adalah KALIBRASI EMPIRIS dari distribusi dataset ini (paper tidak
memberi angka untuk MediaPipe) dan dapat diatur lewat cfg["action_units"] /
Rules panel.
"""

# Nama AU → deskripsi FACS (untuk viz / debug)
AU_NAMES = {
    "AU1":  "Inner Brow Raiser",
    "AU2":  "Outer Brow Raiser",
    "AU4":  "Brow Lowerer",
    "AU7":  "Lid Tightener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU25": "Lips Part",   # mulut terbuka sedikit — Namba 2024: thinking face Component 2
    "AU26": "Jaw Drop",    # rahang turun/mulut terbuka — Namba 2024: thinking face Component 2
    "AU43": "Eyes Closed",
}

# Anchor default (neutral, active) — kalibrasi dari 21.204 frame raw_cache.
# active dipilih di sekitar p90–p99 tiap AU agar hanya aktivasi nyata yang → 1.0.
DEFAULT_AU_CALIB = {
    "AU1_neutral": 0.46, "AU1_active": 0.84,   # browInnerUp (median 0.46, p90 0.89) — active↓ utk sensitivitas alis-naik
    "AU2_neutral": 0.47, "AU2_active": 0.82,   # browOuterUp (median 0.47, p90 0.87) — active↓ utk sensitivitas alis-naik
    # browDown: median 0.001, p99 0.18 — stretch AGRESIF agar AU4 terdeteksi
    # noseSneer co-occur lemah (+0.3) ditambahkan di _raw_action_units()
    "AU4_neutral": 0.001, "AU4_active": 0.05,  # browDown + noseSneer*0.3 (stretch agresif)
    "AU7_neutral": 0.30, "AU7_active": 0.52,   # eyeSquint (median 0.31, p90 0.50)
    "AU12_neutral": 0.05, "AU12_active": 0.55, # mouthSmile (median 0.0, p99 0.75)
    "AU14_neutral": 0.002, "AU14_active": 0.08, # mouthDimple — kalibrasi sensitif (median 0.002)
    "AU25_neutral": 0.05, "AU25_active": 0.50, # mouthOpen  (0=tutup, 0.5+=terbuka jelas)
    "AU26_neutral": 0.05, "AU26_active": 0.60, # jawOpen    (0=tutup, 0.6+=rahang turun jelas)
    "AU43_neutral": 0.12, "AU43_active": 0.55, # eyeBlink   (median 0.11, p90 0.53)
}


def _raw_action_units(blendshapes: dict) -> dict:
    """Blendshape mentah → nilai AU mentah (pemetaan nama saja, belum dinormalisasi)."""
    g = lambda k: blendshapes.get(k, 0.0)
    # AU4: browDown primer + noseSneer sebagai booster co-occur lemah.
    # noseSneer sering co-occur dengan brow lowering saat confusion/frustration;
    # faktor 0.3 memastikan sneer tidak mendominasi — hanya memperkuat sinyal.
    brow_down_raw  = (g("browDownLeft") + g("browDownRight")) / 2
    nose_sneer_raw = (g("noseSneerLeft") + g("noseSneerRight")) / 2
    au4_raw = max(brow_down_raw, brow_down_raw + nose_sneer_raw * 0.3)
    return {
        "AU1":  g("browInnerUp"),
        "AU2":  (g("browOuterUpLeft") + g("browOuterUpRight")) / 2,
        "AU4":  au4_raw,
        "AU7":  (g("eyeSquintLeft")   + g("eyeSquintRight"))   / 2,
        "AU12": max(g("mouthSmileLeft"), g("mouthSmileRight")),
        "AU14": (g("mouthDimpleLeft") + g("mouthDimpleRight")) / 2,
        # PERBAIKAN: blendshape ARKit/MediaPipe TIDAK punya "mouthOpen" (52 nama resmi: lihat
        # MP_BLENDSHAPE_NAMES) — key lama g("mouthOpen") selalu 0 → AU25 mati & mouth-cue setengah jalan.
        # Tidak ada "lips part" terpisah di ARKit; jawOpen = sinyal mouth-open yang benar → dipakai
        # sebagai proxy AU25 (lips part) DAN AU26 (jaw drop). Namba 2024: keduanya = mulut terbuka (thinking face).
        "AU25": g("jawOpen"),     # Lips Part (proxy via jawOpen — ARKit tak punya lips-part terpisah)
        "AU26": g("jawOpen"),     # Jaw Drop  — Namba 2024 thinking face Component 2
        "AU43": (g("eyeBlinkLeft")    + g("eyeBlinkRight"))    / 2,
    }


def compute_action_units(blendshapes: dict, cfg: dict = None,
                         person_neutral: dict = None) -> dict:
    """
    Hitung intensitas AU FACS 0.0–1.0 dari blendshape MediaPipe FaceLandmarker,
    dengan normalisasi baseline-relative.

    Mendukung kalibrasi per-orang (Bosch 2023 + prinsip FACS: intensitas AU =
    seberapa jauh otot bergerak dari NETRAL PRIBADI orang itu, bukan dari median
    populasi). Jika person_neutral disediakan, neutral anchor untuk tiap AU
    di-override dengan nilai netral pribadi yang sudah direkam.

    Pipeline ini adalah satu-satunya sumber AU — tidak ada py-feat, tidak ada
    subprocess eksternal. Semua AU dihitung secara sinkron dalam satu proses.

    Args:
        blendshapes:    dict {nama_blendshape: skor 0–1} dari analyze_frame().
        cfg:            DEFAULT_RULES (atau rules custom). Anchor dibaca dari
                        cfg["action_units"]; fallback ke DEFAULT_AU_CALIB.
        person_neutral: dict {AU_key: nilai_netral} dari MediaPipe AU orang ini
                        saat wajah netral, mis. {"AU1": 0.00, "AU4": 0.05, ...}.
                        Jika None → pakai baseline populasi dari DEFAULT_AU_CALIB.
                        Disimpan via utils/person_neutral.py (Bosch 2023 §3.2).

    Returns:
        dict intensitas AU ter-normalisasi, mis. {"AU1": 0.0, "AU4": 0.7, ...}.
        Kunci "_raw" berisi nilai AU mentah (sebelum normalisasi) untuk viz/debug.

    AU yang tersedia:
        AU1  Inner Brow Raiser  (Frustration primer — Craig 2008)
        AU2  Outer Brow Raiser  (Frustration primer — Craig 2008)
        AU4  Brow Lowerer       (Confusion primer — Craig 2008, Grafsgaard 2011)
        AU7  Lid Tightener      (Confusion primer — Craig 2008)
        AU12 Lip Corner Puller  (gate Confusion/Frustration — Craig 2008)
        AU14 Dimpler            (Frustration sekunder — Grafsgaard 2013)
        AU25 Lips Part          (Confusion cue lemah — Namba 2024: thinking face)
        AU26 Jaw Drop           (Confusion cue lemah — Namba 2024: thinking face)
        AU43 Eyes Closed        (Boredom primer — Craig 2008)
    """
    calib = DEFAULT_AU_CALIB
    if cfg is not None and "action_units" in cfg:
        # Merge: rules custom boleh override sebagian anchor saja
        calib = {**DEFAULT_AU_CALIB, **cfg["action_units"]}

    raw = _raw_action_units(blendshapes)
    out = {}
    for au, val in raw.items():
        neutral = calib.get(f"{au}_neutral", 0.0)
        # Per-person calibration (Bosch 2023 + FACS: intensitas = deviasi dari NETRAL PRIBADI).
        # Jika person_neutral tersedia, override neutral anchor dengan nilai netral orang ini
        # (yang direkam saat Tandai Frame Netral). Active anchor tetap dari populasi/cfg
        # karena sulit merekam "ekspresi aktif penuh" tiap AU per orang.
        # Kunci di person_neutral menggunakan format MediaPipe AU (AU1, AU2, dst).
        if person_neutral and au in person_neutral:
            neutral = person_neutral[au]
        active = calib.get(f"{au}_active", 1.0)
        denom  = max(active - neutral, 1e-6)
        out[au] = max(0.0, min(1.0, (val - neutral) / denom))
    out["_raw"] = raw
    return out
