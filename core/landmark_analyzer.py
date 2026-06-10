"""
core/landmark_analyzer.py

Analisis wajah 3D menggunakan MediaPipe FaceLandmarker + HandLandmarker.

Menghasilkan per frame:
  - Head pose    : yaw (kiri/kanan), pitch (atas/bawah), roll (miring) dalam derajat
  - Iris offset  : (x, y) ternormalisasi −1.0 s/d +1.0 relatif terhadap ukuran mata
  - Blendshapes  : dict nama → skor 0.0–1.0 (52 blendshape ARKit)
  - Hand signals : hand_one (1 tangan), hand_two (≥2 tangan) untuk cue tangan
  - Skor emosi landmark (0–1) untuk hybrid scoring dengan SigLIP

ARSITEKTUR: MediaPipe-only — tidak ada py-feat, tidak ada subprocess eksternal.
Sinyal emosi dihitung dari blendshape MediaPipe via core/blendshape_features.py (sinkron).
Chain paper: Craig 2008 (AU→emosi) + Turrisi 2026 (BF→AU, κ=0.92) → blendshape langsung.
Model diunduh otomatis ke ~/.cache/siglip_labeler/ pada pertama kali digunakan.
"""

import os
import math
import urllib.request
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .blendshape_features import compute_action_units, AU_NAMES

# Sumber blendshape (env BLENDSHAPE_SOURCE):
#   "mediapipe"      (default) = blendshape bawaan MediaPipe FaceLandmarker (gratis, sudah keluar)
#   "mp_blendshapes"           = model py-feat mp_blendshapes (MLP-Mixer mesh→52 blendshape, standalone)
# Keduanya menghasilkan 52 blendshape ARKit yang sama-sama "longgar" terhadap FACS;
# toggle ini untuk eksperimen A/B (lihat core/mp_blendshapes.py + DESIGN_RATIONALE §16).
_BLENDSHAPE_SOURCE = os.getenv("BLENDSHAPE_SOURCE", "mediapipe").strip().lower()

# ── Model URLs & cache ──────────────────────────────────────────────────────
_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_NAME = "face_landmarker.task"

_HAND_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_NAME = "hand_landmarker.task"

# ── Landmark indices ─────────────────────────────────────────────────────────
# Left eye
_L_OUTER, _L_INNER = 33,  133
_L_TOP,   _L_BOT   = 159, 145
_L_IRIS             = 468
# Right eye
_R_OUTER, _R_INNER = 263, 362
_R_TOP,   _R_BOT   = 386, 374
_R_IRIS             = 473


# ── Data class ───────────────────────────────────────────────────────────────
@dataclass
class LandmarkResult:
    yaw:            float = 0.0    # + = kepala ke kanan
    pitch:          float = 0.0    # + = kepala tengadah ke atas
    roll:           float = 0.0    # + = kepala miring ke kanan (ear→shoulder), confusion signal
    iris_x:         float = 0.0    # -1=pupil kiri, 0=center, +1=kanan  (relatif ke sudut mata)
    iris_y:         float = 0.0    # -1=pupil atas, 0=center, +1=bawah  (relatif ke sudut mata)
    # Posisi iris relatif ke PUSAT CROP (image-center-relative gaze)
    # Digunakan untuk Engagement/Boredom: apakah orang menatap ke arah kamera?
    iris_img_x:     float = 0.0    # 0=pusat frame, +=kanan frame, -=kiri frame
    iris_img_y:     float = 0.0    # 0=pusat frame, +=bawah frame, -=atas frame
    left_iris_px:   tuple = (0, 0) # pixel (x,y) di frame
    right_iris_px:  tuple = (0, 0)
    blendshapes:    dict  = field(default_factory=dict)
    face_found:     bool  = False
    face_landmarks: list  = field(default_factory=list)  # raw mediapipe landmarks
    # Sinyal tangan
    hand_one:  float = 0.0  # 1 tangan di wajah — cue Confusion via max(hand_one,hand_two)*hand_conf_w
    #                          (Mahmoud 2011: 14/15 segmen = thinking/unsure ≈93%; Behera 2020: HoF↑difficulty)
    hand_two:  float = 0.0  # 2 tangan di wajah — cue Confusion (HoF) + cue Frustration
    #                          (Grafsgaard 2013b: two-hands ↔ self-efficacy rendah, signifikan)
    hand_landmarks_px: list = field(default_factory=list)  # pixel positions untuk viz
    hand_landmarks_raw: list = field(default_factory=list) # raw mediapipe hand landmarks
    # Per-person neutral calibration (Bosch 2023 + prinsip FACS: intensitas AU =
    # seberapa jauh otot bergerak dari NETRAL PRIBADI). dict {AU_key: nilai_netral}
    # dalam format MediaPipe AU (AU1, AU2, AU4, ...). Diisi dari person_neutrals.json
    # via utils/person_neutral.py. Jika None → pakai baseline populasi (DEFAULT_AU_CALIB).
    person_neutral: dict = None


# ── Per-thread MediaPipe instances (thread-local storage) ────────────────────
# MediaPipe landmarker TIDAK thread-safe — instance yang sama tidak bisa dipakai
# concurrently oleh beberapa thread. Solusi: satu instance per thread via threading.local().
# Setiap thread CPU worker mendapat instance sendiri → benar-benar paralel.
_thread_local = threading.local()

def _ensure_model_files():
    """Download model files jika belum ada. Dipanggil sekali saat pertama kali dibutuhkan."""
    from mediapipe.tasks import python as mp_tasks  # noqa: F401
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "siglip_labeler")
    os.makedirs(cache_dir, exist_ok=True)

    face_path = os.path.join(cache_dir, _LANDMARKER_NAME)
    if not os.path.exists(face_path):
        print(f"Mengunduh FaceLandmarker model ke {face_path}…")
        urllib.request.urlretrieve(_LANDMARKER_URL, face_path)

    hand_path = os.path.join(cache_dir, _HAND_NAME)
    if not os.path.exists(hand_path):
        print(f"Mengunduh HandLandmarker model ke {hand_path}…")
        urllib.request.urlretrieve(_HAND_URL, hand_path)

    return face_path, hand_path

# Lock hanya untuk serialisasi download pertama kali
_download_lock = threading.Lock()
_models_downloaded = False

def _get_landmarker():
    """Kembalikan FaceLandmarker milik thread ini (buat jika belum ada)."""
    global _models_downloaded
    if not hasattr(_thread_local, "face_landmarker"):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        with _download_lock:
            face_path, _ = _ensure_model_files()
            _models_downloaded = True
        base_opts = mp_tasks.BaseOptions(model_asset_path=face_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.40,
            min_face_presence_confidence=0.40,
        )
        _thread_local.face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _thread_local.face_landmarker


def _get_hand_landmarker():
    """Kembalikan HandLandmarker milik thread ini (buat jika belum ada)."""
    if not hasattr(_thread_local, "hand_landmarker"):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        with _download_lock:
            _, hand_path = _ensure_model_files()
        base_opts = mp_tasks.BaseOptions(model_asset_path=hand_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.20,
            min_hand_presence_confidence=0.20,
        )
        _thread_local.hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
    return _thread_local.hand_landmarker


# ── Helpers ──────────────────────────────────────────────────────────────────
def _rotation_matrix_to_euler(matrix):
    """Ekstrak yaw, pitch, roll (derajat) dari 4×4 transformation matrix."""
    R  = np.array(matrix)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
        roll  = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
        roll  = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _iris_offset(lms, iris_idx, inner_idx, outer_idx, top_idx, bot_idx, w, h):
    """Hitung offset iris dari center mata (−1..+1)."""
    iris  = lms[iris_idx]
    inner = lms[inner_idx];  outer = lms[outer_idx]
    top   = lms[top_idx];    bot   = lms[bot_idx]

    ix = iris.x * w;  iy = iris.y * h
    cx = ((inner.x + outer.x) / 2) * w
    cy = ((top.y  + bot.y)  / 2) * h
    hw = abs(inner.x - outer.x) * w / 2 + 1e-6
    hh = abs(top.y   - bot.y)   * h / 2 + 1e-6
    return (ix - cx) / hw, (iy - cy) / hh, int(ix), int(iy)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Main analysis ─────────────────────────────────────────────────────────────
def _analyze_hands(mp_image, h: int, w: int):
    """
    Deteksi tangan berdasarkan jumlah tangan di wajah (Grafsgaard 2013b).

    Returns:
        (hand_one, hand_two, dummy, hand_pts_px, raw)
        hand_one: 1.0 jika ada tepat 1 tangan (Thoughtful/Confusion)
        hand_two: 1.0 jika ada >= 2 tangan (Struggle/Frustration)
        hand_pts_px: list[(x,y)] semua titik tangan untuk viz
    """
    try:
        res = _get_hand_landmarker().detect(mp_image)
    except Exception as e:
        if _DBG_LAND: print(f"  [HAND] Exception saat detect: {e}")
        return 0.0, 0.0, 0.0, [], []

    if not res.hand_landmarks:
        if _DBG_LAND: print(f"  [HAND] Tidak ada tangan terdeteksi")
        return 0.0, 0.0, 0.0, [], []

    # Kumpulkan semua titik landmark dari semua tangan yang terdeteksi
    all_pts = []
    for hand_lms in res.hand_landmarks:
        for lm in hand_lms:
            all_pts.append((lm.x, lm.y))

    if not all_pts:
        return 0.0, 0.0, 0.0, [], []

    if _DBG_LAND: print(f"  [HAND] Terdeteksi {len(res.hand_landmarks)} tangan, total {len(all_pts)} titik")

    # ZONASI TANGAN V3
    centered = sum(1 for x, _ in all_pts if 0.05 <= x <= 0.95)
    if _DBG_LAND: print(f"  [HAND] centered={centered}")

    if centered < 5:
        if _DBG_LAND: print(f"  [HAND] DIFILTER: centered({centered}) < 5, skor di-reset 0")
        hand_one = hand_two = 0.0
    else:
        n_hands = len(res.hand_landmarks)
        hand_one = 1.0 if n_hands == 1 else 0.0
        hand_two = 1.0 if n_hands >= 2 else 0.0
    
    hand_pts_px = [(int(x * w), int(y * h)) for x, y in all_pts]

    return hand_one, hand_two, 0.0, hand_pts_px, res.hand_landmarks


def detect_hands_from_full_frame(
    full_frame_bgr: np.ndarray,
    face_bbox: tuple,
    crop_size: int = 512
) -> tuple:
    """
    Deteksi tangan dari frame FULL (sebelum crop wajah), lalu remap koordinatnya
    ke ruang gambar crop wajah 512x512.

    Args:
        full_frame_bgr : frame BGR asli dari video (belum di-crop)
        face_bbox      : (x1, y1, x2, y2) bounding box wajah di frame asli
        crop_size      : ukuran output crop (default 512)

    Returns:
        (hand_one, hand_two, hand_pts_px_in_crop, hand_raw_lms)
    """
    import mediapipe as mp

    fh, fw = full_frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(full_frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        res = _get_hand_landmarker().detect(mp_img)
    except Exception as e:
        if _DBG_LAND: print(f"  [HAND-FULL] Exception: {e}")
        return 0.0, 0.0, [], []

    if not res.hand_landmarks:
        if _DBG_LAND: print(f"  [HAND-FULL] Tidak ada tangan di full frame")
        return 0.0, 0.0, [], []

    x1_face, y1_face, x2_face, y2_face = face_bbox
    face_w = max(x2_face - x1_face, 1)
    face_h = max(y2_face - y1_face, 1)

    # Remap semua landmark dari koordinat full frame ke koordinat crop wajah
    all_pts_remapped = []
    for hand_lms in res.hand_landmarks:
        for lm in hand_lms:
            # Koordinat pixel di full frame
            px_full = lm.x * fw
            py_full = lm.y * fh
            # Konversi ke koordinat relatif dalam crop wajah (0.0 - 1.0)
            px_crop = (px_full - x1_face) / face_w
            py_crop = (py_full - y1_face) / face_h
            all_pts_remapped.append((px_crop, py_crop))

    if not all_pts_remapped:
        return 0.0, 0.0, [], []

    in_crop = sum(1 for x, y in all_pts_remapped if -0.2 <= x <= 1.2 and -0.2 <= y <= 1.2)

    if _DBG_LAND:
        print(f"  [HAND-FULL] Terdeteksi {len(res.hand_landmarks)} tangan, in_crop={in_crop}")

    if in_crop < 3:
        return 0.0, 0.0, [], []

    n_hands = len(res.hand_landmarks)
    hand_one = 1.0 if n_hands == 1 else 0.0
    hand_two = 1.0 if n_hands >= 2 else 0.0

    hand_pts_px = [(int(x * crop_size), int(y * crop_size)) for x, y in all_pts_remapped]
    return hand_one, hand_two, hand_pts_px, res.hand_landmarks


def analyze_frame(frame_bgr, injected_hand: tuple = None) -> 'LandmarkResult':
    """
    Jalankan FaceLandmarker pada satu frame BGR (sebaiknya 512×512 crop wajah).

    Returns:
        LandmarkResult — berisi head pose, iris offset, blendshapes.
    """
    import mediapipe as mp

    h, w  = frame_bgr.shape[:2]
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res   = _get_landmarker().detect(mp_img)

    # Gunakan injected_hand dari full-frame jika tersedia, fallback ke deteksi crop
    if injected_hand is not None:
        hand_one, hand_two, hand_pts_px, hand_raw = injected_hand
    else:
        hand_one, hand_two, _, hand_pts_px, hand_raw = _analyze_hands(mp_img, h, w)

    if not res.face_landmarks:
        return LandmarkResult(
            face_found=False,
            hand_one=hand_one,
            hand_two=hand_two,
            hand_landmarks_px=hand_pts_px,
            hand_landmarks_raw=hand_raw,
        )

    lms = res.face_landmarks[0]
    bs  = {b.category_name: round(b.score, 4) for b in res.face_blendshapes[0]}
    # Opsi: ganti sumber blendshape ke model py-feat mp_blendshapes (env BLENDSHAPE_SOURCE).
    # Output kompatibel (52 nama ARKit yang sama) → langsung dipakai oleh blendshape_features.py.
    if _BLENDSHAPE_SOURCE == "mp_blendshapes":
        from .mp_blendshapes import compute_blendshapes as _mp_bs
        _alt = _mp_bs(lms)
        if _alt:
            bs = {k: round(v, 4) for k, v in _alt.items()}
    yaw, pitch, roll = _rotation_matrix_to_euler(res.facial_transformation_matrixes[0])

    lx, ly, lxi, lyi = _iris_offset(lms, _L_IRIS, _L_INNER, _L_OUTER, _L_TOP, _L_BOT, w, h)
    rx, ry, rxi, ryi = _iris_offset(lms, _R_IRIS, _R_INNER, _R_OUTER, _R_TOP, _R_BOT, w, h)

    iris_x = float(np.clip((lx + rx) / 2, -1, 1))
    iris_y = float(np.clip((ly + ry) / 2, -1, 1))

    # ── Posisi iris relatif ke PUSAT FRAME CROP (image-center-relative) ──────
    # lms[_L_IRIS].x/y dan lms[_R_IRIS].x/y adalah koordinat 0-1 dalam image.
    # Dikurangi 0.5 → 0 = pusat frame, positif = kanan/bawah frame.
    # Ini mengukur ke mana orang memandang relatif ke kamera (bukan ke mata sendiri).
    l_cx = lms[_L_IRIS].x
    l_cy = lms[_L_IRIS].y
    r_cx = lms[_R_IRIS].x
    r_cy = lms[_R_IRIS].y
    iris_img_x = float(np.clip(((l_cx + r_cx) / 2) - 0.5, -0.5, 0.5))
    iris_img_y = float(np.clip(((l_cy + r_cy) / 2) - 0.5, -0.5, 0.5))

    return LandmarkResult(
        yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 2),
        iris_x=round(iris_x, 4), iris_y=round(iris_y, 4),
        iris_img_x=round(iris_img_x, 4), iris_img_y=round(iris_img_y, 4),
        left_iris_px=(lxi, lyi), right_iris_px=(rxi, ryi),
        blendshapes=bs, face_found=True,
        face_landmarks=lms,
        hand_one=round(hand_one, 4),
        hand_two=round(hand_two, 4),
        hand_landmarks_px=hand_pts_px,
        hand_landmarks_raw=hand_raw,
    )


# ── Emotion scoring ───────────────────────────────────────────────────────────
_DBG_LAND = True   # set False untuk matikan debug log (otomatis dimatikan saat batch)

def compute_emotion_scores(r: LandmarkResult, cfg: dict = None) -> dict:
    """
    Hitung skor landmark 0.0-1.0 per emosi.

    Args:
        r:   LandmarkResult dari analyze_frame().
        cfg: Dict parameter dari rules.py. Jika None, pakai DEFAULT_RULES.

    PRINSIP GAZE:
      Metrik gaze deviation yang DIPAKAI:
      - `gaze_dev_bore` (yaw+iris, ke atas saja, tanpa roll) → Boredom. Nunduk & roll ≠ bosan.
      - `gaze_dev_eng`  (sama dgn bore) → Engagement gate (line `gate`). Roll ditangani
        roll_gate terpisah agar natural head tilt tidak mengurangi engagement.
      Tatapan lurus ke kamera → gaze rendah → Tidak Bosan, Engaged.
      (Confusion TIDAK memakai gaze gate — Craig 2008 mendefinisikannya via AU, bukan arah pandang.)
    """
    from .rules import DEFAULT_RULES
    if cfg is None:
        cfg = DEFAULT_RULES

    if not r.face_found:
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    # Intensitas AU FACS untuk scoring — semua dari blendshape MediaPipe (baseline-normalized).
    # AU1/AU2/AU4/AU7/AU12/AU14/AU25/AU26/AU43 tersedia dari compute_blendshape_features()
    # via alias compute_action_units (chain: Craig 2008 + Turrisi 2026).
    # person_neutral (Bosch 2023): jika ada → override neutral anchor per-AU dengan
    # netral pribadi orang ini → scoring lebih adil lintas individu (Bartlett 1999 FACS).
    # Scoring TIDAK akses blendshape langsung — semua lewat au[...].
    au = compute_action_units(r.blendshapes, cfg,
                              person_neutral=getattr(r, "person_neutral", None))

    gcfg = cfg["gaze"]
    bcfg = cfg["boredom"]
    ecfg = cfg["engagement"]
    ccfg = cfg["confusion"]
    fcfg = cfg["frustration"]

    # ── Iris Y reliability gate ───────────────────────────────────────────────
    # Saat mata hampir menutup, iris_y MediaPipe jadi ekstrem (iris di tepi atas celah
    # mata yang nyaris menutup), bukan gaze sungguhan. Suppress iris_y proporsional
    # terhadap eye-closure (AU43) supaya artifact menutupnya mata tidak polusi gaze_dev.
    # AU43 dari au[] = py-feat (primer) / fallback blendshape — TIDAK akses blendshape langsung.
    _eye_closed      = au["AU43"]
    _iris_blink_th   = gcfg.get("iris_blink_suppress_th", 0.60)
    _iris_blink_zero = gcfg.get("iris_blink_zero_th", 0.0)  # 0 = disabled; > 0: zero iris_y jika AU43 >= nilai ini
    if _iris_blink_zero > 0 and _eye_closed >= _iris_blink_zero:
        # Mata sangat tertutup → iris_y tidak bisa dipercaya sama sekali
        _iris_y_factor = 0.0
    else:
        _iris_y_factor = max(0.0, 1.0 - _eye_closed / max(_iris_blink_th, 1e-6))
    # Catatan: iris_y (eye-space) TIDAK dipakai untuk gaze_v lagi (lihat penjelasan di bawah);
    # _iris_y_factor (blink suppression) tetap diterapkan ke iris_img_y di bawah.

    # ── Gaze deviation (shared basis) ─────────────────────────────────────────
    GAZE_SCALE   = gcfg["scale_h"]
    GAZE_SCALE_V = gcfg["scale_v"]
    gaze_h     = r.yaw + r.iris_x * GAZE_SCALE

    # Untuk gaze VERTIKAL: gunakan iris_img_y (posisi iris absolut di image frame),
    # BUKAN iris_y (relatif ke eye center). Alasan:
    #   iris_y = (iris_pixel_y - eye_center_y) / eye_height — dihitung dalam koordinat
    #   eye frame yang berputar dengan kepala. Saat head roll besar (>8°), eye landmarks
    #   (top=159/bot=145) ikut berputar, sehingga iris_y bisa artifak besar (misal -0.431)
    #   meski iris_img_y ≈ 0 (siswa menatap layar lurus). Ini menyebabkan boredom-dari-atas
    #   palsu saat kepala miring.
    #   iris_img_y = raw iris y position in image - 0.5 — tidak terpengaruh head roll,
    #   langsung merepresentasikan ke mana iris menghadap di ruang image/kamera.
    #   Untuk horizontal (gaze_h): tetap pakai iris_x (eye-space) agar kompensasi head yaw
    #   tetap bekerja (kepala noleh kanan + iris kompensasi kiri → gaze_h ≈ 0).
    _iris_img_y_eff = r.iris_img_y * _iris_y_factor
    gaze_v_raw = -r.pitch + _iris_img_y_eff * GAZE_SCALE_V

    # Pisahkan komponen vertikal atas dan bawah:
    #   - Ke atas : dead zone kecil (5°) — tatapan ke atas = melamun = boredom
    #   - Ke bawah: dead zone besar (15°) — nunduk = baca/ngetik/berpikir = BUKAN boredom
    # Arah vertikal murni dari geometri (pitch kepala + iris_img_y). BUKAN blendshape eyeLookDown.
    gaze_v_up   = max(0.0, -gaze_v_raw)                             # komponen ke atas
    gaze_v_down = max(0.0, gaze_v_raw)                              # komponen ke bawah (debug saja)
    v_dz_up   = gcfg.get("v_dead_zone_up", 5.0)
    gaze_v_up_eff   = max(0.0, gaze_v_up   - v_dz_up)

    # gaze_h sudah = yaw + iris_x*scale_h — menggabungkan head pose + posisi iris.
    # Jika iris mengkompensasi head turn (natap depan walau kepala miring), gaze_h ≈ 0 → benar.
    # iris_side dan abs(yaw) di max() MENGABAIKAN kompensasi ini dan menyebabkan false positive:
    #   kepala miring 12° + iris kompensasi → gaze_h≈0 tapi iris_side=24° → boredom salah trigger.
    # Solusi: gunakan abs(gaze_h) saja — kompensasi mata sudah terhitung.
    gaze_h_eff = abs(gaze_h)

    # Roll (kepala miring, ear→shoulder).
    roll_gaze_eff = max(0.0, abs(r.roll) - gcfg.get("roll_dz", 5.0))

    # gaze_dev_bore: untuk BOREDOM — HANYA dari arah pandang (gaze horizontal + ke atas).
    #   Nunduk (ke bawah) = baca soal / ngetik / mikir = Engagement/Confusion, BUKAN Boredom.
    #   Roll (kepala miring ke bahu) TIDAK ditambahkan: siswa dapat memiringkan kepala
    #   sambil tetap menatap layar — roll ≠ mengalihkan pandang. Hanya yaw (menoleh) = boredom.
    #   roll_gaze_eff tetap dihitung untuk debug log, tapi TIDAK masuk ke gaze_dev_bore.
    gaze_dev_bore = (gaze_h_eff ** 2 + gaze_v_up_eff ** 2) ** 0.5

    # gaze_dev_eng: untuk ENGAGEMENT gate — TANPA roll, TANPA komponen ke bawah.
    # Sümer et al. (2021): "head-down (i.e., taking notes or reading learning material)" —
    # lihat ke bawah = baca/catat = ON-TASK. "Students can still focus on content when
    # looking around or taking notes." → lihat ke bawah TIDAK mengurangi skor Engagement.
    gaze_dev_eng  = (gaze_h_eff ** 2 + gaze_v_up_eff ** 2) ** 0.5

    # == 0: BOREDOM ============================================================
    bore_gaze_raw = _clamp((gaze_dev_bore - bcfg["gaze_dead_zone"]) / max(bcfg["gaze_range"], 1e-6), 0, 1)

    # MUTLAK: hadap depan = gaze tidak menyimpang → komponen bore_gaze di-nol-kan.
    # AU43 (blink_direct) tetap aktif independen dari gaze gate (Craig 2008: primary signal).
    # Range kecil (4°) agar transisi cepat: di gaze_h=9° bore_gaze sudah hampir penuh.
    # Menggunakan jarak deviasi menjauh layar (termasuk dongak atas) agar zoning-out terdeteksi.
    bore_fwd_th    = bcfg.get("fwd_yaw_th", 5.0)    # turun dari 8° ke 5°
    bore_fwd_range = bcfg.get("fwd_yaw_range", 4.0)  # turun dari 8° ke 4°
    gaze_away_mag  = (gaze_h_eff ** 2 + gaze_v_up_eff ** 2) ** 0.5
    bore_fwd_gate  = _clamp((gaze_away_mag - bore_fwd_th) / max(bore_fwd_range, 1e-6), 0, 1)
    bore_gaze = bore_gaze_raw * bore_fwd_gate  # nol saat hadap depan, penuh di gaze_h≥9°

    # AU43 (eye closure) dari au[] = py-feat (primer) / fallback blendshape — ter-normalisasi 0–1.
    # Tidak lagi akses eyeBlink/eyeSquint blendshape langsung; koreksi-squint manual tak perlu
    # karena py-feat AU43 sudah memisahkan eye-closure dari lid-tightener (AU7).
    blink_corrected = au["AU43"]
    blink_v    = _clamp((blink_corrected - bcfg["blink_dead_zone"]) / max(bcfg["blink_range"], 1e-6), 0, 1)
    # expr_gate dari bore_gaze yang sudah di-gate → blink tidak fire saat natap layar
    expr_gate = _clamp(bore_gaze / max(bcfg.get("expr_gaze_gate_th", 0.35), 1e-6), 0, 1)

    # Craig et al. (2008): AU43 (eye closure) adalah satu-satunya sinyal yang divalidasi untuk Boredom.
    # Yawn (jawOpen) TIDAK divalidasi — tidak ada paper yang mengaitkan yawn dengan boredom di learning context.
    sig_expr       = blink_v * bcfg["sig_expr_weight"]
    sig_expr_gated = sig_expr * expr_gate

    # Craig et al. (2008): AU43 (eye closure) = primary boredom signal, independent of gaze deviation.
    # Students show droopy/heavy eyelids when bored even while still facing screen.
    blink_direct_th = bcfg.get("blink_direct_th", 0.45)
    blink_direct_w  = bcfg.get("blink_direct_w", 0.45)
    blink_direct = _clamp(
        (blink_corrected - blink_direct_th) / max(bcfg["blink_range"], 1e-6), 0, 1
    ) * blink_direct_w


    # Craig et al. (2008): AU43 = satu-satunya sinyal yang divalidasi untuk Boredom.
    # Tidak ada paper yang memvalidasi suppressor (eye_wide, squint, smile, browInner) untuk boredom.
    base_bore = max(bore_gaze, sig_expr_gated, blink_direct)
    bore      = _clamp(base_bore * bcfg["blend_a"] + (bore_gaze + sig_expr_gated) * bcfg["blend_b"], 0, 1)

    # == 1: ENGAGEMENT =========================================================
    gate = _clamp(1 - max(0, gaze_dev_eng - ecfg["tegak_dead_zone"]) / max(ecfg["tegak_range"], 1e-6), 0, 1)

    # Yaw gate: kepala miring ke samping (yaw besar) → engagement nol
    yaw_gate_th    = ecfg.get("yaw_gate_th", 20.0)
    yaw_gate_range = ecfg.get("yaw_gate_range", 10.0)
    yaw_gate = _clamp(1.0 - max(0.0, abs(r.yaw) - yaw_gate_th) / max(yaw_gate_range, 1e-6), 0.0, 1.0)

    # Roll gate: kepala miring kiri/kanan (roll besar) → engagement turun.
    # Dead zone 10° agar natural head tilt (roll 5-8°) tidak kena penalti.
    roll_gate_th    = ecfg.get("roll_gate_th", 10.0)   # mulai pengaruhi (°)
    roll_gate_range = ecfg.get("roll_gate_range", 5.0)  # nol di th+range (default: 15°)
    roll_gate = _clamp(1.0 - max(0.0, abs(r.roll) - roll_gate_th) / max(roll_gate_range, 1e-6), 0.0, 1.0)

    # Whitehill et al. (2014): level 1 "eyes completely closed", level 2 "eyes barely open" → anti-engagement
    # AU43 (eye closure, via au[] = py-feat primer/fallback blendshape) = primary anti-engagement cue.
    blink_heavy = max(0.0, blink_corrected - ecfg["blink_heavy_th"]) / max(ecfg["blink_heavy_th"], 1e-6)
    # Pitch gate: mendongak ke atas = tidak fokus ke layar (Whitehill: "looking away from computer")
    pitch_gate_th    = ecfg.get("pitch_gate_th", 15.0)
    pitch_gate_range = ecfg.get("pitch_gate_range", 15.0)
    pitch_gate = _clamp(1.0 - max(0.0, r.pitch - pitch_gate_th) / max(pitch_gate_range, 1e-6), 0.0, 1.0)
    eng = _clamp(gate * yaw_gate * roll_gate * pitch_gate * max(ecfg["blink_heavy_min"], 1.0 - blink_heavy), 0, 1)

    # D'Mello & Graesser (2012): Boredom dan Engagement near-mutually exclusive.
    # Gupta et al. (2016) DAiSEE "Complementary Labels": "When engagement is low,
    # boredom is generally high and vice-versa" — observasi langsung di dataset 4-emosi.
    bore_suppress_th  = ecfg.get("bore_suppress_th", 0.45)
    bore_suppress_v   = _clamp((bore - bore_suppress_th) / max(1.0 - bore_suppress_th, 1e-6), 0, 1)
    bore_eng_suppress = ecfg.get("bore_eng_suppress", 0.40)
    eng = _clamp(eng - bore_suppress_v * bore_eng_suppress, 0, 1)

    # == 2: CONFUSION ==========================================================
    # Craig et al. (2008) Table 2: AU4 (brow lowerer) 95%, AU7 (lid tightener) 78%,
    # AU4+AU7 co-occurrence 73%. AU12 (questioning smile) disebut di PROSA Craig sebagai
    # asosiasi SEKUNDER yang lebih lemah — bukan 95% (95% itu coverage AU4).
    # Grafsgaard et al. (2011): AU4 validated via HMM as primary confusion predictor.
    # AU1/AU2 (brow raise) BUKAN sinyal confusion — itu sinyal frustration.
    # iris/gaze, head pitch, look_dn TIDAK divalidasi paper manapun untuk confusion.
    # Intensitas AU baseline-normalized (au[...]): browDown MediaPipe yang nyaris mati
    # (median 0.001) di-stretch agar AU4 benar-benar dapat terdeteksi saat aktif.

    brow_dn_v    = au["AU4"]                        # AU4 brow lowerer (primary, Craig2008 95%)
    au7_v        = au["AU7"]                         # AU7 lid tightener (Craig2008 78%)
    au4_au7_co   = (brow_dn_v * au7_v) ** 0.5        # geometric mean — butuh KEDUANYA aktif bersamaan
    au4_au7_co_w = ccfg.get("au4_au7_co_w", 0.50)
    au4_au7_sig  = au4_au7_co * au4_au7_co_w

    # Craig et al. (2008) Table 2: AU7 (lid tightener) = 78% coverage sebagai sinyal STANDALONE,
    # bukan hanya co-occurrence. browDown (AU4) nyaris mati di MediaPipe (median 0.001) sementara
    # eyeSquint (AU7) terukur baik → AU7 standalone memberi confusion sinyal landmark yang nyata
    # dan TETAP paper-faithful. Bobotnya 78%/95% relatif ke AU4 (coverage lebih rendah).
    au7_alone_w = ccfg.get("au7_alone_w", 0.78)
    au7_sig     = au7_v * au7_alone_w

    # AU12 (questioning smile): sinyal sekunder — dipakai sebagai gate, bukan sinyal positif.
    # Craig (2008, p.785): AU12 & AU14 muncul di BOTH confusion & frustration (non-diskriminatif).
    # Karena itu AU12 SENGAJA dipakai konservatif (gate ber-floor), TIDAK sebagai pembeda kuat —
    # menambahkannya sebagai sinyal positif ke kedua emosi malah mengaburkan diskriminasi.
    smile_au12 = au["AU12"]

    # HAND-OVER-FACE → cue Confusion (beban kognitif / "unsure"). Basis berlapis:
    #   - Mahmoud & Robinson (2011): index-finger touching face muncul di 12 "thinking" + 2 "unsure"
    #     dari 15 segmen → "associated with cognitive mental states, namely thinking and unsure".
    #     "unsure" ≈ Confusion (D'Mello: "uncertainty about what to do next"). KUANTITATIF, 1 langkah.
    #   - Dong et al. (2026, ConfusionBench): "Hand-to-face actions such as touching the chin, pressing
    #     the forehead, and covering the mouth may indicate thinking, frustration, or hesitation."
    #   - Behera (2020): HoF ↑ saat difficulty ↑; Mahmoud (2016): HoF = "cognitive mental states".
    # max(hand_one, hand_two) — paper tidak bedakan jumlah untuk Confusion. Bobot final hand_conf_w=0.95
    # (≥ au7_alone_w): Mahmoud 2011 coverage 14/15 ≈ 93% ≥ AU7 (78%) → hand = cue KUAT, dinaikkan
    # agar override smile gate (lihat rules.py). Catatan: deteksi count-based, tak bisa bedakan
    # jari-aktif (kognitif) vs bersandar-pasif (rileks) → anotator manusia menilai posisi (dagu vs
    # dahi) saat review; skor otomatis sengaja kuat agar tidak under-detect.
    sig_hand_conf = max(r.hand_one, r.hand_two) * ccfg.get("hand_conf_w", 0.95)

    # MOUTH OPEN (AU25 Lips Part + AU26 Jaw Drop) → cue KUAT Confusion (Namba "most significant").
    # Chain dua paper:
    #   Namba et al. (2024): "Component 2 indicated opening the mouth (AU25, AU26)... can be
    #   considered the most significant component" saat menjawab pertanyaan sulit (thinking face).
    #   D'Mello & Graesser (2012): Confusion = cognitive disequilibrium saat menghadapi impasses —
    #   yaitu pertanyaan/materi sulit → chain: mulut terbuka saat sulit = cue Confusion.
    # Geometric mean AU25·AU26: keduanya harus aktif untuk skor kuat (mulut terbuka + rahang turun).
    au25_v = au.get("AU25", 0.0)
    au26_v = au.get("AU26", 0.0)
    au25_au26_co = (au25_v * au26_v) ** 0.5          # keduanya aktif bersamaan
    au25_alone   = max(au25_v, au26_v)               # salah satu saja sudah cukup (lemah)
    mouth_open_w = ccfg.get("mouth_open_conf_w", 0.78)
    sig_mouth_conf = max(au25_au26_co, au25_alone * 0.5) * mouth_open_w

    # base_conf = max() dari 5 cue: AU4 (95%, w1.0), AU7 (78%, w0.78), AU4+AU7 (73%, w0.50),
    # HoF/tangan (Mahmoud 93%, w0.95), mulut-mangap (Namba "most significant", w0.78).
    # Tangan & mulut kini cue KUAT (bukan lagi cue lemah) → confusion tidak under-detect.
    base_conf = max(brow_dn_v, au7_sig, au4_au7_sig, sig_hand_conf, sig_mouth_conf)
    conf = _clamp(base_conf * ccfg["blend_a"] + max(brow_dn_v, au7_v) * ccfg["blend_b"], 0, 1)

    # AU12 co-occurs dengan confusion — floor cegah senyum men-zero-kan confusion sepenuhnya
    smile_gate_th    = ccfg.get("smile_conf_gate_th", 0.45)
    smile_gate_floor = ccfg.get("smile_conf_gate_floor", 0.30)
    smile_gate = _clamp(1.0 - smile_au12 / max(smile_gate_th, 1e-6), smile_gate_floor, 1.0)
    conf = _clamp(conf * smile_gate, 0, 1)

    # CATATAN: bore→conf suppression DIHAPUS. D'Mello 2012: "Confusion→Boredom occurred at
    # chance levels" = TIDAK ada hubungan signifikan → "at chance" tidak membenarkan boredom
    # menekan confusion. Tak berdasar → dihilangkan (lihat DESIGN_RATIONALE §15).

    # == 3: FRUSTRATION ========================================================
    # Craig et al. (2008): AU1 (inner brow raise) + AU2 (outer brow raise) = PRIMARY frustration signals,
    # present in 100% of frustration episodes and mutually trigger each other.
    # Standard FACS (Ekman & Friesen 1978, confirmed by Grafsgaard 2013): AU1=inner, AU2=outer.
    # Intensitas AU baseline-normalized (au[...]): alis "diam" MediaPipe (~0.46) kini = 0
    # intensity, sehingga frustration TIDAK lagi over-fire pada wajah netral.
    biu_fr = au["AU1"]   # AU1 inner brow raise (browInnerUp)
    bou_fr = au["AU2"]   # AU2 outer brow raise (browOuterUp)
    # AU1+AU2 co-occurrence (geometric mean): kuat hanya jika kedua brow raise aktif bersamaan
    brow_raise_co = (biu_fr * bou_fr) ** 0.5

    # Grafsgaard et al. (2013, verbatim): "Action Unit 4 (brow lowering) was POSITIVELY correlated
    # with frustration" dan "Action Unit 14 (mouth dimpling) was positively correlated with both
    # frustration and learning gain." Paper ini = deteksi OTOMATIS CERT di konteks belajar (paling
    # cocok use-case kita) → AU4/AU14 DINAIKKAN dari sekunder ke PRIMER. AU4 sering nyala di py-feat
    # (median 0.31, p90 0.61) → frustration jadi lebih sering terdeteksi (jawab "frus jarang muncul").
    br_fr  = au["AU4"]    # AU4 brow lowerer  (Grafsgaard 2013: korelat POSITIF frustration)
    # AU14 (mouthDimple): saat tangan terdeteksi di wajah, mouthDimple MediaPipe dapat
    # meningkat akibat tekanan/deformasi pipi oleh tangan — bukan ekspresi dimpling asli.
    # Suppress AU14 proporsional ke kepercayaan deteksi tangan (hand_on_face 0→1).
    hand_on_face = max(r.hand_one, r.hand_two)
    dim_fr_adj = au["AU14"] * max(0.0, 1.0 - hand_on_face)
    face_secondary = max(br_fr, dim_fr_adj)  # AU4 atau AU14 (hand-adjusted) — Grafsgaard 2013

    # Craig et al. (2008) Table 2: AU1 sendiri (100%), AU2 sendiri (100%), dan AU1+AU2 bersama (100%)
    # → single brow raise PUN valid per paper (masing-masing 100% coverage); bobot parsial 0.70 = kalibrasi.
    # Grafsgaard et al. (2013): AU4/AU14 = sinyal sekunder.
    # Two-tier weighting:
    #   brow_raise_co (AU1+AU2 primary) → brow_raise_direct_w lebih tinggi (100% coverage co-occurrence)
    #   face_secondary (AU4/AU14)       → face_weight lebih rendah (secondary signal)
    brow_raise_direct_w = fcfg.get("brow_raise_direct_w", 0.55)  # Craig2008: AU1+AU2 primary (turun: butuh co-signal)
    face_w              = fcfg.get("face_weight", 0.65)           # AU4/AU14 pendukung (Grafsgaard 2013)

    sig_brow_raise = brow_raise_co * brow_raise_direct_w                   # AU1+AU2 co-occurrence, primary
    sig_bou_alone  = max(bou_fr * 0.70, biu_fr * 0.70) * brow_raise_direct_w  # single brow (Craig: tiap-tiap 100%), parsial
    sig_legacy     = face_secondary * face_w                               # AU4/AU14 secondary

    # Frustration → HANYA 2-tangan (hand_two), bukan 1-tangan. Basis:
    #   - Grafsgaard 2013b: two-hands-to-face ↔ self-efficacy RENDAH (temuan SIGNIFIKAN) ≈ Frustration.
    #   - Nojavanasghari et al. (2017, Hand2Face): "hand over face occlusions can provide additional
    #     information for recognition of... frustration and boredom."
    #   - Dong et al. (2026, ConfusionBench): hand-to-face → "thinking, frustration, or hesitation".
    # Cue pendukung (hand_frus_w 0.40) → MENAMBAH, tidak memicu sendiri. 1-tangan TIDAK ke Frustration
    # (Grafsgaard 2013b: 1-tangan = "thoughtful/less-negative", spesifik 2-tangan untuk self-efficacy).
    sig_hand_frus  = r.hand_two * fcfg.get("hand_frus_w", 0.40)
    sig_wajah_frus = _clamp(max(sig_brow_raise, sig_bou_alone, sig_legacy, sig_hand_frus), 0, 1)

    base_frus = sig_wajah_frus
    # Grafsgaard et al. (2013): komponen blend AU4/AU14 menambah sinyal frustration sekunder tervalidasi
    frus = _clamp(base_frus * fcfg["blend_a"] + face_secondary * fcfg["blend_b"], 0, 1)

    # ── Cross-suppression: DIHAPUS (tak berdasar / bertentangan dengan paper) ──
    # - Conf→Eng suppress DIHAPUS: D'Mello 2012 "Confusion→Engagement/Flow significant"
    #   (productive confusion = conf & eng CO-OCCUR) → menekan engagement saat confusion
    #   justru KEBALIKAN dari paper. Kini conf & eng boleh co-exist (multi-label DAiSEE).
    # - Frus→Bore suppress DIHAPUS: D'Mello frus→bore = transisi TEMPORAL, bukan supresi
    #   per-frame. Tak berdasar untuk diterapkan instan.
    # Yang DIPERTAHANKAN (berdasar): Bore↔Eng mutual suppression (di seksi Engagement) —
    # DAiSEE "complementary" + near-mutually-exclusive. Lihat DESIGN_RATIONALE §15.

    # Gupta et al. (2016) DAiSEE: "both boredom and engagement low → confusion/frustration high".
    # ⚠️ CAVEAT KONTEKS: DAiSEE mengecualikan state 'neutral'; data ini bisa punya frame netral
    # → boost ini bisa salah-tembak wajah netral. DEFAULT NONAKTIF (bore_eng_low_boost=0).
    _belb = ecfg.get("bore_eng_low_boost", 0.0)
    if _belb > 0:
        _bel_th = ecfg.get("bore_eng_low_th", 0.25)
        if bore < _bel_th and eng < _bel_th:
            conf = _clamp(conf + _belb, 0, 1)
            frus = _clamp(frus + _belb, 0, 1)

    # Debug log
    if _DBG_LAND:
        print(f"  [LAND] yaw={r.yaw:+.1f} pitch={r.pitch:+.1f} roll={r.roll:+.1f} iris_x={r.iris_x:+.3f} iris_y={r.iris_y:+.3f} "
              f"gH={gaze_h:+.1f}° gVup={gaze_v_up:.1f}° gVdn={gaze_v_down:.1f}° rollG={roll_gaze_eff:.1f}° devBore={gaze_dev_bore:.1f}° devEng={gaze_dev_eng:.1f}° | "
              f"rollGate={roll_gate:.2f} yawGate={yaw_gate:.2f} gate={gate:.2f} | "
              f"boreGaze={bore_gaze:.2f} | "
              f"B={bore:.3f} E={eng:.3f} C={conf:.3f} F={frus:.3f}")
        if conf > 0.5:
            print(f"  [CONF] AU4={brow_dn_v:.2f} AU7={au7_v:.2f} au4_au7_co={au4_au7_co:.2f} "
                  f"au4_au7_sig={au4_au7_sig:.2f} AU12_smile={smile_au12:.2f} base={base_conf:.2f}")
        if frus > 0.5:
            print(f"  [FRUS] AU1_inner={biu_fr:.2f} AU2_outer={bou_fr:.2f} brow_raise_co={brow_raise_co:.2f} "
                  f"AU4={br_fr:.2f} AU14={dim_fr_adj:.2f} sig_wajah={sig_wajah_frus:.2f}")

    return {
        0: round(bore, 4),
        1: round(eng,  4),
        2: round(conf, 4),
        3: round(frus, 4),
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_landmark_viz(frame_bgr: np.ndarray, r: LandmarkResult,
                      emotion_scores: dict | None = None,
                      src_size: int | None = None) -> np.ndarray:
    """
    Gambar overlay landmark/AU/emosi pada `frame_bgr`.

    src_size: resolusi (sisi, px) tempat koordinat px LandmarkResult (iris_px,
    hand_landmarks_px) dihitung. Jika kanvas `frame_bgr` lebih besar dari src_size
    (mis. viz dirender hi-res 512 dari frame asli sementara analisis di 224), semua
    ukuran gambar & titik px di-scale ×(w/src_size) agar tetap presisi & proporsional.
    Default None → sc=1.0 (perilaku lama, kompatibel mundur). Hanya display — tidak
    memengaruhi scoring (SigLIP/py-feat tetap 224).
    """
    viz = frame_bgr.copy()
    h, w = viz.shape[:2]
    sc = (w / float(src_size)) if src_size else 1.0
    def S(v):                      # skala ukuran/offset integer (min 1)
        return max(1, int(round(v * sc)))
    def SP(pt):                    # skala titik px ruang-src → kanvas
        return (int(round(pt[0] * sc)), int(round(pt[1] * sc)))

    if not r.face_found:
        cv2.putText(viz, "No face detected", (S(10), S(30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * sc, (0, 0, 255), S(2), cv2.LINE_AA)
        if r.hand_landmarks_px:
            from mediapipe.tasks.python.vision import HandLandmarksConnections
            n_hands = len(r.hand_landmarks_px) // 21
            for h_idx in range(n_hands):
                pts = [SP(p) for p in r.hand_landmarks_px[h_idx * 21: (h_idx + 1) * 21]]
                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    ci = conn.start if hasattr(conn, 'start') else conn[0]
                    cj = conn.end if hasattr(conn, 'end') else conn[1]
                    if ci < len(pts) and cj < len(pts):
                        cv2.line(viz, pts[ci], pts[cj], (0, 255, 0), S(2))
        return viz

    from mediapipe.tasks.python.vision import FaceLandmarksConnections, HandLandmarksConnections
    g = lambda k: r.blendshapes.get(k, 0.0)

    # ── 1. Face Mesh 3D (putih tipis) ────────────────────────────────────────────
    if r.face_landmarks:
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            si = connection.start if hasattr(connection, 'start') else connection[0]
            ei = connection.end if hasattr(connection, 'end') else connection[1]
            if si < len(r.face_landmarks) and ei < len(r.face_landmarks):
                sp = r.face_landmarks[si]; ep = r.face_landmarks[ei]
                cv2.line(viz, (int(sp.x*w), int(sp.y*h)),
                               (int(ep.x*w), int(ep.y*h)), (70, 70, 70), S(1))

    # ── 2. Hand Skeleton (hijau) ─────────────────────────────────────────────────
    if r.hand_landmarks_px:
        n_hands = len(r.hand_landmarks_px) // 21
        for h_idx in range(n_hands):
            raw = r.hand_landmarks_px[h_idx * 21: (h_idx + 1) * 21]
            if len(raw) < 21:
                continue
            pts = [SP(p) for p in raw]
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                ci = conn.start if hasattr(conn, 'start') else conn[0]
                cj = conn.end if hasattr(conn, 'end') else conn[1]
                if ci < len(pts) and cj < len(pts):
                    cv2.line(viz, pts[ci], pts[cj], (0, 255, 0), S(2))
            for pt in pts:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(viz, pt, S(3), (0, 200, 80), -1)

    # ── 3. Iris circles & per-eye gaze arrows ───────────────────────────────────
    eye_pairs = [
        (r.left_iris_px,  _L_INNER, _L_OUTER, _L_TOP, _L_BOT),
        (r.right_iris_px, _R_INNER, _R_OUTER, _R_TOP, _R_BOT),
    ]
    if r.face_landmarks:
        for (ipx0, ipy0), inner_i, outer_i, top_i, bot_i in eye_pairs:
            ipx, ipy = int(round(ipx0 * sc)), int(round(ipy0 * sc))
            if not (0 < ipx < w and 0 < ipy < h):
                continue
            cv2.circle(viz, (ipx, ipy), S(7), (0, 220, 255), S(2))
            for idx in [inner_i, outer_i, top_i, bot_i]:
                lm = r.face_landmarks[idx]
                cv2.circle(viz, (int(lm.x*w), int(lm.y*h)), S(2), (0, 140, 255), -1)
            inner = r.face_landmarks[inner_i]; outer = r.face_landmarks[outer_i]
            top   = r.face_landmarks[top_i];   bot   = r.face_landmarks[bot_i]
            ecx = int(((inner.x + outer.x) / 2) * w)
            ecy = int(((top.y  + bot.y)  / 2) * h)
            dx = ipx - ecx; dy = ipy - ecy
            ax = int(ecx + dx * 3.0); ay = int(ecy + dy * 3.0)
            ax = max(0, min(w-1, ax));  ay = max(0, min(h-1, ay))
            cv2.arrowedLine(viz, (ecx, ecy), (ax, ay), (255, 200, 0), S(2), tipLength=0.4)

    # ── 4. Head Yaw / Pitch indicator ───────────────────────────────────────────
    bar_cx = w // 2
    yaw_px = int(_clamp(r.yaw / 45, -1, 1) * (w // 3))
    yaw_color = (0, 80, 255) if abs(r.yaw) > 8 else (0, 200, 80)
    cv2.arrowedLine(viz, (bar_cx, S(18)), (bar_cx + yaw_px, S(18)), yaw_color, S(2), tipLength=0.3)
    cv2.putText(viz, f"Yaw{r.yaw:+.0f}", (bar_cx - S(22), S(13)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * sc, yaw_color, S(1), cv2.LINE_AA)

    pitch_cy = h // 2
    pitch_px = int(_clamp(r.pitch / 30, -1, 1) * (h // 4))
    pitch_color = (255, 100, 0) if abs(r.pitch) > 20 else (0, 200, 80)
    cv2.arrowedLine(viz, (S(14), pitch_cy), (S(14), pitch_cy - pitch_px), pitch_color, S(2), tipLength=0.3)
    cv2.putText(viz, f"P{r.pitch:+.0f}", (S(2), pitch_cy - pitch_px - S(4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32 * sc, pitch_color, S(1), cv2.LINE_AA)

    # ── 5. Signal bars (sisi kanan) — Blendshape Features (Turrisi 2026 + Aldenhoven 2026) ─
    # Blendshape → AU chain: Craig 2008 (AU→emosi) + Turrisi 2026 (BF→AU, κ=0.92).
    # eyeLookDown gating (Turrisi: AU64 = gaze direction ≠ AU7/AU43): ditampilkan sbg indikator.
    # AU25/AU26 keduanya dari jawOpen (proxy — ARKit tidak punya mouthOpen terpisah, Namba 2024).
    _eld = (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2
    _gate_start = 0.25
    _gate_full  = 0.60
    _ld_gate = max(0.0, 1.0 - max(0.0, min(1.0, (_eld - _gate_start) / max(_gate_full - _gate_start, 1e-6))))
    signals = [
        ("AU1",   g("browInnerUp"),                                         (255, 140,  80)),  # browInnerUp → Frus (Aldenhoven T1, Turrisi κ=1.00)
        ("AU2",   (g("browOuterUpLeft")+g("browOuterUpRight"))/2,           (255, 170,  90)),  # browOuterUp → Frus (Aldenhoven T1, Turrisi κ=1.00)
        ("AU4",   (g("browDownLeft")+g("browDownRight"))/2,                 (255,  80,  80)),  # browDown → Conf (Aldenhoven T1, Turrisi κ=1.00)
        ("AU7",   (g("eyeSquintLeft")+g("eyeSquintRight"))/2 * _ld_gate,   (255, 160,   0)),  # eyeSquint → Conf (gated by eyeLookDown)
        ("AU14",  (g("mouthDimpleLeft")+g("mouthDimpleRight"))/2,           (255,  60, 120)),  # mouthDimple → Frus (Aldenhoven T1)
        ("AU25",  g("jawOpen"),                                             (0,   180, 255)),  # jawOpen proxy AU25 (Namba 2024)
        ("AU26",  g("jawOpen"),                                             (0,   220, 180)),  # jawOpen → AU26 (Aldenhoven T1)
        ("AU43",  (g("eyeBlinkLeft")+g("eyeBlinkRight"))/2 * _ld_gate,     (100, 100, 255)),  # eyeBlink → Bore (gated by eyeLookDown)
        ("LkDn",  _eld,                                                     (180,  80, 255)),  # eyeLookDown = AU64 (Turrisi) — gate trigger
        ("Gate",  _ld_gate,                                                 (120, 255, 120)),  # gating aktif=1.0, mati=0.0
        ("EyeW",  (g("eyeWideLeft")+g("eyeWideRight"))/2,                  (80,  255, 200)),  # eyeWide → AU5 → Eng
        ("H1",    r.hand_one,                                               (0,   200, 255)),  # 1 tangan → Conf (Mahmoud 2011)
        ("H2",    r.hand_two,                                               (0,   255, 140)),  # 2 tangan → Frus (Grafsgaard 2013b)
    ]
    bar_w_max = S(60)
    bar_h_each = S(14)
    bar_x0 = w - bar_w_max - S(4)
    for idx, (label, val, color) in enumerate(signals):
        by = S(8) + idx * bar_h_each
        filled = int(val * bar_w_max)
        cv2.rectangle(viz, (bar_x0, by), (bar_x0 + bar_w_max, by + bar_h_each - S(2)), (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(viz, (bar_x0, by), (bar_x0 + filled, by + bar_h_each - S(2)), color, -1)
        cv2.putText(viz, f"{label}:{val:.2f}", (bar_x0 - S(2), by + bar_h_each - S(4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28 * sc, (220, 220, 220), S(1), cv2.LINE_AA)

    # ── 6. Hand zone lines ────────────────────────────────────────────────────────
    for zy in [int(h * 0.25), int(h * 0.55)]:
        cv2.line(viz, (0, zy), (w, zy), (60, 60, 60), S(1))

    # ── 7. Emotion score bars (bawah) ────────────────────────────────────────────
    if emotion_scores:
        emo_labels  = ["Bore", "Eng", "Conf", "Frus"]
        emo_colors  = [(0, 150, 255), (0, 255, 100), (255, 200, 0), (0, 80, 255)]
        for ei, (lbl, ecol) in enumerate(zip(emo_labels, emo_colors)):
            val = emotion_scores[ei]
            bar_y2 = h - S(8) - ei * S(14)
            filled2 = int(val * (w // 2))
            cv2.rectangle(viz, (0, bar_y2 - S(10)), (w // 2, bar_y2), (30, 30, 30), -1)
            if filled2 > 0:
                cv2.rectangle(viz, (0, bar_y2 - S(10)), (filled2, bar_y2), ecol, -1)
            cv2.putText(viz, f"{lbl}:{val:.2f}", (S(4), bar_y2 - S(2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32 * sc, (255, 255, 255), S(1), cv2.LINE_AA)

    return viz

