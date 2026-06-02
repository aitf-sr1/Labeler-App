"""
core/landmark_analyzer.py

Analisis wajah 3D menggunakan MediaPipe FaceLandmarker.

Menghasilkan per frame:
  - Head pose  : yaw (kiri/kanan) dan pitch (atas/bawah) dalam derajat
  - Iris offset : (x, y) ternormalisasi −1.0 s/d +1.0 relatif terhadap ukuran mata
  - Blendshapes : dict nama → skor 0.0–1.0
  - Skor emosi landmark (0–1) untuk hybrid scoring dengan SigLIP

Model diunduh otomatis ke ~/.cache/siglip_labeler/ pada pertama kali digunakan.
"""

import os
import math
import urllib.request
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

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
    hand_forehead:  float = 0.0  # tangan di zona dahi/mata (y 0–45%) → Frustration
    hand_chin:      float = 0.0  # tangan di zona pipi/dagu (y 40–80%) → Confusion
    hand_landmarks_px: list = field(default_factory=list)  # pixel positions untuk viz
    hand_landmarks_raw: list = field(default_factory=list) # raw mediapipe hand landmarks


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
    Deteksi tangan dan klasifikasikan posisinya dalam frame.

    Dalam 512×512 crop wajah:
      - y 0.00–0.45 → zona dahi/mata   → tangan di sini = Frustration
      - y 0.40–0.80 → zona pipi/dagu   → tangan di sini = Confusion

    Returns:
        (hand_forehead, hand_chin, hand_pts_px)
        hand_forehead: float 0–1, proporsi titik tangan di zona dahi
        hand_chin    : float 0–1, proporsi titik tangan di zona pipi/dagu
        hand_pts_px  : list[(x,y)] semua titik tangan untuk viz
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
    n = len(all_pts)
    pts_top = sum(1 for _, y in all_pts if -0.20 <= y < 0.25)
    pts_mid = sum(1 for _, y in all_pts if  0.25 <= y < 0.55)
    pts_bot = sum(1 for _, y in all_pts if  0.55 <= y <= 1.20)
    centered = sum(1 for x, _ in all_pts if 0.05 <= x <= 0.95)
    if _DBG_LAND: print(f"  [HAND] centered={centered}, pts_top={pts_top}, pts_mid={pts_mid}, pts_bot={pts_bot}")

    if centered < 5:
        if _DBG_LAND: print(f"  [HAND] DIFILTER: centered({centered}) < 5, skor di-reset 0")
        hand_top = hand_mid_bot = 0.0
    else:
        hand_top = _clamp(pts_top / 5, 0, 1)
        hand_mid_bot = _clamp((pts_mid + pts_bot) / 5, 0, 1)
    
    hand_pts_px = [(int(x * w), int(y * h)) for x, y in all_pts]

    return hand_top, hand_mid_bot, 0.0, hand_pts_px, res.hand_landmarks


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
        (hand_top, hand_mid_bot, hand_pts_px_in_crop, hand_raw_lms)
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

    n = len(all_pts_remapped)
    pts_top = sum(1 for _, y in all_pts_remapped if -0.20 <= y < 0.25)
    pts_mid = sum(1 for _, y in all_pts_remapped if  0.25 <= y < 0.55)
    pts_bot = sum(1 for _, y in all_pts_remapped if  0.55 <= y <= 1.20)
    # Tidak ada filter centered karena koordinat sudah di-remap ke crop
    in_crop = sum(1 for x, y in all_pts_remapped if -0.2 <= x <= 1.2 and -0.2 <= y <= 1.2)

    if _DBG_LAND:
        print(f"  [HAND-FULL] Terdeteksi {len(res.hand_landmarks)} tangan, in_crop={in_crop}, "
              f"pts_top={pts_top}, pts_mid={pts_mid}, pts_bot={pts_bot}")

    if in_crop < 3:
        return 0.0, 0.0, [], []

    hand_top     = _clamp(pts_top / 5, 0, 1)
    hand_mid_bot = _clamp((pts_mid + pts_bot) / 5, 0, 1)

    hand_pts_px = [(int(x * crop_size), int(y * crop_size)) for x, y in all_pts_remapped]
    return hand_top, hand_mid_bot, hand_pts_px, res.hand_landmarks


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
        hand_top, hand_mid_bot, hand_pts_px, hand_raw = injected_hand
    else:
        hand_top, hand_mid_bot, _, hand_pts_px, hand_raw = _analyze_hands(mp_img, h, w)

    if not res.face_landmarks:
        return LandmarkResult(
            face_found=False,
            hand_forehead=hand_top,                 # Frustration trigger (zona dahi y∈[-0.20, 0.25))
            hand_chin=hand_mid_bot,                 # Confusion trigger (zona pipi/dagu y∈[0.25, 1.20])
            hand_landmarks_px=hand_pts_px,
            hand_landmarks_raw=hand_raw,
        )

    lms = res.face_landmarks[0]
    bs  = {b.category_name: round(b.score, 4) for b in res.face_blendshapes[0]}
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
        hand_forehead=round(hand_top, 4),
        hand_chin=round(hand_mid_bot, 4),
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
      Dua metrik gaze deviation:
      - `gaze_dev_bore` (include roll) → Boredom. Kepala miring/noleh = bosan.
      - `gaze_dev` (tanpa roll) → Engagement gate. Roll ditangani roll_gate terpisah
        agar natural head tilt tidak mengurangi engagement.
      Tatapan lurus ke kamera → gaze rendah → Tidak Bosan, Engaged.
    """
    from .rules import DEFAULT_RULES
    if cfg is None:
        cfg = DEFAULT_RULES

    if not r.face_found:
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    g = lambda k: r.blendshapes.get(k, 0.0)

    gcfg = cfg["gaze"]
    bcfg = cfg["boredom"]
    ecfg = cfg["engagement"]
    ccfg = cfg["confusion"]
    fcfg = cfg["frustration"]

    # ── Iris Y reliability gate ───────────────────────────────────────────────
    # Saat mata hampir menutup (blink tinggi), MediaPipe melaporkan iris_y ekstrem
    # (iris di tepi atas celah mata yang nyaris menutup), bukan gaze sungguhan.
    # Suppress iris_y secara proporsional terhadap blink_corrected supaya artifact
    # menutupnya mata tidak polusi gaze_dev dan skor emosi.
    _blink_pre  = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
    _squint_pre = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
    _blink_corr_pre = max(0.0, _blink_pre - _squint_pre * bcfg.get("squint_blink_correction", 0.5))
    _iris_blink_th   = gcfg.get("iris_blink_suppress_th", 0.60)
    _iris_blink_zero = gcfg.get("iris_blink_zero_th", 0.0)  # 0 = disabled; > 0: zero iris_y jika blink_corr >= nilai ini
    if _iris_blink_zero > 0 and _blink_corr_pre >= _iris_blink_zero:
        # Mata sangat tertutup (misal: squint fokus) → iris_y tidak bisa dipercaya sama sekali
        _iris_y_factor = 0.0
    else:
        _iris_y_factor = max(0.0, 1.0 - _blink_corr_pre / max(_iris_blink_th, 1e-6))
    iris_y_eff      = r.iris_y * _iris_y_factor   # scaled iris_y; = r.iris_y saat mata terbuka

    # ── Gaze deviation (shared basis) ─────────────────────────────────────────
    GAZE_SCALE   = gcfg["scale_h"]
    GAZE_SCALE_V = gcfg["scale_v"]
    gaze_h     = r.yaw + r.iris_x * GAZE_SCALE
    gaze_v_raw = -r.pitch + iris_y_eff * GAZE_SCALE_V

    look_down_v = (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2

    # Pisahkan komponen vertikal atas dan bawah:
    #   - Ke atas : dead zone kecil (5°) — tatapan ke atas = melamun = boredom
    #   - Ke bawah: dead zone besar (15°) — nunduk = baca/ngetik/berpikir = BUKAN boredom
    gaze_v_up   = max(0.0, -gaze_v_raw)                             # komponen ke atas
    gaze_v_down = max(max(0.0, gaze_v_raw), look_down_v * 40)       # komponen ke bawah + look_down blendshape
    v_dz_up   = gcfg.get("v_dead_zone_up", 5.0)
    v_dz_down = gcfg["v_dead_zone"]
    gaze_v_up_eff   = max(0.0, gaze_v_up   - v_dz_up)
    gaze_v_down_eff = max(0.0, gaze_v_down - v_dz_down)
    gaze_v_eff = max(gaze_v_up_eff, gaze_v_down_eff)    # untuk engagement gate (atas dan bawah sama-sama suppress engagement)

    # gaze_h sudah = yaw + iris_x*scale_h — menggabungkan head pose + posisi iris.
    # Jika iris mengkompensasi head turn (natap depan walau kepala miring), gaze_h ≈ 0 → benar.
    # iris_side dan abs(yaw) di max() MENGABAIKAN kompensasi ini dan menyebabkan false positive:
    #   kepala miring 12° + iris kompensasi → gaze_h≈0 tapi iris_side=24° → boredom salah trigger.
    # Solusi: gunakan abs(gaze_h) saja — kompensasi mata sudah terhitung.
    gaze_h_eff = abs(gaze_h)

    # Roll (kepala miring, ear→shoulder).
    roll_gaze_eff = max(0.0, abs(r.roll) - gcfg.get("roll_dz", 5.0))

    # gaze_dev_bore: untuk BOREDOM — include roll, HANYA gaze ke atas (bukan ke bawah).
    #   Nunduk (ke bawah) = baca soal / ngetik / mikir = Engagement/Confusion, BUKAN Boredom.
    #   Boredom datang dari: gaze ke samping + gaze ke atas + kepala miring.
    #
    # Roll ditambahkan secara ADITIF (bukan kuadrat/Euclidean):
    #   Alasan: roll (kepala miring ke bahu) adalah dimensi inattention yang INDEPENDEN,
    #   bukan komponen vektor gaze yang sama. Dengan kuadrat, roll=10° + gH=10° = 14.1° (terlalu kecil).
    #   Dengan aditif, roll=10° + gH=10° = 20° — lebih mencerminkan dua cue distinct yang saling memperkuat.
    gaze_dev_bore = (gaze_h_eff ** 2 + gaze_v_up_eff ** 2) ** 0.5 + roll_gaze_eff

    # gaze_dev_eng: untuk ENGAGEMENT gate — TANPA roll, TANPA komponen ke bawah.
    #   Lihat ke bawah = baca/ngetik = ENGAGEMENT, tidak boleh mengurangi gate engagement.
    #   Ke bawah tetap dilindungi dead zone 15° via gaze_v_down_eff (sudah sangat toleran).
    gaze_dev_eng  = (gaze_h_eff ** 2 + gaze_v_up_eff ** 2) ** 0.5

    # gaze_dev: untuk confusion attentive gate — pakai SEMUA arah (atas+bawah), TANPA roll.
    #   Confusion membutuhkan attention ke konten — gaze jauh ke manapun mengurangi gate.
    gaze_dev      = (gaze_h_eff ** 2 + gaze_v_eff ** 2) ** 0.5

    # == 0: BOREDOM ============================================================
    bore_gaze_raw = _clamp((gaze_dev_bore - bcfg["gaze_dead_zone"]) / max(bcfg["gaze_range"], 1e-6), 0, 1)

    # MUTLAK: hadap depan = gaze tidak menyimpang → komponen bore_gaze di-nol-kan.
    # AU43 (blink_direct) tetap aktif independen dari gaze gate (Craig 2008: primary signal).
    # Range kecil (4°) agar transisi cepat: di gaze_h=9° bore_gaze sudah hampir penuh.
    bore_fwd_th    = bcfg.get("fwd_yaw_th", 5.0)    # turun dari 8° ke 5°
    bore_fwd_range = bcfg.get("fwd_yaw_range", 4.0)  # turun dari 8° ke 4°
    bore_fwd_gate  = _clamp((gaze_h_eff - bore_fwd_th) / max(bore_fwd_range, 1e-6), 0, 1)
    bore_gaze = bore_gaze_raw * bore_fwd_gate  # nol saat hadap depan, penuh di gaze_h≥9°

    blink_avg  = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
    squint_avg = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
    squint_blink_correction = squint_avg * bcfg.get("squint_blink_correction", 0.5)
    blink_corrected = max(0.0, blink_avg - squint_blink_correction)
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
    # AU43 (eye closure) = primary anti-engagement cue per Whitehill level descriptions
    blink_heavy_raw = max(0.0, blink_corrected - ecfg["blink_heavy_th"]) / max(ecfg["blink_heavy_th"], 1e-6)
    blink_heavy = blink_heavy_raw
    # eyeWide = inverse of "eyes barely open" (Whitehill level 2) — mild positive signal
    eye_wide    = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    eye_wide_boost = ecfg.get("eye_wide_boost", 0.20)
    # Pitch gate: mendongak ke atas = tidak fokus ke layar (Whitehill: "looking away from computer")
    pitch_gate_th    = ecfg.get("pitch_gate_th", 15.0)
    pitch_gate_range = ecfg.get("pitch_gate_range", 15.0)
    pitch_gate = _clamp(1.0 - max(0.0, r.pitch - pitch_gate_th) / max(pitch_gate_range, 1e-6), 0.0, 1.0)
    eng = _clamp(gate * yaw_gate * roll_gate * pitch_gate * max(ecfg["blink_heavy_min"], 1.0 - blink_heavy + eye_wide * eye_wide_boost), 0, 1)

    # D'Mello & Graesser (2012): Boredom dan Engagement near-mutually exclusive
    bore_suppress_th  = ecfg.get("bore_suppress_th", 0.45)
    bore_suppress_v   = _clamp((bore - bore_suppress_th) / max(1.0 - bore_suppress_th, 1e-6), 0, 1)
    bore_eng_suppress = ecfg.get("bore_eng_suppress", 0.40)
    eng = _clamp(eng - bore_suppress_v * bore_eng_suppress, 0, 1)

    # == 2: CONFUSION ==========================================================
    # Craig et al. (2008) Table 2: AU4 (brow lowerer) 95%, AU7 (lid tightener) 78%,
    # AU4+AU7 co-occurrence 73%, AU12 (questioning smile) 95% secondary.
    # Grafsgaard et al. (2011): AU4 validated via HMM as primary confusion predictor.
    # browInnerUp (AU2) is NOT in Craig 2008 for confusion — it is a frustration signal.
    # iris/gaze direction, head pitch, look_dn are NOT validated in any paper for confusion.

    # AU4 (brow lowerer): primary confusion signal
    brow_dn_v = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / max(ccfg["brow_dn_th"], 1e-6), 0, 1)

    # AU7 (lid tightener = eyeSquint): validated only in co-occurrence with AU4
    au7_th       = ccfg.get("au7_th", 0.15)
    au7_v        = _clamp(squint_avg / max(au7_th, 1e-6), 0, 1)
    au4_au7_co   = (brow_dn_v * au7_v) ** 0.5   # geometric mean — requires BOTH active simultaneously
    au4_au7_co_w = ccfg.get("au4_au7_co_w", 0.50)
    au4_au7_sig  = au4_au7_co * au4_au7_co_w

    # AU12 (mouthSmile = questioning smile): 95% coverage secondary — acts as gate, not positive signal
    smile_raw = max(g("mouthSmileLeft"), g("mouthSmileRight"))

    # base_conf: AU4 alone (95% coverage) OR AU4+AU7 together (73% coverage)
    base_conf = max(brow_dn_v, au4_au7_sig)
    conf = _clamp(base_conf * ccfg["blend_a"] + brow_dn_v * ccfg["blend_b"], 0, 1)

    # Craig et al. (2008): AU12 co-occurs 95% — floor prevents smile from zeroing confusion entirely
    smile_gate_th    = ccfg.get("smile_conf_gate_th", 0.35)
    smile_gate_floor = ccfg.get("smile_conf_gate_floor", 0.30)
    smile_gate = _clamp(1.0 - smile_raw / max(smile_gate_th, 1e-6), smile_gate_floor, 1.0)
    conf = _clamp(conf * smile_gate, 0, 1)

    # D'Mello & Graesser (2012): "Confusion→Boredom occurred at chance" — boredom suppresses confusion
    bore_conf_suppress_val = ccfg.get("bore_conf_suppress_bore", 0.40)
    conf = _clamp(conf - bore * bore_conf_suppress_val, 0, 1)

    # == 3: FRUSTRATION ========================================================
    # Craig et al. (2008): AU1 (inner brow raise) + AU2 (outer brow raise) = PRIMARY frustration signals,
    # present in 100% of frustration episodes and mutually trigger each other.
    # Standard FACS (Ekman & Friesen 1978, confirmed by Grafsgaard 2013): AU1=inner, AU2=outer.
    bou_fr = _clamp(
        (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 / max(fcfg.get("brow_outer_up_th", 0.20), 1e-6), 0, 1
    )  # AU2 outer brow raise (browOuterUp)
    biu_fr = _clamp(
        g("browInnerUp") / max(fcfg.get("brow_inner_up_th", 0.20), 1e-6), 0, 1
    )  # AU1 inner brow raise (browInnerUp)
    # AU1+AU2 co-occurrence (geometric mean): fires strongly only when both brow raises are active simultaneously
    brow_raise_co = (bou_fr * biu_fr) ** 0.5

    # Grafsgaard et al. (2013): AU4 (brow lowering) positively correlated with frustration
    br_fr = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / max(fcfg["brow_dn_th"], 1e-6), 0, 1)

    # Craig et al. (2008): AU1+AU2 co-occurrence = 100% coverage (primary)
    # Grafsgaard et al. (2013): AU4 (brow lowering) positively correlated with frustration (secondary)
    # Two-tier weighting:
    #   brow_raise_co (AU1+AU2 primary) → brow_raise_direct_w lebih tinggi (100% coverage)
    #   face_secondary (AU4)             → face_weight lebih rendah (secondary signal)
    face_secondary = br_fr  # AU4 (browDown), validated by Grafsgaard et al. (2013)
    brow_raise_direct_w = fcfg.get("brow_raise_direct_w", 0.65)  # Craig2008: AU1+AU2 primary
    face_w              = fcfg.get("face_weight", 0.45)           # untuk AU4 secondary

    sig_brow_raise = brow_raise_co * brow_raise_direct_w                   # AU1+AU2 co-occurrence, primary
    sig_bou_alone  = max(bou_fr * 0.70, biu_fr * 0.70) * brow_raise_direct_w  # single brow raise, partial
    sig_legacy     = face_secondary * face_w                               # AU4 secondary
    sig_wajah_frus = _clamp(max(sig_brow_raise, sig_bou_alone, sig_legacy), 0, 1)

    base_frus = sig_wajah_frus
    # Grafsgaard et al. (2013): AU4 blend component adds validated secondary frustration signal
    frus = _clamp(base_frus * fcfg["blend_a"] + br_fr * fcfg["blend_b"], 0, 1)

    # ── Post-computation cross-suppression ────────────────────────────────────
    # Confusion → Engagement: orang bingung yang menatap layar tidak sama dengan engaged.
    # Floor fwd_eng_min (0.35) sengaja diturunkan agar suppression ini bisa mengalahkannya.
    # Dihitung SETELAH semua skor final — ini mengoverride floor engagement saat conf tinggi.
    conf_eng_sup_th = ecfg.get("conf_eng_suppress_th", 0.40)
    conf_eng_sup    = ecfg.get("conf_eng_suppress",    0.55)
    conf_sup_v = _clamp((conf - conf_eng_sup_th) / max(1.0 - conf_eng_sup_th, 1e-6), 0, 1)
    eng = _clamp(eng - conf_sup_v * conf_eng_sup, 0, 1)

    # Frustration → Boredom: ekspresi tegang frustrasi sering overlap dengan sinyal bosan
    # (browDown, blink berat). Ketika frustrasi dominan, boredom ditekan.
    frus_bore_sup_th = bcfg.get("frus_bore_suppress_th", 0.40)
    frus_bore_sup    = bcfg.get("frus_bore_suppress",    0.45)
    frus_sup_v = _clamp((frus - frus_bore_sup_th) / max(1.0 - frus_bore_sup_th, 1e-6), 0, 1)
    bore = _clamp(bore - frus_sup_v * frus_bore_sup, 0, 1)

    # Debug log
    if _DBG_LAND:
        print(f"  [LAND] yaw={r.yaw:+.1f} pitch={r.pitch:+.1f} roll={r.roll:+.1f} iris_x={r.iris_x:+.3f} iris_y={r.iris_y:+.3f} "
              f"lookDn={look_down_v:.2f} | gH={gaze_h:+.1f}° gVup={gaze_v_up:.1f}° gVdn={gaze_v_down:.1f}° rollG={roll_gaze_eff:.1f}° devBore={gaze_dev_bore:.1f}° devEng={gaze_dev:.1f}° | "
              f"rollGate={roll_gate:.2f} yawGate={yaw_gate:.2f} gate={gate:.2f} | "
              f"boreGaze={bore_gaze:.2f} | "
              f"B={bore:.3f} E={eng:.3f} C={conf:.3f} F={frus:.3f}")
        if conf > 0.5:
            print(f"  [CONF] brow_dn(AU4)={brow_dn_v:.2f} au7(AU7)={au7_v:.2f} au4_au7_co={au4_au7_co:.2f} "
                  f"au4_au7_sig={au4_au7_sig:.2f} smile_raw={smile_raw:.2f} base={base_conf:.2f}")
        if frus > 0.5:
            print(f"  [FRUS] bou(AU1)={bou_fr:.2f} biu(AU2)={biu_fr:.2f} brow_raise_co={brow_raise_co:.2f} "
                  f"br(AU4)={br_fr:.2f} sig_wajah={sig_wajah_frus:.2f}")

    return {
        0: round(bore, 4),
        1: round(eng,  4),
        2: round(conf, 4),
        3: round(frus, 4),
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_landmark_viz(frame_bgr: np.ndarray, r: LandmarkResult,
                      emotion_scores: dict | None = None) -> np.ndarray:
    viz = frame_bgr.copy()
    h, w = viz.shape[:2]

    if not r.face_found:
        cv2.putText(viz, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        if r.hand_landmarks_px:
            from mediapipe.tasks.python.vision import HandLandmarksConnections
            n_hands = len(r.hand_landmarks_px) // 21
            for h_idx in range(n_hands):
                pts = r.hand_landmarks_px[h_idx * 21: (h_idx + 1) * 21]
                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    s = conn.start if hasattr(conn, 'start') else conn[0]
                    e = conn.end if hasattr(conn, 'end') else conn[1]
                    if s < len(pts) and e < len(pts):
                        cv2.line(viz, pts[s], pts[e], (0, 255, 0), 2)
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
                               (int(ep.x*w), int(ep.y*h)), (70, 70, 70), 1)

    # ── 2. Hand Skeleton (hijau) ─────────────────────────────────────────────────
    if r.hand_landmarks_px:
        n_hands = len(r.hand_landmarks_px) // 21
        for h_idx in range(n_hands):
            pts = r.hand_landmarks_px[h_idx * 21: (h_idx + 1) * 21]
            if len(pts) < 21:
                continue
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                s = conn.start if hasattr(conn, 'start') else conn[0]
                e = conn.end if hasattr(conn, 'end') else conn[1]
                if s < len(pts) and e < len(pts):
                    cv2.line(viz, pts[s], pts[e], (0, 255, 0), 2)
            for pt in pts:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(viz, pt, 3, (0, 200, 80), -1)

    # ── 3. Iris circles & per-eye gaze arrows ───────────────────────────────────
    eye_pairs = [
        (r.left_iris_px,  _L_INNER, _L_OUTER, _L_TOP, _L_BOT),
        (r.right_iris_px, _R_INNER, _R_OUTER, _R_TOP, _R_BOT),
    ]
    if r.face_landmarks:
        for (ipx, ipy), inner_i, outer_i, top_i, bot_i in eye_pairs:
            if not (0 < ipx < w and 0 < ipy < h):
                continue
            cv2.circle(viz, (ipx, ipy), 7, (0, 220, 255), 2)
            for idx in [inner_i, outer_i, top_i, bot_i]:
                lm = r.face_landmarks[idx]
                cv2.circle(viz, (int(lm.x*w), int(lm.y*h)), 2, (0, 140, 255), -1)
            inner = r.face_landmarks[inner_i]; outer = r.face_landmarks[outer_i]
            top   = r.face_landmarks[top_i];   bot   = r.face_landmarks[bot_i]
            ecx = int(((inner.x + outer.x) / 2) * w)
            ecy = int(((top.y  + bot.y)  / 2) * h)
            dx = ipx - ecx; dy = ipy - ecy
            ax = int(ecx + dx * 3.0); ay = int(ecy + dy * 3.0)
            ax = max(0, min(w-1, ax));  ay = max(0, min(h-1, ay))
            cv2.arrowedLine(viz, (ecx, ecy), (ax, ay), (255, 200, 0), 2, tipLength=0.4)

    # ── 4. Head Yaw / Pitch indicator ───────────────────────────────────────────
    bar_cx = w // 2
    yaw_px = int(_clamp(r.yaw / 45, -1, 1) * (w // 3))
    yaw_color = (0, 80, 255) if abs(r.yaw) > 8 else (0, 200, 80)
    cv2.arrowedLine(viz, (bar_cx, 18), (bar_cx + yaw_px, 18), yaw_color, 2, tipLength=0.3)
    cv2.putText(viz, f"Yaw{r.yaw:+.0f}", (bar_cx - 22, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, yaw_color, 1, cv2.LINE_AA)

    pitch_cy = h // 2
    pitch_px = int(_clamp(r.pitch / 30, -1, 1) * (h // 4))
    pitch_color = (255, 100, 0) if abs(r.pitch) > 20 else (0, 200, 80)
    cv2.arrowedLine(viz, (14, pitch_cy), (14, pitch_cy - pitch_px), pitch_color, 2, tipLength=0.3)
    cv2.putText(viz, f"P{r.pitch:+.0f}", (2, pitch_cy - pitch_px - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, pitch_color, 1, cv2.LINE_AA)

    # ── 5. Blendshape signal bars (sisi kanan) ───────────────────────────────────
    signals = [
        ("Blink",  max(g("eyeBlinkLeft"),   g("eyeBlinkRight")),          (100, 100, 255)),
        ("Yawn",   g("jawOpen"),                                           (255, 180,   0)),
        ("BrowDn", (g("browDownLeft") + g("browDownRight")) / 2,           (255,  80,  80)),
        ("BrowIn", g("browInnerUp"),                                       (255, 140,  80)),
        ("LookUp", max(g("eyeLookUpLeft"),  g("eyeLookUpRight")),          (180, 255, 100)),
        ("Smile",  max(g("mouthSmileLeft"), g("mouthSmileRight")),         (0,   220, 220)),
        ("Pucker", g("mouthPucker"),                                       (180,  80, 255)),
        ("Sneer",  max(g("noseSneerLeft"),  g("noseSneerRight")),          (255,  40,  40)),
        ("Press",  (g("mouthPressLeft") + g("mouthPressRight")) / 2,      (200,  80, 200)),
        ("Squint", (g("eyeSquintLeft") + g("eyeSquintRight")) / 2,        (255, 160,   0)),
        ("Frown",  (g("mouthFrownLeft") + g("mouthFrownRight")) / 2,      (255,  60, 120)),
        ("EyeWide",(g("eyeWideLeft") + g("eyeWideRight")) / 2,            (80,  255, 200)),
        ("HFore",  r.hand_forehead,                                       (0,   200, 255)),
        ("HChin",  r.hand_chin,                                           (0,   255, 140)),
    ]
    bar_w_max = 60
    bar_h_each = 14
    bar_x0 = w - bar_w_max - 4
    for idx, (label, val, color) in enumerate(signals):
        by = 8 + idx * bar_h_each
        filled = int(val * bar_w_max)
        cv2.rectangle(viz, (bar_x0, by), (bar_x0 + bar_w_max, by + bar_h_each - 2), (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(viz, (bar_x0, by), (bar_x0 + filled, by + bar_h_each - 2), color, -1)
        cv2.putText(viz, f"{label}:{val:.2f}", (bar_x0 - 2, by + bar_h_each - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 6. Hand zone lines ────────────────────────────────────────────────────────
    for zy in [int(h * 0.25), int(h * 0.55)]:
        cv2.line(viz, (0, zy), (w, zy), (60, 60, 60), 1)

    # ── 7. Emotion score bars (bawah) ────────────────────────────────────────────
    if emotion_scores:
        emo_labels  = ["Bore", "Eng", "Conf", "Frus"]
        emo_colors  = [(0, 150, 255), (0, 255, 100), (255, 200, 0), (0, 80, 255)]
        for ei, (lbl, ecol) in enumerate(zip(emo_labels, emo_colors)):
            val = emotion_scores[ei]
            bar_y2 = h - 8 - ei * 14
            filled2 = int(val * (w // 2))
            cv2.rectangle(viz, (0, bar_y2 - 10), (w // 2, bar_y2), (30, 30, 30), -1)
            if filled2 > 0:
                cv2.rectangle(viz, (0, bar_y2 - 10), (filled2, bar_y2), ecol, -1)
            cv2.putText(viz, f"{lbl}:{val:.2f}", (4, bar_y2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    return viz

