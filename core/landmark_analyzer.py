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

    # ── Teeth gate (shared): gigi terlihat = BUKAN menguap ────────────────────
    # Menguap = mulut terbuka lebar, bibir menutupi gigi (bentuk O).
    # Jika gigi kelihatan (bibir atas naik / senyum) → jawOpen bukan menguap.
    teeth_upper = (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2
    teeth_smile = max(g("mouthSmileLeft"), g("mouthSmileRight"))
    teeth_signal = max(teeth_upper, teeth_smile)
    teeth_gate_th = bcfg.get("teeth_gate_th", 0.20)
    teeth_gate = _clamp(1.0 - teeth_signal / max(teeth_gate_th, 1e-6), 0, 1)

    # == 0: BOREDOM ============================================================
    bore_gaze_raw = _clamp((gaze_dev_bore - bcfg["gaze_dead_zone"]) / max(bcfg["gaze_range"], 1e-6), 0, 1)

    # MUTLAK: hadap depan = gaze tidak menyimpang → komponen bore_gaze di-nol-kan.
    # Tapi ekspresi (yawn, blink) dan chin-resting TETAP valid.
    # Range kecil (4°) agar transisi cepat: di gaze_h=9° boredom sudah hampir penuh.
    bore_fwd_th    = bcfg.get("fwd_yaw_th", 5.0)    # turun dari 8° ke 5°
    bore_fwd_range = bcfg.get("fwd_yaw_range", 4.0)  # turun dari 8° ke 4°
    bore_fwd_gate  = _clamp((gaze_h_eff - bore_fwd_th) / max(bore_fwd_range, 1e-6), 0, 1)
    bore_gaze = bore_gaze_raw * bore_fwd_gate  # nol saat hadap depan, penuh di gaze_h≥9°

    blink_avg  = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
    squint_avg = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
    squint_blink_correction = squint_avg * bcfg.get("squint_blink_correction", 0.5)
    blink_corrected = max(0.0, blink_avg - squint_blink_correction)
    blink_v    = _clamp((blink_corrected - bcfg["blink_dead_zone"]) / max(bcfg["blink_range"], 1e-6), 0, 1)
    yawn_dz    = bcfg.get("yawn_dead_zone", 0.20)
    _jaw_yawn  = max(0.0, g("jawOpen") - yawn_dz)
    yawn_raw   = _clamp(_jaw_yawn / max(bcfg["yawn_threshold"] - yawn_dz, 1e-6), 0, 1) * teeth_gate if r.pitch < 15 else 0.0
    pitch_up_v = _clamp((r.pitch - bcfg.get("pitch_up_th", 20.0)) / max(bcfg.get("pitch_up_range", 25.0), 1e-6), 0, 1)

    # expr_gate dari bore_gaze yang sudah di-gate → blink/droopy tidak fire saat natap layar
    expr_gate = _clamp(bore_gaze / max(bcfg.get("expr_gaze_gate_th", 0.35), 1e-6), 0, 1)

    # yawn_direct: bypass gaze gate kalau yawn kuat (menguap nyata = bosan meski natap layar)
    jaw_open_raw      = g("jawOpen") * teeth_gate if r.pitch < 15 else 0.0
    yawn_strong_th    = bcfg.get("yawn_strong_th", 0.50)
    yawn_strong_range = bcfg.get("yawn_strong_range", 0.25)
    yawn_gate_bypass  = _clamp((jaw_open_raw - yawn_strong_th) / max(yawn_strong_range, 1e-6), 0, 1)
    yawn_direct = yawn_raw * bcfg.get("yawn_bore_w", 0.75) * max(yawn_gate_bypass, expr_gate)
    sig_expr       = max(blink_v, pitch_up_v) * bcfg["sig_expr_weight"]
    sig_expr_gated = sig_expr * expr_gate

    # Craig et al. (2008): AU43 (eye closure) = primary boredom signal, independent of gaze deviation.
    # Students show droopy/heavy eyelids when bored even while still facing screen.
    blink_direct_th = bcfg.get("blink_direct_th", 0.45)
    blink_direct_w  = bcfg.get("blink_direct_w", 0.45)
    blink_direct = _clamp(
        (blink_corrected - blink_direct_th) / max(bcfg["blink_range"], 1e-6), 0, 1
    ) * blink_direct_w

    base_bore = max(bore_gaze, sig_expr_gated, yawn_direct, blink_direct)
    bore      = _clamp(base_bore * bcfg["blend_a"] + (bore_gaze + sig_expr_gated) * bcfg["blend_b"], 0, 1)
    _eye_wide_pre = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    bore = _clamp(bore - _eye_wide_pre * bcfg.get("eye_wide_suppress", 0.3), 0, 1)
    bore = _clamp(bore - squint_avg * bcfg.get("squint_suppress", 0.3), 0, 1)
    # browInnerUp tinggi = waspada/fokus/khawatir — bukan ekspresi bosan.
    # Suppress boredom proporsional ketika inner brows naik signifikan.
    _biu        = g("browInnerUp")
    _biu_th     = bcfg.get("brow_inner_suppress_th", 0.45)
    _biu_max    = bcfg.get("brow_inner_suppress",    0.55)
    _biu_sup    = _clamp((_biu - _biu_th) / max(1.0 - _biu_th, 1e-6), 0, 1)
    bore = _clamp(bore - _biu_sup * _biu_max, 0, 1)
    smile_gaze_max = bcfg.get("smile_gaze_max", 15.0)
    facing_fwd_bore = _clamp(1.0 - gaze_dev_bore / max(smile_gaze_max, 1e-6), 0, 1)
    bore = _clamp(bore - teeth_signal * bcfg.get("smile_suppress", 0.40) * facing_fwd_bore, 0, 1)

    # chin-resting: bisa bosan meski natap layar (tidak butuh gaze gate)
    chin_bore_th    = bcfg.get("chin_bore_th",    0.30)
    chin_bore_range = bcfg.get("chin_bore_range", 0.40)
    chin_bore_max   = bcfg.get("chin_bore_max",   0.70)
    # Jaw gate: menopang dagu (mulut tertutup) = boredom pasif.
    # Mulut terbuka + tangan di dagu = bicara/aktif → chin TIDAK boost boredom.
    _chin_jaw_dz  = bcfg.get("chin_jaw_closed_dz", 0.08)  # di bawah ini = mulut "tertutup"
    _chin_jaw_th  = bcfg.get("chin_jaw_open_th",   0.18)  # di atas ini   = mulut "terbuka"
    _jaw_open_for_bore = _clamp(
        (g("jawOpen") - _chin_jaw_dz) / max(_chin_jaw_th - _chin_jaw_dz, 1e-6), 0, 1
    )
    chin_bore_v = _clamp((r.hand_chin - chin_bore_th) / max(chin_bore_range, 1e-6), 0, 1)
    bore = _clamp(max(bore, chin_bore_v * chin_bore_max * (1.0 - _jaw_open_for_bore)), 0, 1)
    # Tidak ada bore_fwd_gate pada bore final — sudah diterapkan di bore_gaze di awal

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

    # Sama seperti boredom: pakai rata-rata dan koreksi squint agar sipit tidak dianggap merem
    blink_heavy_raw = max(0.0, blink_corrected - ecfg["blink_heavy_th"]) / max(ecfg["blink_heavy_th"], 1e-6)
    blink_heavy = blink_heavy_raw
    # Mata terbuka lebar (eyeWide) = sinyal attentif/fokus — kompensasi blink ringan
    eye_wide    = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    eye_wide_boost = ecfg.get("eye_wide_boost", 0.20)
    # Sipit (eyeSquint) = konsentrasi aktif — juga sinyal engagement, seperti eye_wide tapi lebih subtle
    eye_squint_boost = ecfg.get("eye_squint_boost", 0.15)
    # Menguap = tidak engaged. Pakai jawOpen langsung (BUKAN yawn_raw yang sudah dinormalisasi
    # oleh boredom yawn_threshold) agar threshold engagement independen dan lebih stabil.
    jaw_open_v   = g("jawOpen") * teeth_gate if r.pitch < 15 else 0.0
    yawn_eng_pen = _clamp(jaw_open_v / max(ecfg.get("yawn_eng_th", 0.55), 1e-6), 0, 1) * ecfg.get("yawn_eng_pen_w", 0.35)
    # Pitch gate: mendongak ke atas (pitch besar) = tidak fokus ke layar → engagement turun.
    # Dead zone 15° agar variasi postur duduk normal tidak kena penalti (pitch 5-12° = wajar).
    pitch_gate_th    = ecfg.get("pitch_gate_th", 15.0)
    pitch_gate_range = ecfg.get("pitch_gate_range", 15.0)
    pitch_gate = _clamp(1.0 - max(0.0, r.pitch - pitch_gate_th) / max(pitch_gate_range, 1e-6), 0.0, 1.0)
    eng = _clamp(gate * yaw_gate * roll_gate * pitch_gate * max(ecfg["blink_heavy_min"], 1.0 - blink_heavy - yawn_eng_pen + eye_wide * eye_wide_boost + squint_avg * eye_squint_boost), 0, 1)
    # Senyum/gigi = sinyal engagement, tapi hanya kalau hadap depan (ketawa ke temen = bukan engaged)
    smile_gaze_max_eng = ecfg.get("smile_gaze_max", 15.0)
    facing_fwd_eng = _clamp(1.0 - gaze_dev_eng / max(smile_gaze_max_eng, 1e-6), 0, 1)
    eng = _clamp(eng + teeth_signal * ecfg.get("smile_boost", 0.30) * facing_fwd_eng, 0, 1)

    # Lihat ke bawah + hadap depan = baca/ngetik = boost engagement langsung.
    # Gated oleh yaw (noleh sambil nunduk = bukan engaged ke konten).
    look_dn_eng_th    = ecfg.get("look_dn_eng_th", 0.25)
    look_dn_eng_boost = ecfg.get("look_dn_eng_boost", 0.20)
    look_dn_eng_yaw_max = ecfg.get("look_dn_eng_yaw_max", 20.0)
    look_dn_eng_gate  = _clamp(1.0 - abs(r.yaw) / max(look_dn_eng_yaw_max, 1e-6), 0, 1)
    look_dn_eng_v     = _clamp((look_down_v - look_dn_eng_th) / max(0.3, 1e-6), 0, 1)
    eng = _clamp(eng + look_dn_eng_v * look_dn_eng_boost * look_dn_eng_gate, 0, 1)

    # Gaze tepat ke depan (dalam dead zone) = sinyal kuat engagement.
    # Formula selama ini hanya menghindari penalti, tidak ada bonus aktif saat benar-benar natap layar.
    gaze_fwd_v     = _clamp(1.0 - gaze_dev_eng / max(ecfg["tegak_dead_zone"], 1e-6), 0, 1)
    gaze_fwd_bonus = ecfg.get("gaze_fwd_bonus", 0.15)
    eng = _clamp(eng + gaze_fwd_v * gaze_fwd_bonus, 0, 1)

    # Menopang dagu: penalti engagement HANYA ketika mulut tertutup (postur pasif).
    # Mulut terbuka + tangan di dagu = bicara sambil gestur → aktif/engaged.
    chin_eng_pen_w = ecfg.get("hand_chin_eng_pen", 0.5)
    _cjdz  = ecfg.get("chin_jaw_closed_dz",    0.08)
    _cjth  = ecfg.get("chin_jaw_open_th",       0.18)
    _jaw_open_for_eng = _clamp(
        (g("jawOpen") - _cjdz) / max(_cjth - _cjdz, 1e-6), 0, 1
    )
    # Penalti berkurang proporsional saat jaw terbuka
    _pen_reduce = ecfg.get("chin_jaw_open_pen_reduce", 0.80)
    eng = _clamp(eng - r.hand_chin * chin_eng_pen_w * (1.0 - _jaw_open_for_eng * _pen_reduce), 0, 1)
    # Tangan di dagu + mulut terbuka = bicara/aktif → boost kecil engagement
    _chin_open_boost = ecfg.get("chin_mouth_open_boost", 0.20)
    eng = _clamp(eng + r.hand_chin * _jaw_open_for_eng * _chin_open_boost, 0, 1)

    # Boredom dan engagement adalah near-mutually exclusive secara semantik.
    # Dead zone 0.30 — boredom > 0.30 mulai suppress engagement secara agresif.
    bore_suppress_th  = ecfg.get("bore_suppress_th", 0.30)
    bore_suppress_v   = _clamp((bore - bore_suppress_th) / max(1.0 - bore_suppress_th, 1e-6), 0, 1)
    bore_eng_suppress = ecfg.get("bore_eng_suppress", 0.70)
    eng = _clamp(eng - bore_suppress_v * bore_eng_suppress, 0, 1)

    # FLOOR hadap depan — tapi HANYA ketika boredom rendah.
    # Kalau boredom sudah tinggi (bore > 0.30), floor tidak berlaku:
    # siswa yang bosan tapi masih natap layar tidak boleh dipaksa jadi engaged.
    fwd_eng_min     = ecfg.get("fwd_eng_min", 0.55)
    fwd_eng_gaze_th = ecfg.get("fwd_eng_gaze_max", 10.0)
    fwd_eng_gate    = _clamp(1.0 - gaze_dev_eng / max(fwd_eng_gaze_th, 1e-6), 0, 1)
    bore_floor_cancel = _clamp(bore / max(ecfg.get("bore_floor_cancel_th", 0.30), 1e-6), 0, 1)
    eng = max(eng, fwd_eng_min * fwd_eng_gate * (1.0 - bore_floor_cancel))

    # == 2: CONFUSION ==========================================================
    iris_up_v  = _clamp((-iris_y_eff - ccfg["iris_up_dead_zone"]) / max(ccfg["iris_up_range"], 1e-6), 0, 1)
    look_up_v  = _clamp(max(g("eyeLookUpLeft"), g("eyeLookUpRight")) / max(ccfg["look_up_threshold"], 1e-6), 0, 1)
    # Lihat ke bawah (baca soal/layar) juga bisa confusion — gated oleh hadap depan
    look_dn_th = ccfg.get("look_dn_th", 0.40)
    look_dn_v  = _clamp((look_down_v - look_dn_th) / max(ccfg.get("look_dn_range", 0.30), 1e-6), 0, 1)
    # Hanya aktif kalau hadap depan (yaw kecil) — lihat bawah sambil noleh = bukan confusion
    look_dn_fwd_gate = _clamp(1.0 - abs(r.yaw) / max(ccfg.get("look_dn_yaw_max", 15.0), 1e-6), 0, 1)
    look_dn_v *= look_dn_fwd_gate

    pitch_cu   = _clamp((r.pitch - ccfg["pitch_start"]) / max(ccfg["pitch_range"], 1e-6), 0, 1)
    brow_dn_v  = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / max(ccfg["brow_dn_th"], 1e-6), 0, 1)
    brow_in_raw = g("browInnerUp")

    # squint sebagai sinyal mata: sipit + alis naik dalam = thinking/confused
    # Dijadikan co_signal khusus browInnerUp (iris/mata only, per kalibrasi)
    squint_conf_th    = ccfg.get("squint_conf_th", 0.15)
    squint_conf_range = ccfg.get("squint_conf_range", 0.25)
    squint_as_co  = _clamp(squint_avg / max(squint_conf_th, 1e-6), 0, 1)
    squint_conf_v = _clamp((squint_avg - squint_conf_th) / max(squint_conf_range, 1e-6), 0, 1)

    co_signal  = max(iris_up_v, look_up_v, pitch_cu, look_dn_v)
    brow_in_co = max(iris_up_v, look_up_v, squint_as_co)   # iris/mata only
    brow_in_v  = _clamp(brow_in_raw / max(ccfg["brow_in_th"], 1e-6), 0, 1) * _clamp(brow_in_co / max(ccfg["brow_in_co_gate"], 1e-6), 0, 1)

    # Kepala miring ke samping (roll) — bukan muterin kepala (yaw) seperti boredom.
    # Siswa bingung sering miringin kepala sedikit. Dead zone supaya noise tidak trigger.
    roll_v = _clamp((abs(r.roll) - ccfg.get("roll_dead_zone", 8.0)) / max(ccfg.get("roll_range", 15.0), 1e-6), 0, 1)

    smile_raw  = max(g("mouthSmileLeft"), g("mouthSmileRight"))
    smile_pen  = max(0.0, smile_raw - ccfg["smile_penalty_th"])

    jo = g("jawOpen")
    jaw_rise = max(ccfg["jaw_peak"] - ccfg["jaw_start"], 1e-6)
    jaw_fall = max(ccfg["jaw_end"]  - ccfg["jaw_peak"],  1e-6)
    if jo <= ccfg["jaw_start"]:
        jaw_val_conf = 0.0
    elif jo <= ccfg["jaw_peak"]:
        jaw_val_conf = (jo - ccfg["jaw_start"]) / jaw_rise
    elif jo <= ccfg["jaw_end"]:
        jaw_val_conf = 1.0 - (jo - ccfg["jaw_peak"]) / jaw_fall
    else:
        jaw_val_conf = 0.0

    # Teeth gate: mulut terbuka + gigi kelihatan = bukan bingung, lebih ke bicara/senyum
    jaw_val_conf *= teeth_gate
    jaw_co = _clamp(jaw_val_conf - smile_pen * 1.5, 0, 1)

    pucker_co  = _clamp(g("mouthPucker") / max(ccfg["pucker_th"], 1e-6), 0, 1)

    # Ketegangan bibir atas dengan rahang tertutup: bibir menegang saat fokus/konsentrasi berat.
    # "mouthUpperUp" tinggi + jawOpen rendah + tanpa senyum = lippress saat bingung/konsentrasi,
    # bukan ekspresi bicara atau tertawa. Hanya aktif kalau jaw nyaris tertutup.
    jaw_closed_gate = _clamp(1.0 - jo / max(ccfg.get("jaw_closed_th", 0.10), 1e-6), 0, 1)
    mu_avg    = (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2
    mu_conf_v = _clamp((mu_avg - ccfg.get("mu_conf_th", 0.40)) / max(ccfg.get("mu_conf_range", 0.30), 1e-6), 0, 1) * jaw_closed_gate

    # Craig et al. (2008): AU4 (brow lowerer) + AU7 (lid tightener) co-occurrence = 73% confusion coverage.
    # eyeSquint = lid tightener (AU7). When both browDown (AU4) and eyeSquint (AU7) fire together,
    # it is a strong empirically-validated confusion indicator.
    au7_th = ccfg.get("au7_th", 0.15)
    au7_v  = _clamp(squint_avg / max(au7_th, 1e-6), 0, 1)
    au4_au7_co = (brow_dn_v * au7_v) ** 0.5   # geometric mean — requires BOTH active
    au4_au7_co_w = ccfg.get("au4_au7_co_w", 0.50)
    au4_au7_sig = au4_au7_co * au4_au7_co_w

    sig_brow_conf = max(brow_dn_v, brow_in_v, au4_au7_sig)
    sig_mata_conf = max(iris_up_v, look_up_v, look_dn_v)  # tatap ke atas ATAU bawah = ciri confusion

    # Confusion membutuhkan CO-OCCURRENCE minimal 2 sinyal.
    # max() tunggal terlalu permisif — roll sedikit SAJA atau jaw sedikit SAJA
    # seharusnya tidak cukup. Rata-rata 2 sinyal terkuat mensyaratkan
    # setidaknya 2 cue hadir bersamaan untuk skor tinggi.
    _conf_signals = sorted([sig_brow_conf, sig_mata_conf, jaw_co, pucker_co, roll_v, squint_conf_v, mu_conf_v], reverse=True)
    base_conf = _conf_signals[0] * 0.6 + _conf_signals[1] * 0.4
    conf = _clamp(base_conf * ccfg["blend_a"] + (sig_mata_conf + sig_brow_conf) * ccfg["blend_b"], 0, 1)

    # Gaze gate: confusion masih membutuhkan attention ke konten (definisi semantik).
    # Pakai gaze_dev_eng (tanpa komponen ke bawah) — lihat keyboard/kertas = berpikir = BUKAN distraksi.
    # Sama dengan engagement: nunduk tidak mengurangi gate. Hanya gaze ke samping/atas yang mengurangi.
    attentive_dead  = ccfg.get("attentive_dead",  8.0)
    attentive_range = ccfg.get("attentive_range", 20.0)
    attentive_floor = ccfg.get("attentive_floor", 0.3)
    attentive_gate  = _clamp(1.0 - max(0.0, gaze_dev_eng - attentive_dead) / max(attentive_range, 1e-6), attentive_floor, 1.0)
    conf = _clamp(conf * attentive_gate, 0, 1)

    # Smile gate: senyum dan bingung adalah mutual exclusive secara semantik.
    # Senyum sedang (>= smile_conf_gate_th) → conf turun proporsional ke 0.
    smile_gate_th = ccfg.get("smile_conf_gate_th", 0.20)
    smile_gate    = _clamp(1.0 - smile_raw / max(smile_gate_th, 1e-6), 0.0, 1.0)
    conf = _clamp(conf * smile_gate, 0, 1)

    # Hand suppression: kehadiran tangan dekat kepala (zona manapun) = cue Frustrasi, BUKAN Confusion.
    # Tangan terdeteksi → confusion ditekan additive supaya tidak fire bersama Frustration.
    hand_near_head = max(r.hand_forehead, r.hand_chin)
    conf = _clamp(conf - hand_near_head, 0, 1)

    # look_dn boosts confusion ONLY when other confusion signals are present.
    # Spec Rule 3: looking down + typing/reading = engagement, NOT confusion.
    # Tanpa gate ini, semua siswa nunduk baca otomatis kena confusion floor.
    look_dn_boost = ccfg.get("look_dn_boost", 0.15)
    if base_conf > 0.15:
        conf = max(conf, look_dn_v * look_dn_boost * _clamp(base_conf / 0.30, 0, 1))

    # Boredom dan confusion saling eksklusif: sangat bosan = sudah "checked out",
    # tidak sedang aktif bingung memproses konten. Ketika boredom tinggi, sinyal confusion
    # seperti browInnerUp dan lookDn sering merupakan artefak ekspresi bosan/mengantuk,
    # bukan kebingungan aktif. Ditaruh SETELAH look_dn_boost floor.
    bore_conf_suppress_val = ccfg.get("bore_conf_suppress_bore", 0.40)
    conf = _clamp(conf - bore * bore_conf_suppress_val, 0, 1)

    # == 3: FRUSTRATION ========================================================
    # Craig et al. (2008): AU1 (outer brow raise) + AU2 (inner brow raise) = PRIMARY frustration signals,
    # present in 100% of frustration episodes and mutually trigger each other.
    bou_fr = _clamp(
        (g("browOuterUpLeft") + g("browOuterUpRight")) / 2 / max(fcfg.get("brow_outer_up_th", 0.20), 1e-6), 0, 1
    )  # AU1 outer brow raise
    biu_fr = _clamp(
        g("browInnerUp") / max(fcfg.get("brow_inner_up_th", 0.20), 1e-6), 0, 1
    )  # AU2 inner brow raise
    # AU1+AU2 co-occurrence (geometric mean): fires strongly only when both brow raises are active simultaneously
    brow_raise_co = (bou_fr * biu_fr) ** 0.5

    # Secondary/supplementary signals (legacy signals — not in Craig2008 Table 2 primary findings)
    br_fr = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / max(fcfg["brow_dn_th"], 1e-6), 0, 1)
    ns_fr = _clamp(max(g("noseSneerLeft"), g("noseSneerRight")) / max(fcfg["nose_sneer_th"], 1e-6), 0, 1)
    ck_fr = _clamp((g("cheekSquintLeft") + g("cheekSquintRight")) / 2 / max(fcfg["cheek_squint_th"], 1e-6), 0, 1)
    lp_fr = _clamp((g("mouthPressLeft") + g("mouthPressRight")) / 2 / max(fcfg["mouth_press_th"], 1e-6), 0, 1)
    ey_fr = _clamp((g("eyeSquintLeft") + g("eyeSquintRight")) / 2 / max(fcfg["eye_squint_th"], 1e-6), 0, 1)

    jaw_val_frus = max(0.0, jo - fcfg["jaw_start"])
    jw_fr = _clamp((jaw_val_frus - smile_pen * 1.5) / max(fcfg["jaw_range"], 1e-6), 0, 1)
    # Sudut mulut turun (frown) = ekspresi kecewa/frustasi yang sering terlewat
    mf_fr = _clamp((g("mouthFrownLeft") + g("mouthFrownRight")) / 2 / max(fcfg.get("mouth_frown_th", 0.25), 1e-6), 0, 1)

    # Tangan adalah sinyal primer frustasi; ekspresi wajah hanya suplemen.
    # Craig2008 primary: brow_raise_co (AU1+AU2). Single raises at 0.70 scale. Legacy signals secondary.
    face_secondary = max(ns_fr, br_fr, lp_fr, ey_fr, ck_fr, mf_fr)
    face_peak = max(
        brow_raise_co,          # AU1+AU2 co-occurrence (Craig2008 primary, 100% coverage)
        bou_fr * 0.70,          # AU1 alone — partial signal
        biu_fr * 0.70,          # AU2 alone — partial signal
        face_secondary,         # legacy supplementary signals
    )
    face_w    = fcfg.get("face_weight", 0.45)
    sig_wajah_frus = _clamp(face_peak * face_w - smile_pen * 1.5, 0, 1)

    # Weighted-max fusion: tangan adalah cue penting tapi bukan single-cue dominant.
    # max(weighted_avg, hand_alone, face_alone): single cue kuat tetap valid, kombinasi
    # tidak saturasi aditif, signal lemah tunggal tidak meledakkan score.
    # Tangan di zona MANAPUN dekat kepala = cue Frustrasi (dahi, mata, pipi, dagu, menutup wajah).
    hand_trigger_frus = max(r.hand_forehead, r.hand_chin)
    hand_w = fcfg.get("hand_weight", 0.65)
    base_frus = _clamp(
        max(
            hand_trigger_frus * hand_w + sig_wajah_frus * (1.0 - hand_w),
            hand_trigger_frus,
            sig_wajah_frus,
        ),
        0, 1,
    )
    frus = _clamp(base_frus * fcfg["blend_a"] + (ck_fr + jw_fr) * fcfg["blend_b"], 0, 1)

    # Frustration suppression pada Confusion: kedua emosi tidak boleh tinggi bersamaan.
    # Sinyal alis (browDown) dipakai di keduanya — ketika frustrasi tinggi, itu alis tegang
    # bukan alis bingung. Terapkan setelah frus selesai dihitung.
    frus_conf_suppress = ccfg.get("frus_conf_suppress", 0.5)
    conf = _clamp(conf - frus * frus_conf_suppress, 0, 1)

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
              f"lookDn={look_down_v:.2f} teeth={teeth_signal:.2f}(gate={teeth_gate:.2f}) | gH={gaze_h:+.1f}° gVup={gaze_v_up:.1f}° gVdn={gaze_v_down:.1f}° rollG={roll_gaze_eff:.1f}° devBore={gaze_dev_bore:.1f}° devEng={gaze_dev:.1f}° | "
              f"rollGate={roll_gate:.2f} yawGate={yaw_gate:.2f} gate={gate:.2f} | "
              f"boreGaze={bore_gaze:.2f} | "
              f"B={bore:.3f} E={eng:.3f} C={conf:.3f} F={frus:.3f}")
        if conf > 0.5:
            print(f"  [CONF] brow_dn(AU4)={brow_dn_v:.2f} au7={au7_v:.2f} au4_au7_co={au4_au7_co:.2f} "
                  f"brow_in={brow_in_v:.2f} co={co_signal:.2f} "
                  f"iris_up={iris_up_v:.2f} look_up={look_up_v:.2f} roll={roll_v:.2f} "
                  f"jaw={jaw_co:.2f} pucker={pucker_co:.2f} "
                  f"base={base_conf:.2f}")
        if frus > 0.5:
            print(f"  [FRUS] bou(AU1)={bou_fr:.2f} biu(AU2)={biu_fr:.2f} brow_raise_co={brow_raise_co:.2f} "
                  f"ns={ns_fr:.2f} br={br_fr:.2f} ck={ck_fr:.2f} face_peak={face_peak:.2f} "
                  f"hand={max(r.hand_forehead, r.hand_chin):.2f}")

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

