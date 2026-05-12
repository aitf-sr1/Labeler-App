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
_landmarker      = None

_HAND_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_NAME = "hand_landmarker.task"
_hand_landmarker = None

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


# ── Lazy-load FaceLandmarker ─────────────────────────────────────────────────
_face_lock = threading.Lock()
def _get_landmarker():
    global _landmarker
    with _face_lock:
        if _landmarker is None:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "siglip_labeler")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, _LANDMARKER_NAME)

            if not os.path.exists(model_path):
                print(f"Mengunduh FaceLandmarker model ke {model_path}…")
                urllib.request.urlretrieve(_LANDMARKER_URL, model_path)

            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.40,
                min_face_presence_confidence=0.40,
            )
            _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


# ── Lazy-load HandLandmarker ─────────────────────────────────────────────────
_hand_lock = threading.Lock()
def _get_hand_landmarker():
    global _hand_landmarker
    with _hand_lock:
        if _hand_landmarker is None:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "siglip_labeler")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, _HAND_NAME)

            if not os.path.exists(model_path):
                print(f"Mengunduh HandLandmarker model ke {model_path}…")
                urllib.request.urlretrieve(_HAND_URL, model_path)

            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=2,
                min_hand_detection_confidence=0.20,
                min_hand_presence_confidence=0.20,
            )
            _hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
    return _hand_landmarker


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
        print(f"  [HAND] Exception saat detect: {e}")
        return 0.0, 0.0, 0.0, [], []

    if not res.hand_landmarks:
        print(f"  [HAND] Tidak ada tangan terdeteksi (res.hand_landmarks kosong)")
        return 0.0, 0.0, 0.0, [], []

    # Kumpulkan semua titik landmark dari semua tangan yang terdeteksi
    all_pts = []
    for hand_lms in res.hand_landmarks:
        for lm in hand_lms:
            all_pts.append((lm.x, lm.y))

    if not all_pts:
        return 0.0, 0.0, 0.0, [], []

    print(f"  [HAND] Terdeteksi {len(res.hand_landmarks)} tangan, total {len(all_pts)} titik")
    
    # ZONASI TANGAN V3
    n = len(all_pts)
    pts_top = sum(1 for _, y in all_pts if -0.20 <= y < 0.25)
    pts_mid = sum(1 for _, y in all_pts if  0.25 <= y < 0.55)
    pts_bot = sum(1 for _, y in all_pts if  0.55 <= y <= 1.20)
    centered = sum(1 for x, _ in all_pts if 0.05 <= x <= 0.95)
    print(f"  [HAND] centered={centered}, pts_top={pts_top}, pts_mid={pts_mid}, pts_bot={pts_bot}")
    
    if centered < 5:
        print(f"  [HAND] DIFILTER: centered({centered}) < 5, skor di-reset 0")
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
        print(f"  [HAND-FULL] Exception: {e}")
        return 0.0, 0.0, [], []

    if not res.hand_landmarks:
        print(f"  [HAND-FULL] Tidak ada tangan di full frame")
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
            hand_forehead=hand_mid_bot,             # Frustration trigger
            hand_chin=hand_top,                     # Confusion trigger
            hand_landmarks_px=hand_pts_px,
            hand_landmarks_raw=hand_raw,
        )

    lms = res.face_landmarks[0]
    bs  = {b.category_name: round(b.score, 4) for b in res.face_blendshapes[0]}
    yaw, pitch, _ = _rotation_matrix_to_euler(res.facial_transformation_matrixes[0])

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
        yaw=round(yaw, 2), pitch=round(pitch, 2),
        iris_x=round(iris_x, 4), iris_y=round(iris_y, 4),
        iris_img_x=round(iris_img_x, 4), iris_img_y=round(iris_img_y, 4),
        left_iris_px=(lxi, lyi), right_iris_px=(rxi, ryi),
        blendshapes=bs, face_found=True,
        face_landmarks=lms,
        hand_forehead=round(hand_mid_bot, 4),
        hand_chin=round(hand_top, 4),
        hand_landmarks_px=hand_pts_px,
        hand_landmarks_raw=hand_raw,
    )


# ── Emotion scoring ───────────────────────────────────────────────────────────
_DBG_LAND = True   # set False untuk matikan debug log

def compute_emotion_scores(r: LandmarkResult) -> dict:
    """
    Hitung skor landmark 0.0-1.0 per emosi.

    PRINSIP GAZE:
      Satu metrik unified `gaze_dev` (derajat 2D dari arah kamera) untuk Boredom & Engagement.
      - Tatapan ke mana saja (kiri/kanan/atas/bawah) → gaze_dev tinggi → Bosan, Tidak Engaged.
      - Tatapan lurus ke kamera → gaze_dev rendah → Tidak Bosan, Engaged.
      Keduanya inversely related — tidak bisa tinggi bersamaan karena pakai metrik yang sama.

    Komponen gaze_dev:
      gaze_h = yaw + iris_x × 30  (direction-aware: iris berlawanan yaw = kompensasi ke kamera)
      gaze_v = max(|-pitch + iris_y × 25|, eyeLookDown × 40)  (fallback ke bentuk kelopak)
      gaze_dev = sqrt(gaze_h² + gaze_v²)
    """
    if not r.face_found:
        return {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}

    g = lambda k: r.blendshapes.get(k, 0.0)

    # ── Gaze deviation (shared basis) ─────────────────────────────────────────
    # Horizontal: yaw kepala + iris_x (direction-aware).
    # iris berlawanan arah yaw = kompensasi ke kamera → deviasi lebih kecil.
    GAZE_SCALE   = 35   # iris_x=0.4 → 14°; lebih sensitif dari 30 agar sideways iris terdeteksi
    GAZE_SCALE_V = 25   # vertikal: mata lebih terbatas gerak ke atas/bawah
    gaze_h = r.yaw + r.iris_x * GAZE_SCALE          # signed, +=kanan kamera

    # Vertikal: pitch kepala + iris_y.
    # pitch<0=kepala nunduk → -pitch>0; iris_y>0=pupil bawah → keduanya compound.
    gaze_v_raw = -r.pitch + r.iris_y * GAZE_SCALE_V  # signed, +=bawah kamera

    # Fallback: iris bisa gagal terdeteksi saat lihat ke bawah (tertutup kelopak).
    # eyeLookDown dihitung dari BENTUK KELOPAK ("U") — lebih robust dari posisi pupil.
    look_down_v = (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2
    gaze_v     = max(abs(gaze_v_raw), look_down_v * 40)  # vertikal magnitude (°)
    # Dead zone 15° untuk vertikal: layar biasanya sedikit di bawah mata (wajar).
    # Mata sipit pun bisa trigger eyeLookDown — dead zone meredam kedua kasus ini.
    gaze_v_eff = max(0.0, gaze_v - 15.0)

    # Floor 1 — iris magnitude: iris jelas ke samping tetap berkontribusi meski kepala kompensasi.
    # Faktor 2.0 (bukan 0.9): iris_x=0.2 → 14°, iris_x=0.3 → 21° → gate turun signifikan.
    # Diperlukan karena iris_x=0.2 secara visual sudah terlihat "miring" di video.
    iris_side = abs(r.iris_x) * GAZE_SCALE * 2.0
    # Floor 2 — yaw kepala: mukanya miring = gaze_dev minimal sebesar sudut yaw.
    gaze_h_eff = max(abs(gaze_h), iris_side, abs(r.yaw))

    # Total deviasi angular 2D dari arah kamera (derajat).
    gaze_dev = (gaze_h_eff ** 2 + gaze_v_eff ** 2) ** 0.5

    # == 0: BOREDOM ============================================================
    # Tatapan ke mana saja → bore_gaze naik. Dead zone 5°, penuh di 25°.
    bore_gaze  = _clamp((gaze_dev - 5) / 20, 0, 1)

    # Ekspresi bosan (pendukung): mata berat/tutup, menguap, kepala mendongak.
    # Dead zone 0.20 untuk mata sipit alami — baru mulai naik di atas itu.
    blink_v    = _clamp((max(g("eyeBlinkLeft"), g("eyeBlinkRight")) - 0.20) / 0.50, 0, 1)
    yawn_v     = _clamp(g("jawOpen") / 0.35, 0, 1) if r.pitch < 15 else 0.0
    pitch_up_v = _clamp((r.pitch - 20) / 25, 0, 1)
    sig_expr   = max(blink_v, yawn_v, pitch_up_v) * 0.70

    base_bore  = max(bore_gaze, sig_expr)
    bore       = _clamp(base_bore * 0.85 + (bore_gaze + sig_expr) * 0.15, 0, 1)

    # == 1: ENGAGEMENT =========================================================
    # Gate berbasis gaze_dev yang SAMA — inversely related dengan Boredom.
    # Tatapan ke mana saja (horizontal ATAU vertikal) → gate turun → Tidak Engaged.
    # Dead zone 5°, nol di 17° (range 12° — lebih ketat dari sebelumnya).
    gate = _clamp(1 - max(0, gaze_dev - 5) / 12, 0, 1)
    # eye_op: hanya droopy parah (blink > 0.50) yang kurangi engagement.
    # Mata sipit alami (blink < 0.50) tidak dipenalti agar tidak false-negative.
    blink_heavy = max(0.0, max(g("eyeBlinkLeft"), g("eyeBlinkRight")) - 0.50) / 0.50
    eng = gate * max(0.30, 1.0 - blink_heavy)

    # == 2: CONFUSION -- ekspresi bingung ========================================
    # iris_y negatif = pupil ke atas. Dead zone 0.15 (dikurangi dari 0.25 yg terlalu ketat).
    iris_up_v  = _clamp((-r.iris_y - 0.15) / 0.30, 0, 1)
    # lookUp threshold 0.35 (dikurangi dari 0.45 yg terlalu ketat).
    look_up_v  = _clamp(max(g("eyeLookUpLeft"), g("eyeLookUpRight")) / 0.35, 0, 1)
    # pitch: mulai dari 5° (dikurangi dari 10°), penuh di 20°.
    pitch_cu   = _clamp((r.pitch - 5) / 15, 0, 1)
    # browDown: threshold 0.23 (kompromi antara 0.15 yg terlalu sensitif dan 0.30 yg terlalu ketat).
    brow_dn_v  = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / 0.23, 0, 1)
    # browInnerUp: HANYA berkontribusi jika ada sinyal lain bersamaan.
    # Gate diturunkan 0.30 → 0.25 agar lebih mudah diaktifkan ketika iris/mata memberi sinyal.
    brow_in_raw = g("browInnerUp")
    co_signal   = max(iris_up_v, look_up_v, pitch_cu)  # sinyal penguat (iris/mata/kepala)
    brow_in_v   = _clamp(brow_in_raw / 0.30, 0, 1) * _clamp(co_signal / 0.25, 0, 1)

    # "Mangap" sedikit karena bingung (bukan karena tertawa/senyum)
    smile_raw  = max(g("mouthSmileLeft"), g("mouthSmileRight"))
    smile_pen  = max(0.0, smile_raw - 0.15)

    # Confusion: Mulut terbuka sedikit (0.05–0.25 puncak).
    # Jika terlalu lebar (>0.40), itu menguap/berteriak (bukan bingung).
    jo = g("jawOpen")
    if jo <= 0.05:
        jaw_val_conf = 0.0
    elif jo <= 0.25:
        jaw_val_conf = (jo - 0.05) / 0.20
    elif jo <= 0.40:
        jaw_val_conf = 1.0 - (jo - 0.25) / 0.15
    else:
        jaw_val_conf = 0.0

    jaw_co = _clamp(jaw_val_conf - smile_pen * 1.5, 0, 1)

    # pucker: threshold 0.30 (dikurangi dari 0.40 yg terlalu ketat).
    pucker_co  = _clamp(g("mouthPucker") / 0.30, 0, 1)

    sig_brow_conf = max(brow_dn_v, brow_in_v)
    sig_mata_conf = max(iris_up_v, look_up_v)

    base_conf = max(sig_brow_conf, sig_mata_conf, jaw_co, pucker_co, r.hand_chin)
    conf = _clamp(base_conf * 0.85 + (pitch_cu + sig_brow_conf) * 0.15, 0, 1)
    
    # MUTLAK: Tangan yang menutupi area TENGAH WAJAH (mata/hidung) membatalkan Confusion
    # Tapi jika tangannya dominan menggaruk kepala (r.hand_chin), jangan dibatalkan!
    suppression = r.hand_forehead if r.hand_chin < 0.5 else 0.0
    conf = _clamp(conf - suppression, 0, 1)

    # == 3: FRUSTRATION -- ekspresi tegang (Soft OR logic) ===========================
    # Threshold dinaikkan drastis agar orang yang sedang rileks/ngelamun (mulut nutup biasa,
    # mata sedikit sayu) tidak memicu Frustration secara tidak sengaja.
    # Threshold dinaikkan SANGAT TINGGI agar "mikir/bingung" (Confusion) yang 
    # biasanya juga mengerutkan alis/bibir tidak bocor ke Frustration.
    # Frustration murni butuh ekspresi ekstrem (marah/sangat stres/menangis).
    br_fr = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / 0.40, 0, 1)
    ns_fr = _clamp(max(g("noseSneerLeft"), g("noseSneerRight")) / 0.20, 0, 1)
    ck_fr = _clamp((g("cheekSquintLeft") + g("cheekSquintRight")) / 2 / 0.40, 0, 1)
    lp_fr = _clamp((g("mouthPressLeft") + g("mouthPressRight")) / 2 / 0.40, 0, 1)
    ey_fr = _clamp((g("eyeSquintLeft") + g("eyeSquintRight")) / 2 / 0.40, 0, 1)
    
    # Rahang tegang/berteriak (bisa terbuka lebar, abaikan mulut terbuka sedikit)
    jaw_val_frus = max(0.0, jo - 0.10)
    jw_fr = _clamp((jaw_val_frus - smile_pen * 1.5) / 0.20, 0, 1)

    # SUM LOGIC: Frustration butuh lebih dari 1 otot tegang (kombinasi).
    # Jika hanya 1 otot (misal browDown karena bingung), skornya tidak akan tembus 1.0.
    # Namun noseSneer (mengernyit jijik/marah) sangat kuat, jadi nilainya mutlak.
    sig_wajah_frus = _clamp(ns_fr + (br_fr + lp_fr + ey_fr + ck_fr) / 2.0, 0, 1)
    
    # Kurangi skor Frustration jika orang tersebut sedang tersenyum lebar
    sig_wajah_frus = _clamp(sig_wajah_frus - smile_pen * 1.5, 0, 1)
    
    # Tangan Frustration (hand_mid_bot yang disimpan di hand_forehead)
    # Jika tangan menutupi wajah bawah/tengah, Frustration langsung naik drastis.
    hand_trigger_frus = r.hand_forehead
    base_frus = _clamp(sig_wajah_frus + hand_trigger_frus, 0, 1)
    frus = _clamp(base_frus * 0.85 + (ck_fr + jw_fr) * 0.15, 0, 1)

    # Debug log
    if _DBG_LAND:
        print(f"  [LAND] yaw={r.yaw:+.1f} iris_x={r.iris_x:+.3f} iris_y={r.iris_y:+.3f} "
              f"lookDn={look_down_v:.2f} | gH={gaze_h:+.1f}° gV={gaze_v:.1f}° dev={gaze_dev:.1f}° | "
              f"boreGaze={bore_gaze:.2f} gate={gate:.2f} | "
              f"B={bore:.3f} E={eng:.3f} C={conf:.3f} F={frus:.3f}")
        if conf > 0.5:
            print(f"  [CONF] brow_dn={brow_dn_v:.2f} brow_in={brow_in_v:.2f} "
                  f"iris_up={iris_up_v:.2f} look_up={look_up_v:.2f} "
                  f"jaw={jaw_co:.2f} pucker={pucker_co:.2f} pitch_cu={pitch_cu:.2f} "
                  f"base={base_conf:.2f}")

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

