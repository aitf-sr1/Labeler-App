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
    # Sinyal tangan
    hand_forehead:  float = 0.0  # tangan di zona dahi/mata (y 0–45%) → Frustration
    hand_chin:      float = 0.0  # tangan di zona pipi/dagu (y 40–80%) → Confusion
    hand_landmarks_px: list = field(default_factory=list)  # pixel positions untuk viz


# ── Lazy-load FaceLandmarker ─────────────────────────────────────────────────
def _get_landmarker():
    global _landmarker
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
def _get_hand_landmarker():
    global _hand_landmarker
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
            min_hand_detection_confidence=0.30,
            min_hand_presence_confidence=0.30,
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
    except Exception:
        return 0.0, 0.0, []

    if not res.hand_landmarks:
        return 0.0, 0.0, []

    # Kumpulkan semua titik landmark dari semua tangan yang terdeteksi
    all_pts = []
    for hand_lms in res.hand_landmarks:
        for lm in hand_lms:
            all_pts.append((lm.x, lm.y))

    if not all_pts:
        return 0.0, 0.0, []

    # User meminta: SEMUA posisi tangan di layar (atas kepala, dahi, mata, pipi, mulut, dagu) 
    # langsung masuk ke Frustration. Tidak perlu lagi dipisah berdasarkan zona atas/bawah.
    # Selama berada di frame (dan relatif di tengah), itu dihitung sebagai sentuhan wajah/kepala.
    n = len(all_pts)
    
    # Hitung semua titik yang ada di layar (0.00 <= y <= 1.00)
    face_pts = sum(1 for _, y in all_pts if -0.20 <= y <= 1.20)  # Toleransi keluar frame sedikit
    
    # Hanya count jika tangan tidak terlalu di pinggir banget
    centered     = sum(1 for x, _ in all_pts if 0.05 <= x <= 0.95)
    center_ratio = centered / n
    
    hand_on_face = _clamp((face_pts / n) * center_ratio, 0, 1)
    
    hand_pts_px   = [(int(x * w), int(y * h)) for x, y in all_pts]

    # Return hand_on_face ke slot hand_forehead (agar kompatibel dengan kode yang ada)
    return hand_on_face, 0.0, hand_pts_px


def analyze_frame(frame_bgr) -> LandmarkResult:
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

    # Deteksi tangan (selalu dijalankan, terlepas dari wajah ditemukan atau tidak)
    hand_forehead, hand_chin, hand_pts_px = _analyze_hands(mp_img, h, w)

    if not res.face_landmarks:
        return LandmarkResult(
            face_found=False,
            hand_forehead=hand_forehead,
            hand_chin=hand_chin,
            hand_landmarks_px=hand_pts_px,
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
        hand_forehead=round(hand_forehead, 4),
        hand_chin=round(hand_chin, 4),
        hand_landmarks_px=hand_pts_px,
    )


# ── Emotion scoring ───────────────────────────────────────────────────────────
_DBG_LAND = True   # set False untuk matikan debug log

def compute_emotion_scores(r: LandmarkResult) -> dict:
    """
    Hitung skor landmark 0.0-1.0 per emosi.

    PRINSIP UTAMA:
      - Face crop SELALU center pada wajah -> SigLIP tidak bisa bedakan arah.
      - iris_img_x TIDAK DIPAKAI (cacat karena crop mengikuti wajah).
      - HANYA yaw (rotasi 3D) dan iris_x (posisi pupil dalam mata) yang reliable.
      - Boredom: MAX(kepala noleh, mata lirik) -- satu saja cukup.
      - Engagement: kepala lurus DAN mata lurus DAN mata buka -- SEMUA harus terpenuhi.
    """
    if not r.face_found:
        return {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}

    g = lambda k: r.blendshapes.get(k, 0.0)
    ya = abs(r.yaw)
    ix = abs(r.iris_x)

    # == 0: BOREDOM -- MAX(noleh, lirik) agar SATU sinyal saja cukup ==========
    # A) Kepala noleh (yaw >=8° mulai naik, >=18° = penuh)
    #    Dinaikkan agar gerakan kepala natural/membaca layar tidak dihitung noleh.
    sig_yaw    = _clamp((ya - 8) / 10, 0, 1)

    # B) Mata lirik ke samping (iris_x >=0.12 mulai, >=0.35 = penuh)
    sig_iris   = _clamp((ix - 0.12) / 0.23, 0, 1)

    # Sinyal ARAH: ambil MAX -- kepala noleh ATAU mata lirik = BORED
    sig_arah   = max(sig_yaw, sig_iris)

    # C) Ekspresi bosan (pendukung): mata berat/tutup, menguap, kepala mendongak
    # Menghapus 'eye_low_v' karena melihat ke bawah (ke keyboard/layar) adalah wajar.
    blink_v    = _clamp(max(g("eyeBlinkLeft"), g("eyeBlinkRight")) / 0.4, 0, 1)
    yawn_v     = _clamp(g("jawOpen") / 0.35, 0, 1) if r.pitch < 8 else 0.0
    pitch_up_v = _clamp((r.pitch - 20) / 25, 0, 1)

    sig_expr   = max(blink_v, yawn_v, pitch_up_v) * 0.5

    # Final: Soft OR logic. 
    # (Tangan dihapus dari Boredom karena user meminta semua sentuhan wajah masuk ke Frustration)
    base_bore = max(sig_arah, sig_expr)
    bore = _clamp(base_bore * 0.85 + (sig_arah + sig_expr) * 0.15, 0, 1)

    # == 1: ENGAGEMENT -- semua gate harus ON (AND logic) ======================
    # Gate A: kepala lurus — DEAD ZONE: yaw ≤3° = sempurna (1.0), lalu turun, 0 di ≥10°
    gate_yaw   = _clamp(1 - max(0, ya - 3) / 7, 0, 1)

    # Gate B: mata lurus — DEAD ZONE: iris ≤0.08 = sempurna (1.0), lalu turun, 0 di ≥0.25
    gate_iris  = _clamp(1 - max(0, ix - 0.08) / 0.17, 0, 1)

    # Gate gabungan — keduanya harus non-zero
    gate       = gate_yaw * gate_iris

    # Kualitas engagement
    if -25 <= r.pitch <= 15:
        p_ok = 1.0
    elif r.pitch > 15:
        p_ok = _clamp(1 - (r.pitch - 15) / 20, 0, 1)
    else:
        p_ok = _clamp(1 - (-r.pitch - 25) / 15, 0, 1)
    eye_op = 1 - max(g("eyeBlinkLeft"), g("eyeBlinkRight"))

    eng = gate * (0.60 * p_ok + 0.40 * eye_op)

    # == 2: CONFUSION -- ekspresi bingung (Soft OR logic agar lebih sensitif) ========
    iris_up_v  = _clamp((-r.iris_y - 0.15) / 0.35, 0, 1)
    look_up_v  = _clamp(max(g("eyeLookUpLeft"), g("eyeLookUpRight")) / 0.3, 0, 1)
    pitch_cu   = _clamp((r.pitch - 8) / 17, 0, 1)
    # Sensitivitas diturunkan sedikit agar tidak bocor (0.12 -> 0.15)
    brow_dn_v  = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / 0.15, 0, 1)
    brow_in_v  = _clamp(g("browInnerUp") / 0.15, 0, 1)
    
    # "Mangap" sedikit karena bingung (bukan karena tertawa/senyum)
    smile_raw  = max(g("mouthSmileLeft"), g("mouthSmileRight"))
    # Dead zone: abaikan senyum tipis (<0.15) karena sering muncul saat meringis/bingung
    smile_pen  = max(0.0, smile_raw - 0.15)
    
    # Confusion: Mulut terbuka sedikit (0.05 - 0.20). 
    # Jika terlalu lebar (>0.35), itu menguap/berteriak (bukan bingung).
    jo = g("jawOpen")
    if jo <= 0.05:
        jaw_val_conf = 0.0
    elif jo <= 0.20:
        jaw_val_conf = (jo - 0.05) / 0.15
    elif jo <= 0.35:
        jaw_val_conf = 1.0 - (jo - 0.20) / 0.15
    else:
        jaw_val_conf = 0.0
        
    # Penalti dikali 1.5 agar senyum asli benar-benar membatalkan Confusion
    jaw_co     = _clamp(jaw_val_conf - smile_pen * 1.5, 0, 1)
    
    # "Cemburut" / mengerucutkan bibir saat berpikir/bingung
    pucker_co  = _clamp(g("mouthPucker") / 0.20, 0, 1)

    sig_brow_conf = max(brow_dn_v, brow_in_v)
    sig_mata_conf = max(iris_up_v, look_up_v)
    
    # Jika ADA SALAH SATU ciri yang kuat, skor dasar tinggi
    base_conf = max(sig_brow_conf, sig_mata_conf, jaw_co, pucker_co)
    conf = _clamp(base_conf * 0.85 + (pitch_cu + sig_brow_conf) * 0.15, 0, 1)
    
    # MUTLAK: Tangan di wajah membatalkan Confusion (mencegah false positive dari occlusion).
    # Sesuai permintaan user, tangan di wajah murni milik Frustration.
    conf = _clamp(conf - r.hand_forehead, 0, 1)

    # == 3: FRUSTRATION -- ekspresi tegang (Soft OR logic) ===========================
    br_fr = _clamp((g("browDownLeft") + g("browDownRight")) / 2 / 0.15, 0, 1)
    ns_fr = _clamp(max(g("noseSneerLeft"), g("noseSneerRight")) / 0.15, 0, 1)
    ck_fr = _clamp((g("cheekSquintLeft") + g("cheekSquintRight")) / 2 / 0.15, 0, 1)
    lp_fr = _clamp((g("mouthPressLeft") + g("mouthPressRight")) / 2 / 0.15, 0, 1)
    ey_fr = _clamp((g("eyeSquintLeft") + g("eyeSquintRight")) / 2 / 0.15, 0, 1)
    
    # Rahang tegang/berteriak (bisa terbuka lebar, abaikan mulut terbuka sedikit)
    jaw_val_frus = max(0.0, jo - 0.10)
    jw_fr = _clamp((jaw_val_frus - smile_pen * 1.5) / 0.20, 0, 1)

    sig_wajah_frus = max(br_fr, ns_fr, lp_fr, ey_fr, ck_fr)
    # Kurangi skor Frustration jika orang tersebut sedang tersenyum lebar (menghindari false positive dari cheekSquint/eyeSquint saat tertawa)
    sig_wajah_frus = _clamp(sig_wajah_frus - smile_pen * 1.5, 0, 1)
    
    # Jika ada tangan di wajah (hand_forehead sekarang merepresentasikan ALL hand_on_face),
    # ekspresi wajah mungkin tidak terbaca. User menganggap segala sentuhan wajah
    # akibat pusing/stres sebagai Frustration.
    hand_trigger_frus = r.hand_forehead ** 2
    base_frus = max(sig_wajah_frus, hand_trigger_frus)
    frus = _clamp(base_frus * 0.85 + (ck_fr + jw_fr) * 0.15, 0, 1)

    # Debug log
    if _DBG_LAND:
        print(f"  [LAND] yaw={r.yaw:+.1f} iris_x={r.iris_x:+.3f} | "
              f"sig_yaw={sig_yaw:.2f} sig_iris={sig_iris:.2f} arah={sig_arah:.2f} | "
              f"gate_yaw={gate_yaw:.2f} gate_iris={gate_iris:.2f} gate={gate:.2f} | "
              f"B={bore:.3f} E={eng:.3f} C={conf:.3f} F={frus:.3f}")

    return {
        0: round(bore, 4),
        1: round(eng,  4),
        2: round(conf, 4),
        3: round(frus, 4),
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def draw_landmark_viz(frame_bgr: np.ndarray, r: LandmarkResult,
                      emotion_scores: dict | None = None) -> np.ndarray:
    """
    Gambar overlay landmark pada frame BGR (512×512 crop wajah):
      - Lingkaran iris kiri & kanan (kuning)
      - Panah arah pandangan dari center frame (biru)
      - Teks: yaw, pitch, iris offset
      - Teks: skor emosi landmark (opsional)

    Returns BGR image dengan overlay.
    """
    viz = frame_bgr.copy()
    h, w = viz.shape[:2]

    if not r.face_found:
        cv2.putText(viz, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return viz

    # Lingkaran iris
    for px, py in (r.left_iris_px, r.right_iris_px):
        if 0 < px < w and 0 < py < h:
            cv2.circle(viz, (px, py), 6, (0, 220, 255), 2)

    # Titik tangan (hijau tua = chin zone, oranye = forehead zone)
    for hx, hy in r.hand_landmarks_px:
        if 0 <= hx < w and 0 <= hy < h:
            zone_color = (0, 200, 80) if hy > h * 0.40 else (0, 140, 255)
            cv2.circle(viz, (hx, hy), 3, zone_color, -1)

    # Garis batas zona (dahi/pipi)
    zone_y = int(h * 0.45)
    cv2.line(viz, (0, zone_y), (w, zone_y), (80, 80, 80), 1)

    # Panah arah pandangan (dari center frame)
    cx, cy  = w // 2, h // 2
    ax = int(cx + r.iris_x * w * 0.18)
    ay = int(cy + r.iris_y * h * 0.18)
    cv2.arrowedLine(viz, (cx, cy), (ax, ay), (255, 120, 0), 2, tipLength=0.30)

    # Teks pose & iris
    info = [
        f"Yaw:{r.yaw:+.1f}  Pitch:{r.pitch:+.1f}",
        f"IrisX:{r.iris_x:+.2f} Y:{r.iris_y:+.2f}  HFore:{r.hand_forehead:.2f} HChin:{r.hand_chin:.2f}",
    ]
    if emotion_scores:
        labels = ["Bore", "Eng", "Conf", "Frus"]
        scores = "  ".join(f"{labels[i]}:{emotion_scores[i]:.2f}" for i in range(4))
        info.append(scores)

    for i, line in enumerate(info):
        cv2.putText(viz, line, (6, h - 12 - (len(info) - 1 - i) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
    return viz
