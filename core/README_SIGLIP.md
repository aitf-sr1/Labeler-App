# Dokumentasi Teknis: Pipeline Inferensi SigLIP2 + MediaPipe

Dokumen ini menjelaskan arsitektur teknis dan alur matematis dari pipeline inferensi emosi pada `core/`.

> Untuk panduan anotasi dan ciri-ciri setiap emosi, lihat [README.md utama](../README.md).

## Ringkasan Alur

```
Video (MP4)
  └─► face_detector.py      → Crop wajah 512×512 per frame (4 frame)
  └─► landmark_analyzer.py  → detect_hands_from_full_frame() sebelum crop
        └─► inference.py    → SigLIP scoring + Landmark scoring → Hybrid score
              ├─► siglip_model.py       → SigLIP2 zero-shot visual scoring
              └─► landmark_analyzer.py  → MediaPipe 3D geometry scoring
```

---

## 1. Ekstraksi Frame & Crop Wajah (`face_detector.py`)

**Input:** File video MP4.  
**Output:** 4 frame PIL Image (crop wajah 512×512).

Alur:
1. Sampel **4 frame** dari distribusi merata sepanjang durasi video.
2. Deteksi wajah menggunakan MediaPipe BlazeFace.
3. Jika wajah ditemukan: crop area wajah + padding (default 0.30), resize ke 512×512.
4. Jika tidak ditemukan: fallback ke center crop 80%.

`crop_face()` mengembalikan 4-tuple `(crop_bgr, face_found, n_faces, (x1,y1,x2,y2))`. Bounding box wajah disimpan dan dipakai oleh `detect_hands_from_full_frame()` agar deteksi tangan berjalan dari frame penuh (bukan dari crop yang terlalu ketat).

---

## 2. SigLIP2 Scoring (`siglip_model.py` + `inference.py`)

### Model

- **Model:** `google/siglip2-base-patch16-224` (default, dapat diubah via `SIGLIP_MODEL_ID` di `.env`)
- **Tipe:** Vision-Language Contrastive Model (zero-shot multi-label classification)
- **Input:** Image 224×224 + teks prompt
- **Output:** Logit cosine similarity antara gambar dan teks

### Prompt per Label

Setiap label memiliki 6 prompt positif di `ui/constants.py`. Prompt ditulis untuk mendeskripsikan ekspresi visual spesifik dalam konteks siswa belajar — bukan deskripsi emosi generik. Prompt bisa diedit langsung dari UI panel kanan.

### Murni Sigmoid + Empirical Bias

SigLIP (Sigmoid-Loss Image-Language Pretraining) dilatih dengan fungsi Sigmoid independen, bukan Softmax. Logit asli langsung dimasukkan ke `sigmoid()`:

```python
EMPIRICAL_BIAS = 3.5  # menggeser logit negatif ke area sensitivitas sigmoid

for i in range(n_labels):
    group_logits = logits[:, prompt_indices[i]]          # [n_frames, 6]
    probs        = torch.sigmoid(group_logits + EMPIRICAL_BIAS)  # [n_frames, 6]
    siglip_score[i] = probs.mean(dim=1)                  # [n_frames]
```

**Kenapa Empirical Bias?**  
Logit zero-shot SigLIP untuk deskripsi spesifik seringkali bernilai negatif (-3 sampai -6). Tanpa bias, `sigmoid(-5)` ≈ 0.007 — terlalu kecil untuk hybrid scoring yang bermakna. Bias +3.5 menggeser kurva ke area 0.2–0.9 tanpa memanipulasi distribusi relatif antar prompt.

**Kenapa bukan normalisasi min-max?**  
Normalisasi min-max per frame memaksa nilai tertinggi di setiap grup *selalu* menjadi 1.0 — bahkan ketika emosi tersebut tidak ada sama sekali. Sigmoid + bias mempertahankan keyakinan absolut model.

---

## 3. MediaPipe Landmark Scoring (`landmark_analyzer.py`)

### Model

- **FaceLandmarker:** 478 landmark 3D, head pose matrix 4×4, 52 blendshapes.
- **HandLandmarker:** 21 landmark per tangan, deteksi dari **full frame** (bukan crop).

### Koordinat & Sinyal Utama

| Sinyal | Definisi | Digunakan untuk |
|---|---|---|
| `yaw` | Rotasi kepala horizontal (derajat, + = kanan) | Boredom, Engagement |
| `pitch` | Rotasi kepala vertikal (derajat, + = mendongak) | Engagement, Confusion, Boredom |
| `iris_x` | Offset iris horizontal relatif ke sudut mata (−1..+1) | Boredom, Engagement |
| `iris_y` | Offset iris vertikal relatif ke sudut mata (−1..+1) | Confusion |
| Blendshapes | 52 koefisien otot wajah (0..1) dari MediaPipe | Semua label |
| `hand_chin` (field) | Proporsi titik tangan di **zona y < 0.25** crop (atas frame) | +Confusion |
| `hand_forehead` (field) | Proporsi titik tangan di **zona y 0.25–1.20** crop (tengah/bawah) | +Frustration, −Confusion |

> **Catatan penamaan:** `hand_chin` menyimpan skor zona ATAS crop (y < 0.25 = area di atas mata/kepala → menggaruk kepala → Confusion). `hand_forehead` menyimpan skor zona TENGAH/BAWAH (y 0.25–1.20 = area wajah/dagu → facepalm → Frustration). Penamaan ini berlawanan intuisi tapi konsisten di seluruh kode.

### Catatan: Koordinat Iris

Frame yang diinput adalah **crop yang sudah mengikuti wajah**, sehingga `iris_img_x/y` (posisi iris relatif ke pusat frame) selalu ≈ 0 dan tidak dipakai. Yang dipakai adalah `iris_x` (posisi pupil relatif ke **sudut dalam/luar mata**), yang tetap valid meski frame sudah di-crop.

### Deteksi Tangan dari Full Frame

HandLandmarker memerlukan telapak tangan untuk tahap deteksi pertama. Dengan padding crop 0.20–0.30, telapak sering terpotong. Solusi: tangan dideteksi dari **full frame video asli**, lalu koordinatnya di-remap ke ruang crop wajah:

```python
# utils/video.py — alur per frame:
crop, face_found, n_faces, face_bbox = crop_face(full_frame)
hand_top, hand_mid_bot, hand_pts_px, hand_raw = detect_hands_from_full_frame(
    full_frame, face_bbox, crop_size=512
)
lr = analyze_frame(crop_bgr, injected_hand=(hand_top, hand_mid_bot, hand_pts_px, hand_raw))
```

---

## 4. Hybrid Scoring (`inference.py`)

```python
hybrid_score[frame] = α × siglip_score[frame] + β × landmark_score[frame]
avg_score            = mean(hybrid_score for 4 frames)
prediction           = 1 if vote_pos >= 2 else 0   # mayoritas: ≥2 dari 4 frame positif
```

**Bobot per label (nilai aktual di `.env`, dapat diubah):**

| Label | α (SigLIP) | β (Landmark) | Alasan |
|---|---|---|---|
| Boredom | 0.50 | 0.50 | Landmark baik untuk yaw/iris; SigLIP baik untuk ekspresi lelah/kosong |
| Engagement | 0.50 | 0.50 | AND gate Landmark sangat presisi; SigLIP mendukung visual |
| Confusion | **0.60** | **0.40** | SigLIP lebih baik untuk ekspresi berpikir halus; landmark mudah false-positive/negative |
| Frustration | 0.50 | 0.50 | Landmark untuk gestur tangan; SigLIP untuk aura stres |

**Temporal Restlessness Bonus (Boredom):**  
Jika std deviasi yaw ≥ 3° di 4 frame, skor Boredom mendapat bonus hingga +0.15. Menangkap pola tolah-toleh yang tidak terlihat per-frame.

---

## 5. Konfigurasi Bobot via Environment

```env
# Bobot global (fallback jika per-label tidak diset)
SIGLIP_WEIGHT=0.5
LANDMARK_WEIGHT=0.5

# Override per label
BOREDOM_SIGLIP_WEIGHT=0.50
BOREDOM_LANDMARK_WEIGHT=0.50
ENGAGEMENT_SIGLIP_WEIGHT=0.50
ENGAGEMENT_LANDMARK_WEIGHT=0.50
CONFUSION_SIGLIP_WEIGHT=0.60    # ← SigLIP lebih dominan
CONFUSION_LANDMARK_WEIGHT=0.40
FRUSTRATION_SIGLIP_WEIGHT=0.50
FRUSTRATION_LANDMARK_WEIGHT=0.50
```

Urutan prioritas: **per-label env** → **global env** → **hardcoded default (0.50/0.50)**.

---

## 6. Debug Output

Saat `_DBG_LAND = True` di `landmark_analyzer.py` (default aktif), setiap frame mencetak:

```
[LAND] yaw=+12.3 iris_x=+0.210 iris_y=-0.031 lookDn=0.04 | gH=+19.7° gV=0.0° dev=19.7° | boreGaze=0.73 gate=0.00 | B=0.621 E=0.000 C=0.081 F=0.043
[CONF] brow_dn=0.43 brow_in=0.00 iris_up=0.12 look_up=0.34 jaw=0.10 pucker=0.03 pitch_cu=0.00 base=0.43
[HAND-FULL] Terdeteksi 1 tangan, in_crop=12, pts_top=3, pts_mid=6, pts_bot=3
```

Format log `[LAND]`: `gH` = komponen horizontal gaze (°), `gV` = komponen vertikal (°), `dev` = total `gaze_dev`, `boreGaze` = kontribusi gaze ke Boredom, `gate` = gate Engagement dari gaze yang sama.

Set `_DBG_LAND = False` untuk mematikan log di mode production/batch.

---

## Referensi

- SigLIP2: [arxiv.org/abs/2502.08769](https://arxiv.org/abs/2502.08769)
- MediaPipe FaceLandmarker: [mediapipe.readthedocs.io](https://mediapipe.readthedocs.io/en/latest/solutions/face_landmarker.html)
- MediaPipe BlazeFace Blendshapes: [developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
