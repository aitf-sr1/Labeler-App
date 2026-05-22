# Dokumentasi Teknis: Pipeline Inferensi SigLIP2 + MediaPipe

> **Navigasi:** [README utama](../README.md) · [Perhitungan Step-by-Step](../docs/COMPUTATION.md) · [Rules Editor](../docs/RULES_PANEL.md)

Dokumen ini menjelaskan **arsitektur dan alur data** pipeline inferensi emosi pada `core/` — bagaimana modul-modul terhubung, apa yang masuk dan keluar dari setiap tahap, dan bagaimana sistem cache bekerja.

> Untuk **rumus matematis lengkap** dengan contoh angka nyata (cara menghitung yaw, iris offset, gaze_dev, setiap skor emosi, SigLIP sigmoid, hybrid combine), lihat [docs/COMPUTATION.md](../docs/COMPUTATION.md).

## Ringkasan Alur

```
Video (MP4)
  └─► utils/video.py          → extract_6_frames() → 6 frame BGR
        └─► face_detector.py  → crop wajah 512×512 per frame + bbox
        └─► landmark_analyzer.py → detect_hands_from_full_frame()
              └─► inference.py → SigLIP scoring + Landmark scoring → Hybrid score
                    ├─► siglip_model.py       → SigLIP2 zero-shot visual scoring
                    └─► landmark_analyzer.py  → MediaPipe 3D geometry scoring

Cache (saat pertama diproses):
  raw_cache/{safe_name}.json     ← MediaPipe features per frame
  siglip_cache/{safe_name}.json  ← SigLIP scores per frame

Recalculate (tanpa re-run model):
  raw_cache + siglip_cache → recalculate.py → batch_history + frame_annotations
```

---

## 1. Ekstraksi Frame & Crop Wajah (`face_detector.py`, `utils/video.py`)

**Input:** File video MP4.  
**Output:** 6 frame PIL Image (crop wajah 512×512) + landmark results + viz images.

Alur di `utils/video.py`:
1. Sampel **6 frame** dari distribusi merata sepanjang durasi video (`extract_6_frames`)
2. Per frame: deteksi tangan dari **full frame asli** (sebelum crop) via `detect_hands_from_full_frame()`
3. Crop wajah via `crop_face()` — MediaPipe BlazeFace + padding (default 0.60) + resize ke 512×512
4. Analisis landmark dari crop via `analyze_frame()` dengan hand results yang sudah di-remap ke ruang crop
5. Hitung emotion scores via `compute_emotion_scores(lr, cfg)` untuk viz rendering
6. Simpan ke `raw_cache/` jika belum ada

`crop_face()` mengembalikan 4-tuple `(crop_bgr, face_found, n_faces, bbox)`. Bounding box wajah disimpan dan dipakai oleh `detect_hands_from_full_frame()` agar deteksi tangan berjalan dari full frame (bukan dari crop yang terlalu ketat untuk menampung telapak tangan).

---

## 2. SigLIP2 Scoring (`siglip_model.py` + `inference.py`)

### Model

- **Model:** `google/siglip2-base-patch16-224` (default, dapat diubah via `SIGLIP_MODEL_ID` di `.env`)
- **Tipe:** Vision-Language Contrastive Model (zero-shot multi-label classification)
- **Input:** Image 224×224 + teks prompt per label
- **Output:** Logit cosine similarity antara gambar dan teks

### Prompt per Label

Setiap label memiliki 6–8 prompt positif di `ui/constants.py`. Prompt ditulis untuk mendeskripsikan ekspresi visual spesifik dalam konteks siswa belajar — bukan deskripsi emosi generik. Prompt bisa diedit langsung dari UI panel kanan.

### Sigmoid + Empirical Bias

SigLIP (Sigmoid-Loss Image-Language Pretraining) dilatih dengan fungsi Sigmoid independen, bukan Softmax. Logit asli langsung dimasukkan ke `sigmoid()`:

```python
EMPIRICAL_BIAS = cfg["hybrid"]["empirical_bias"]   # default: 3.5

for i in range(n_labels):
    group_logits = logits[:, prompt_indices[i]]               # [n_frames, n_prompts]
    probs        = torch.sigmoid(group_logits + EMPIRICAL_BIAS)  # [n_frames, n_prompts]
    siglip_score[i] = probs.mean(dim=1)                       # [n_frames]
```

**Kenapa Empirical Bias?**  
Logit zero-shot SigLIP untuk deskripsi spesifik seringkali bernilai negatif (-3 sampai -6). Tanpa bias, `sigmoid(-5)` ≈ 0.007 — terlalu kecil untuk hybrid scoring yang bermakna. Bias +3.5 menggeser kurva ke area 0.2–0.9 tanpa memanipulasi distribusi relatif antar prompt.

**Kenapa bukan normalisasi min-max?**  
Normalisasi min-max per frame memaksa nilai tertinggi di setiap grup *selalu* menjadi 1.0 — bahkan ketika emosi tersebut tidak ada sama sekali. Sigmoid + bias mempertahankan keyakinan absolut model.

### Penyimpanan SigLIP Cache

Setelah scoring, skor SigLIP per frame (sebelum hybrid combine) disimpan ke `siglip_cache/`:

```json
{
  "video_rel": "kelas1/siswa_abc/video.mp4",
  "generated_at": "2025-01-01T12:00:00",
  "frames": [
    { "frame_idx": 0, "siglip_scores": [0.31, 0.62, 0.28, 0.19] },
    { "frame_idx": 1, "siglip_scores": [0.29, 0.58, 0.31, 0.22] },
    ...
  ]
}
```

Cache ini memungkinkan Recalculate tanpa re-run SigLIP (model terbesar dalam pipeline).

---

## 3. MediaPipe Landmark Scoring (`landmark_analyzer.py`)

### Model

- **FaceLandmarker:** 478 landmark 3D, head pose matrix 4×4, 52 blendshapes.
- **HandLandmarker:** 21 landmark per tangan, deteksi dari **full frame** (bukan crop).

### Koordinat & Sinyal Utama

| Sinyal | Definisi | Digunakan untuk |
|---|---|---|
| `yaw` | Rotasi kepala horizontal (°, + = kanan) | Boredom, Engagement, Frustration |
| `pitch` | Rotasi kepala vertikal (°, + = mendongak) | Boredom, Engagement, Confusion |
| `iris_x` | Offset iris horizontal relatif ke sudut mata (−1..+1) | Boredom, Engagement |
| `iris_y` | Offset iris vertikal relatif ke sudut mata (−1..+1) | Confusion |
| Blendshapes | 52 koefisien otot wajah (0..1) dari MediaPipe | Semua label |
| `hand_forehead` | Proporsi tangan di zona DAHI (y ∈ [-0.20, 0.25) dari crop) | + Frustration, − Confusion |
| `hand_chin` | Proporsi tangan di zona PIPI/DAGU (y ∈ [0.25, 1.20] dari crop) | + Confusion |

### Koordinat Iris

Frame yang diinput adalah **crop yang sudah mengikuti wajah**, sehingga `iris_img_x/y` (posisi iris absolut relatif ke pusat frame) selalu ≈ 0 dan tidak dipakai. Yang dipakai adalah `iris_x` (posisi pupil relatif ke **sudut dalam/luar mata**), yang tetap valid meski frame sudah di-crop.

### Deteksi Tangan dari Full Frame

HandLandmarker memerlukan telapak tangan untuk tahap deteksi pertama. Dengan padding crop 0.30–0.60, telapak sering terpotong. Solusi: tangan dideteksi dari **full frame video asli**, lalu koordinatnya di-remap ke ruang crop wajah:

```python
# utils/video.py — alur per frame:
hand_top, hand_mid_bot, hand_pts_px, hand_raw = detect_hands_from_full_frame(
    full_frame, face_bbox, crop_size=512
)
lr = analyze_frame(crop_bgr, injected_hand=(hand_top, hand_mid_bot, hand_pts_px, hand_raw))
```

### Penyimpanan Raw Feature Cache

Setelah landmark analysis, fitur per frame disimpan ke `raw_cache/`:

```json
{
  "video_rel": "kelas1/siswa_abc/video.mp4",
  "generated_at": "2025-01-01T12:00:00",
  "frames": [
    {
      "frame_idx": 0,
      "face_found": true,
      "yaw": 5.2,
      "pitch": -2.1,
      "iris_x": 0.12,
      "iris_y": -0.05,
      "iris_img_x": 0.01,
      "iris_img_y": 0.02,
      "blendshapes": { "browDownLeft": 0.12, "jawOpen": 0.08, "... (52 total)": 0 },
      "hand_forehead": 0.0,
      "hand_chin": 0.0
    }
  ]
}
```

**Yang tidak disimpan di cache:** face_landmarks (478 titik), hand_landmarks_px, iris_px — tidak diperlukan untuk recalculate, hanya untuk viz rendering. Viz di-regenerate dari landmark_results yang tersimpan di gallery cache memori.

---

## 4. Hybrid Scoring (`inference.py`)

```python
hybrid_score[frame] = (sw × siglip_score[frame] + lw × landmark_score[frame]) / (sw + lw)
avg_score            = mean(hybrid_score untuk 6 frame)
prediction           = 1 jika vote_pos >= ceil(n_valid / 2) else 0
```

Threshold diterapkan per frame: `frame_pred = 1 if hybrid_score >= threshold else 0`, lalu voting mayoritas dari frame yang tidak di-reject.

**Bobot per label (dari `rules["hybrid"]["siglip_w"]` dan `rules["hybrid"]["land_w"]`):**

| Label | SigLIP (α) | Landmark (β) | Alasan |
|---|---|---|---|
| Boredom | 0.50 | 0.50 | Landmark kuat untuk yaw/iris; SigLIP untuk ekspresi lelah/kosong |
| Engagement | 0.50 | 0.50 | Gate logic Landmark sangat presisi; SigLIP mendukung visual |
| Confusion | **0.60** | **0.40** | SigLIP lebih baik untuk ekspresi berpikir halus |
| Frustration | 0.50 | 0.50 | Seimbang: Landmark untuk gestur tangan, SigLIP untuk aura stres |

**Temporal Restlessness Bonus (Boredom):**  
Jika std deviasi yaw antar frame ≥ `restless_std_min` (default: 3°), skor Boredom mendapat bonus hingga `restless_bonus_max` (default: +0.15). Menangkap pola tolah-toleh yang tidak terlihat per-frame.

---

## 5. Recalculate Pipeline (`core/recalculate.py`)

Recalculate memungkinkan perubahan rules/threshold tanpa re-run SigLIP atau MediaPipe:

```
raw_cache/{safe}.json  →  _reconstruct_lr()  →  LandmarkResult
                                                      ↓
                                         compute_emotion_scores(lr, rules_baru)
                                                      ↓
siglip_cache/{safe}.json  ────────────────────── hybrid combine
                                                      ↓
                                          restless bonus (Boredom)
                                                      ↓
                                          threshold → frame_preds
                                                      ↓
                                          vote (exclude rejected frames)
                                                      ↓
                                     updated_batch_history + updated_frame_annotations
```

`_reconstruct_lr()` membuat objek `LandmarkResult` dari dict cache — semua field scoring (yaw, pitch, iris, blendshapes, hand zones) tersedia, tapi field viz (face_landmarks 3D, hand_landmarks_px, iris_px) diset ke None/kosong.

---

## 6. Viz Regeneration

Setelah Recalculate, viz thumbnail untuk video aktif di-regenerate menggunakan:
- `pil_images` (clean crops dari gallery cache memori)
- `landmark_results` (hasil MediaPipe dari terakhir kali video dimuat, tersimpan di gallery cache)
- `compute_emotion_scores(lr, rules_baru)` → `draw_landmark_viz(bgr, lr, scores)`

Viz tidak disimpan di `raw_cache/` karena membutuhkan data visual (full LandmarkResult dengan face_landmarks 3D untuk mesh tesselation), bukan hanya scalar features.

---

## 7. Konfigurasi Parameter

Semua konstanta scoring tersimpan di `core/rules.py` sebagai `DEFAULT_RULES`. Parameter ini bisa diubah via Rules Editor di UI → disimpan ke `rules.json` di folder output → dimuat saat aplikasi dibuka.

Lihat [docs/RULES_PANEL.md](../docs/RULES_PANEL.md) untuk daftar lengkap semua parameter dan panduan tuning.

---

## 8. Debug Output

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

| Komponen | Sumber |
|---|---|
| SigLIP (sigmoid loss) | Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV. arXiv [2303.15343](https://arxiv.org/abs/2303.15343) |
| SigLIP 2 (model yang dipakai) | Tschannen et al. (2025). *SigLIP 2.* arXiv [2502.08769](https://arxiv.org/abs/2502.08769) |
| MediaPipe FaceLandmarker | Google LLC. [developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |
| MediaPipe HandLandmarker | Google LLC. [developers.google.com/mediapipe/solutions/vision/hand_landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |
| 52 Blendshapes (standar ARKit) | Apple Inc. [ARFaceAnchor.BlendShapeLocation](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation) |
| Euler angles dari rotation matrix | Diebel (2006). *Representing Attitude.* Stanford TR. [diebel.com](https://www.diebel.com/attitude/Diebel2006.pdf) |
| FACS (basis blendshape) | Ekman & Friesen (1978). *Facial Action Coding System.* |
| Engagement & head pose | D'Mello & Graesser (2012). *Dynamics of Affective States during Complex Learning.* |

> Referensi lengkap dengan konteks penggunaan dalam kode ada di [docs/COMPUTATION.md — Referensi & Sumber](../docs/COMPUTATION.md#referensi--sumber).

---

## Dokumen Terkait

- [docs/COMPUTATION.md](../docs/COMPUTATION.md) — Rumus step-by-step, contoh angka, sumber akademis
- [docs/RULES_PANEL.md](../docs/RULES_PANEL.md) — Cara mengatur parameter dan strategi tuning
- [README.md](../README.md) — Panduan instalasi, UI, FAQ
