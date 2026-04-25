# Dokumentasi Teknis: Pipeline Inferensi SigLIP2 + MediaPipe

Dokumen ini menjelaskan arsitektur teknis dan alur matematis dari pipeline inferensi emosi pada `core/`.

## Ringkasan Alur

```
Video (MP4)
  └─► face_detector.py     → Crop wajah 512×512 per frame (16 frame)
        └─► inference.py   → SigLIP scoring + Landmark scoring → Hybrid score
              ├─► siglip_model.py        → SigLIP2 zero-shot visual scoring
              └─► landmark_analyzer.py   → MediaPipe 3D geometry scoring
```

---

## 1. Ekstraksi Frame & Crop Wajah (`face_detector.py`)

**Input:** File video MP4.  
**Output:** 16 frame PIL Image (crop wajah 512×512 atau resize dari frame asli).

Alur:
1. Sampel 16 frame dari distribusi merata sepanjang durasi video.
2. Deteksi wajah menggunakan MediaPipe Face Detector (BlazeFace).
3. Jika wajah ditemukan: crop area wajah + padding 20%, resize ke 512×512.
4. Jika tidak ditemukan: gunakan frame asli diresized ke 224×224 (di-flag).

---

## 2. SigLIP2 Scoring (`siglip_model.py` + `inference.py`)

### Model

- **Model:** `google/siglip2-base-patch16-224` (default, dapat diubah via env)
- **Tipe:** Vision-Language Contrastive Model (zero-shot classification)
- **Input:** Image 224×224 + teks prompt
- **Output:** Logit cosine similarity antara gambar dan teks

### Prompt per Label

Setiap label memiliki 6 prompt positif yang ditulis di `ui/constants.py`. Prompt dirancang untuk mendeskripsikan ekspresi visual spesifik dalam konteks belajar siswa (bukan deskripsi emosi generik).

### Per-Group Sigmoid Shift

Ini adalah teknik kunci yang membuat scoring menjadi **independen per emosi**:

```python
# Untuk setiap emosi i dan setiap frame f:
group_logits = logits[f, prompt_indices[i]]       # [6 logit]
max_logit    = max(group_logits)
shifted      = group_logits - max_logit + 2.0     # best prompt → 2.0
probs        = sigmoid(shifted)                    # best prompt → sigmoid(2.0) ≈ 0.88
siglip_score[f][i] = mean(probs)                  # rata-rata 6 prompt
```

**Mengapa shift ke 2.0?**  
`sigmoid(2.0) ≈ 0.88` memberikan skor tinggi yang masuk akal untuk prompt terbaik dalam grup. Tanpa shift, semua logit bernilai negatif (cosine similarity rendah) dan sigmoid-nya akan sangat kecil.

**Mengapa per-group, bukan global?**  
Jika semua 24 prompt (4 emosi × 6 prompt) dinormalisasi bersama, emosi dengan logit paling tinggi (biasanya Engagement) akan selalu mendominasi. Per-group shift memastikan setiap emosi **dinilai relatif terhadap prompt terbaiknya sendiri**.

---

## 3. MediaPipe Landmark Scoring (`landmark_analyzer.py`)

### Model

- **FaceLandmarker:** Menghasilkan 478 landmark 3D, head pose matrix, dan 52 blendshapes.
- **HandLandmarker:** Menghasilkan landmark tangan untuk deteksi gesture Frustration/Confusion.

### Koordinat & Sinyal Utama

| Sinyal | Definisi | Digunakan untuk |
|---|---|---|
| `yaw` | Rotasi kepala horizontal (derajat) | Boredom, Engagement |
| `pitch` | Rotasi kepala vertikal (derajat) | Engagement, Confusion |
| `iris_x` | Offset iris horizontal relatif ke sudut mata (−1..+1) | Boredom, Engagement |
| `iris_y` | Offset iris vertikal relatif ke sudut mata (−1..+1) | Confusion |
| Blendshapes | 52 koefisien otot wajah (0..1) | Semua label |
| `hand_forehead` | Proporsi titik tangan di zona dahi (y < 0.45) | Frustration |
| `hand_chin` | Proporsi titik tangan di zona pipi/dagu (y 0.45–0.80) | Confusion |

### Catatan Penting: Koordinat Iris

Karena frame yang diinput adalah **crop yang sudah mengikuti wajah** (`face_detector.py`), posisi iris **relatif ke frame (iris_img_x/y) tidak dipakai** — nilainya akan selalu mendekati 0.5 karena wajah selalu di tengah crop.

Yang dipakai adalah `iris_x` (posisi pupil relatif ke **sudut mata kiri-kanan**), yang masih mengandung informasi arah pandangan yang valid meski frame sudah di-crop.

### Formula Scoring per Emosi

Lihat [README.md](../README.md#aturan-scoring--perhitungan) untuk formula lengkap dengan semua koefisien.

---

## 4. Hybrid Scoring (`inference.py`)

```python
hybrid_score = α × siglip_score + β × landmark_score

# α dan β dibaca dari env, berbeda per label:
# - Boredom:     α=0.45, β=0.55  (landmark dominan)
# - Engagement:  α=0.45, β=0.55  (landmark dominan)
# - Confusion:   α=0.75, β=0.25  (SigLIP dominan)
# - Frustration: α=0.65, β=0.35  (SigLIP dominan)
```

**Alasan bobot berbeda per label:**

- **Boredom & Engagement**: Sinyal geometri (arah kepala, arah mata) sangat spesifik dan tidak ambigu. SigLIP cenderung bingung membedakan siswa yang "fokus" vs "bosan" secara visual.
- **Confusion & Frustration**: Blendshapes wajah (kerutan alis, cibiran hidung, dll) sangat subtle sehingga landmark tidak cukup sensitif. SigLIP lebih baik dalam menangkap ekspresi wajah yang lebih global.

**Prediksi akhir:**
```python
avg_score  = mean(hybrid_score for all 16 frames)
prediction = 1 if avg_score >= threshold else 0
```

---

## 5. Konfigurasi Bobot via Environment

Semua bobot dapat dikonfigurasi tanpa mengubah kode:

```env
# Bobot global (fallback)
SIGLIP_WEIGHT=0.5
LANDMARK_WEIGHT=0.5

# Override per label (LABEL = BOREDOM | ENGAGEMENT | CONFUSION | FRUSTRATION)
BOREDOM_SIGLIP_WEIGHT=0.45
BOREDOM_LANDMARK_WEIGHT=0.55
```

Urutan prioritas: **per-label env** → **global env** → **hardcoded default**.

---

## 6. Debug Output

Saat `_DBG_LAND = True` di `landmark_analyzer.py`, setiap frame mencetak:

```
[LAND] yaw=+12.3 iris_x=+0.210 | sig_yaw=0.73 sig_iris=0.44 arah=0.73 | gate_yaw=0.00 gate_iris=0.81 gate=0.00 | B=0.547 E=0.000 C=0.124 F=0.089
```

Set `_DBG_LAND = False` untuk mode production.

---

## Referensi

- SigLIP2: [arxiv.org/abs/2502.08769](https://arxiv.org/abs/2502.08769)
- MediaPipe FaceLandmarker: [mediapipe.readthedocs.io](https://mediapipe.readthedocs.io/en/latest/solutions/face_landmarker.html)
- MediaPipe BlazeFace Blendshapes: [developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
