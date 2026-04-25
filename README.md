# Labeler Emosi SigLIP2

Aplikasi desktop untuk melabeli emosi pada video secara manual maupun semi-otomatis. Menggunakan model SigLIP2 (Google) dikombinasikan dengan analisis geometri wajah 3D dari MediaPipe (Hybrid Scoring).

## Daftar Isi

1. [Deskripsi](#deskripsi)
2. [Prasyarat & Instalasi](#prasyarat--instalasi)
3. [Menjalankan Aplikasi](#menjalankan-aplikasi)
4. [Panduan Penggunaan](#panduan-penggunaan)
5. [Aturan Scoring & Perhitungan](#aturan-scoring--perhitungan)
6. [Konfigurasi (.env)](#konfigurasi-env)
7. [Struktur Kode](#struktur-kode)
8. [File Output](#file-output)
9. [FAQ](#faq)

---

## Deskripsi

Aplikasi ini melabeli video dengan 4 kelas emosi. Setiap label bernilai `0` (tidak ada) atau `1` (ada). Sistem menggunakan pendekatan **multi-label** — satu video dapat memiliki lebih dari satu emosi aktif secara bersamaan.

| Label | Kode | Arti |
|---|---|---|
| Boredom | 0 | Siswa terlihat bosan, tidak memperhatikan |
| Engagement | 1 | Siswa terlihat fokus dan terlibat aktif |
| Confusion | 2 | Siswa terlihat bingung atau tidak mengerti |
| Frustration | 3 | Siswa terlihat frustrasi atau kesal |

Ada tiga mode pelabelan:
- **Manual** — nilai tiap label ditentukan secara manual oleh labeler
- **Semi-otomatis** — AI memberikan saran awal, labeler melakukan koreksi
- **Batch AI** — AI memproses seluruh dataset secara berurutan di background thread

---

## Prasyarat & Instalasi

**Persyaratan sistem:**
- Python 3.9 atau lebih baru
- pip
- GPU NVIDIA (opsional) — inferensi juga bisa berjalan di CPU

**Instalasi:**
```bash
pip install -r requirements.txt
```

Model SigLIP2 (~400 MB) dan MediaPipe FaceLandmarker (~30 MB) diunduh secara otomatis pada pertama kali aplikasi dijalankan.

---

## Menjalankan Aplikasi

```bash
# Salin dan sesuaikan konfigurasi (opsional)
cp .env.example .env

# Jalankan aplikasi
python app.py
```

---

## Panduan Penggunaan

### Membuka Dataset

Klik tombol **Buka Folder** di pojok kiri atas, lalu pilih folder root yang berisi file `.mp4`. Aplikasi akan:
1. Mencari semua file video secara rekursif
2. Membuat folder `hasil_label6/` sebagai direktori output
3. Memuat ulang label yang sudah pernah disimpan secara otomatis

### Memutar Video

Video diputar otomatis saat pertama dimuat. Klik area video untuk pause/play. Di bawah slider terdapat 16 thumbnail yang mewakili titik distribusi merata sepanjang video.

### Labeling Manual

Panel kanan berisi tombol `0` dan `1` untuk setiap label emosi. Untuk labeling per-frame:
1. Pilih tab label aktif (misalnya `Engagement`) di atas galeri frame
2. Klik kanan pada frame untuk toggle — border berwarna = positif, abu = negatif
3. Jika **8 atau lebih dari 16 frame** ditandai positif, nilai label video otomatis berubah menjadi `1`

### Menggunakan AI (Hybrid Scoring)

Panel kanan bawah berisi kontrol inferensi SigLIP2 + MediaPipe.

- **Proses Video Ini** — inferensi dijalankan pada video yang sedang ditampilkan
- **Batch Semua** — inferensi dijalankan pada seluruh dataset secara berurutan

Bar di samping tiap label menunjukkan **Hybrid Score** (gabungan SigLIP + Landmark). Video yang sudah diproses akan di-skip otomatis berdasarkan `batch_history.json`.

### Flag Otomatis

Jika MediaPipe tidak mendeteksi wajah pada lebih dari 50% frame, video akan secara otomatis di-flag (ditandai merah di daftar). Jumlah video terflag ditampilkan di topbar. Video terflag masih bisa dilabeli manual.

---

## Aturan Scoring & Perhitungan

Ini adalah bagian inti dari sistem. Skor akhir per label dihitung melalui dua tahap: **SigLIP Scoring** dan **Landmark Scoring**, lalu digabungkan menjadi **Hybrid Score**.

### 1. SigLIP Scoring (Visual)

**Input:** 16 frame crop wajah (PIL Image) + prompt teks per label.

**Alur:**
```
Frame (1-16) × Prompt (6 per label) → Logits [n_frames × n_prompts]
```

**Murni Sigmoid (Independent Scoring)**:
```
sigmoid_score = sigmoid(logit)               # hitung probabilitas (0-1) secara independen
siglip_score  = mean(sigmoid_score)           # rata-rata 6 prompt per frame
```

*Kenapa Sigmoid langsung?* SigLIP (Sigmoid-Loss Image-Language Pretraining) secara arsitektur dilatih menggunakan fungsi Sigmoid independen, bukan Softmax yang saling berkompetisi. Oleh karena itu, logit asli bisa langsung dimasukkan ke `sigmoid()` untuk mendapatkan probabilitas kemunculan emosi tersebut secara mandiri.

### 2. Landmark Scoring (Geometri Wajah)

**Input:** Frame BGR → MediaPipe FaceLandmarker + HandLandmarker.

**Output per frame:** Skor 0.0–1.0 untuk setiap emosi.

#### Boredom (label 0)
```
sig_yaw = clamp((|yaw| - 8°) / 10, 0, 1)       # noleh ≥8° mulai naik
sig_iris = clamp((|iris_x| - 0.12) / 0.23, 0, 1) # mata lirik ≥0.12 mulai naik
sig_arah = max(sig_yaw, sig_iris)                # OR logic: salah satu cukup

# Faktor pendukung (ekspresi):
blink_v    = clamp(max(eyeBlinkL, eyeBlinkR) / 0.4, 0, 1)
yawn_v     = clamp(jawOpen / 0.35, 0, 1)   # hanya jika pitch < 8°
pitch_up_v = clamp((pitch - 20°) / 25, 0, 1)
sig_expr   = max(blink_v, yawn_v, pitch_up_v) × 0.5

# Soft OR logic: (Tangan dihapus dari Boredom, pindah ke Frustration)
base_bore = max(sig_arah, sig_expr)
bore = clamp(base_bore × 0.85 + (sig_arah + sig_expr) × 0.15, 0, 1)
```

#### Engagement (label 1)
```
# AND logic: semua gate harus non-zero
gate_yaw  = clamp(1 - max(0, |yaw|-3°) / 7,  0, 1)   # dead zone ≤3°, 0 di ≥10°
gate_iris = clamp(1 - max(0, |iris_x|-0.08) / 0.17, 0, 1)  # dead zone ≤0.08
gate      = gate_yaw × gate_iris                        # keduanya harus ON

p_ok  = kualitas pitch (-25°≤pitch≤15° = 1.0, turun di luar range)
eye_op = 1 - max(eyeBlinkLeft, eyeBlinkRight)

eng = gate × (0.60 × p_ok + 0.40 × eye_op)
```

#### Confusion (label 2)
```
iris_up_v = clamp((-iris_y - 0.15) / 0.35, 0, 1)   
look_up_v = clamp(max(eyeLookUpL, eyeLookUpR) / 0.3, 0, 1)
pitch_cu  = clamp((pitch - 8°) / 17, 0, 1)
brow_dn_v = clamp(mean(browDownL, browDownR) / 0.12, 0, 1)
brow_in_v = clamp(browInnerUp / 0.12, 0, 1)
jaw_co    = clamp(jawOpen / 0.15, 0, 1)     # "mangap dikit" langsung memicu

sig_brow_conf = max(brow_dn_v, brow_in_v)
sig_mata_conf = max(iris_up_v, look_up_v)
smile_pen = max(0.0, max(mouthSmileLeft, mouthSmileRight) - 0.15)

# jaw_val_conf = naik saat jawOpen 0.05->0.20, turun jadi 0 saat jawOpen >0.35 (menguap/berteriak)
jaw_co = clamp(jaw_val_conf - smile_pen × 1.5, 0, 1)
pucker_co = clamp(mouthPucker / 0.20, 0, 1)

# Soft OR logic: Tangan dihapus dari Confusion (fokus ke ekspresi)
base_conf = max(sig_brow_conf, sig_mata_conf, jaw_co, pucker_co)
conf = clamp(base_conf × 0.85 + (pitch_cu + sig_brow_conf) × 0.15, 0, 1)
# Pembatalan Mutlak: Tangan menutupi wajah akan menihilkan Confusion (karena masuk ke Frustration)
conf = clamp(conf - hand_on_face, 0, 1)
```

#### Frustration (label 3)
```
br_fr = clamp(mean(browDownL, browDownR) / 0.12, 0, 1)
ns_fr = clamp(max(noseSneerL, noseSneerR) / 0.12, 0, 1)
ck_fr = clamp(mean(cheekSquintL, cheekSquintR) / 0.15, 0, 1)
lp_fr = clamp(mean(mouthPressL, mouthPressR) / 0.15, 0, 1)
ey_fr = clamp(mean(eyeSquintL, eyeSquintR) / 0.15, 0, 1)
jaw_val_frus = max(0.0, jawOpen - 0.10)
jw_fr = clamp((jaw_val_frus - smile_pen × 1.5) / 0.20, 0, 1)

# SUM LOGIC: Butuh kombinasi beberapa otot tegang (kecuali noseSneer yang mutlak)
sig_wajah_frus = clamp(ns_fr + (br_fr + lp_fr + ey_fr + ck_fr) / 2.0, 0, 1)
sig_wajah_frus = clamp(sig_wajah_frus - smile_pen × 1.5, 0, 1)

# Jika tangan menutupi wajah bawah/tengah (hand_mid_bot), skor ditambah secara absolut
hand_trigger_frus = hand_forehead
base_frus = clamp(sig_wajah_frus + hand_trigger_frus, 0, 1)
frus = clamp(base_frus × 0.85 + (ck_fr + jw_fr) × 0.15, 0, 1)
```

### 3. Hybrid Score & Prediksi Akhir

```
hybrid_score[frame] = α × siglip_score[frame] + β × landmark_score[frame]

avg_score = mean(hybrid_score[frame] for frame in 1..16)
prediction = 1 if avg_score >= threshold else 0
```

**Bobot default per label** (dapat diubah di `.env`):

| Label | α (SigLIP) | β (Landmark) | Alasan |
|---|---|---|---|
| Boredom | 0.50 | 0.50 | Seimbang: Landmark untuk pose (noleh), SigLIP untuk ekspresi (ngantuk) |
| Engagement | 0.50 | 0.50 | Seimbang: Landmark untuk pose tegak, SigLIP untuk mata fokus |
| Confusion | 0.50 | 0.50 | Seimbang: Landmark untuk kerutan alis, SigLIP untuk ekspresi bingung |
| Frustration | 0.50 | 0.50 | Seimbang: Keduanya saling melengkapi untuk deteksi stres |

**Threshold default:** 0.5 untuk semua label (dapat diubah di UI atau `.env`).

### 4. Aturan Label Video (Voting)

Prediksi akhir **level video** ditentukan dari skor rata-rata (bukan voting mayoritas):
```
label_video = 1  if avg_score >= threshold
label_video = 0  otherwise
```

Untuk labeling **manual per-frame** (klik kanan di galeri):
```
label_video = 1  if jumlah frame positif >= 8 dari 16
label_video = 0  otherwise
```

---

## Konfigurasi (.env)

Salin `.env.example` ke `.env` dan sesuaikan:

```env
# Model SigLIP2 yang digunakan
SIGLIP_MODEL_ID=google/siglip2-base-patch16-224

# Bobot hybrid global (fallback jika per-label tidak diset)
SIGLIP_WEIGHT=0.5
LANDMARK_WEIGHT=0.5

# Override per label (format: {LABEL}_SIGLIP_WEIGHT / {LABEL}_LANDMARK_WEIGHT)
BOREDOM_SIGLIP_WEIGHT=0.50
BOREDOM_LANDMARK_WEIGHT=0.50
ENGAGEMENT_SIGLIP_WEIGHT=0.50
ENGAGEMENT_LANDMARK_WEIGHT=0.50
CONFUSION_SIGLIP_WEIGHT=0.50
CONFUSION_LANDMARK_WEIGHT=0.50
FRUSTRATION_SIGLIP_WEIGHT=0.50
FRUSTRATION_LANDMARK_WEIGHT=0.50
```

---

## Struktur Kode

```
siglip2_Labeler_App/
├── app.py                      # Entry point — GUI utama & event orchestration
├── ai_service.py               # FastAPI microservice (opsional, headless mode)
├── requirements.txt
├── .env.example
│
├── core/
│   ├── siglip_model.py         # Singleton loader model SigLIP2 (lazy load)
│   ├── inference.py            # Hybrid scoring: SigLIP + Landmark fusion
│   ├── landmark_analyzer.py    # MediaPipe head pose, iris, blendshapes, hand
│   ├── face_detector.py        # Crop wajah dari frame video
│   └── README_SIGLIP.md        # Dokumentasi teknis pipeline inferensi
│
├── ui/
│   ├── constants.py            # Label, warna, prompt default
│   ├── left_panel.py           # Panel kiri: daftar video
│   ├── right_panel.py          # Panel kanan: tombol label & kontrol AI
│   └── video_player.py         # Widget video player + galeri frame
│
└── utils/
    ├── io.py                   # Baca/tulis CSV output
    └── video.py                # Ekstraksi frame dari file video
```

---

## File Output

Setiap video yang dilabeli menghasilkan satu baris di file CSV output (`hasil_label6/labels.csv`):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `video_path` | str | Path relatif ke file video |
| `Boredom` | 0/1 | Label boredom |
| `Engagement` | 0/1 | Label engagement |
| `Confusion` | 0/1 | Label confusion |
| `Frustration` | 0/1 | Label frustration |
| `labeled_by` | str | `"manual"` atau `"ai"` |
| `timestamp` | str | Waktu pelabelan |

---

## FAQ

**Q: Berapa lama proses Batch AI?**
A: Sekitar 3–8 detik per video tergantung panjang video dan apakah GPU tersedia. MediaPipe berjalan di CPU secara paralel dengan SigLIP di GPU.

**Q: Apakah bisa dipakai tanpa GPU?**
A: Bisa. Inferensi akan berjalan di CPU, sekitar 3–5× lebih lambat.

**Q: Kenapa ada video yang terflag merah?**
A: MediaPipe tidak berhasil mendeteksi wajah pada >50% frame. Ini biasanya terjadi pada video dengan kualitas rendah, pencahayaan buruk, atau wajah yang terlalu kecil/terhalang.

**Q: Bagaimana cara mengubah threshold?**
A: Slider threshold tersedia di panel kanan bawah. Perubahan hanya berlaku untuk sesi saat ini, kecuali disimpan ke `.env`.

**Q: Apakah skor SigLIP dan Landmark bisa dilihat terpisah?**
A: Ya. Hover pada bar skor di panel kanan untuk melihat detail `siglip_avg` dan `landmark_avg` secara terpisah.
