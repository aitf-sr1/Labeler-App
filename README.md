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
### 2. Evaluasi Skor Emosi (Landmark-based)

Setiap kelas memiliki kombinasi "otot wajah" spesifik (Blendshapes) dan posisi kepala. Rentang nilai selalu **0.0 - 1.0** menggunakan fungsi `clamp()`.

#### A. Zonasi Tangan (Hand Zoning 5-Titik)
Untuk membedakan gestur *mikir/bingung* dengan gestur *stres/frustrasi*, tangan dibagi menjadi zona:
1. **Zona Atas (`y < 0.25`)**: Gestur **menggaruk kepala**. Langsung memicu *Confusion* (1.0).
2. **Zona Tengah & Bawah (`0.25 <= y <= 1.20`)**: Gestur **facepalm, kucek mata, atau menopang dagu**. Langsung memicu *Frustration* (1.0).

**Pertimbangan Matematis & Akurasi:**
- **Threshold 5 Titik:** MediaPipe membutuhkan 21 titik untuk 1 tangan penuh. Namun, saat siswa melakukan *facepalm* atau bertopang dagu dari bawah *frame* kamera, seringkali **hanya ujung jarinya saja (sekitar 5 titik)** yang tertangkap. Dengan menetapkan ambang batas proporsional ke 5 titik (dibagi 5), kita menjamin bahwa ujung jari yang menutupi wajah pun sudah cukup untuk memicu nilai tangan `1.0` secara maksimal.
- **Filter Noise:** Jika total titik tangan di area tengah kurang dari 5, sistem akan membuangnya menjadi `0.0`. Ini mencegah *noise* (seperti MediaPipe salah mengira kerah baju sebagai jari) menghancurkan skor emosi.

#### B. Confusion (label 2)
Fokus pada ekspresi berpikir: mengerutkan alis, memiringkan kepala, mulut mengerucut, atau **menggaruk kepala**.

```python
iris_up_v  = clamp((-iris_y - 0.15) / 0.35, 0, 1) # Lirik ke atas
look_up_v  = clamp(max(eyeLookUpL, eyeLookUpR) / 0.3, 0, 1)
pitch_cu   = clamp((pitch - 8) / 17, 0, 1)        # Kepala mendongak/miring
brow_dn_v  = clamp(mean(browDownL, browDownR) / 0.15, 0, 1) # Alis mengkerut (mikir)
brow_in_v  = clamp(browInnerUp / 0.15, 0, 1)      # Alis dalam naik (bingung)

# Mulut sedikit terbuka (mangap bingung) atau bibir mengerucut (mikir keras)
jaw_co = clamp(jaw_val_conf - smile_pen × 1.5, 0, 1)
pucker_co = clamp(mouthPucker / 0.20, 0, 1)

# Jika garuk kepala (hand_top), base_conf menjadi tinggi
base_conf = max(sig_brow_conf, sig_mata_conf, jaw_co, pucker_co, hand_top)
conf = clamp(base_conf × 0.85 + (pitch_cu + sig_brow_conf) × 0.15, 0, 1)

# PEMBATALAN MUTLAK (Suppression Logic):
# Jika tangan berada di area wajah/dagu (Frustration), Confusion HARUS dimatikan jadi 0.
# Namun, jika tangan sedang menggaruk kepala (hand_top dominan), Confusion dibiarkan hidup.
suppression = hand_mid_bot if hand_top < 0.5 else 0.0
conf = clamp(conf - suppression, 0, 1)
```

#### C. Frustration (label 3)
Fokus pada ketegangan ekstrem: marah, mengernyit jijik, menutup mata rapat, merapatkan bibir, dan **menyentuh wajah/menopang dagu**.

**Alasan Penggunaan SUM Logic (Bukan MAX):**
Pada versi awal, digunakan logika `MAX`. Jika salah satu otot saja bernilai tinggi (misal: bibir merapat karena sedang diam/ngelamun santai), maka *Frustration* langsung tembus 1.0. Ini menyebabkan siswa yang diam salah terdeteksi sebagai Frustrasi.
Oleh karena itu, sistem diubah menjadi **SUM Logic (Kombinasi Rata-rata)**. *Frustration* wajah HANYA akan tembus 1.0 jika siswa secara bersamaan memadukan beberapa otot stres (alis tegang + bibir rapat + mata menyipit tajam).

```python
# Threshold dinaikkan drastis (0.40) agar ekspresi mikir biasa tidak bocor ke Frustrasi
br_fr = clamp(mean(browDownL, browDownR) / 0.40, 0, 1)
ns_fr = clamp(max(noseSneerL, noseSneerR) / 0.20, 0, 1)
ck_fr = clamp(mean(cheekSquintL, cheekSquintR) / 0.40, 0, 1)
lp_fr = clamp(mean(mouthPressL, mouthPressR) / 0.40, 0, 1)
ey_fr = clamp(mean(eyeSquintL, eyeSquintR) / 0.40, 0, 1)
jw_fr = clamp((max(0.0, jawOpen - 0.10) - smile_pen × 1.5) / 0.20, 0, 1)

# SUM LOGIC: Butuh kombinasi beberapa otot tegang (kecuali noseSneer yang mutlak)
sig_wajah_frus = clamp(ns_fr + (br_fr + lp_fr + ey_fr + ck_fr) / 2.0, 0, 1)
sig_wajah_frus = clamp(sig_wajah_frus - smile_pen × 1.5, 0, 1)

# Jika tangan menutupi wajah bawah/tengah (hand_mid_bot), skor ditambah secara absolut
hand_trigger_frus = hand_mid_bot
base_frus = clamp(sig_wajah_frus + hand_trigger_frus, 0, 1)
frus = clamp(base_frus × 0.85 + (ck_fr + jw_fr) × 0.15, 0, 1)
```

### 3. Hybrid Ratio Scoring & Sigmoid SigLIP

Sistem menggunakan metode **Late Fusion (Ensemble)** untuk menggabungkan dua model AI yang berbeda secara fundamental:
1. **MediaPipe (Landmark):** AI Geometris yang menghitung jarak dan ketegangan otot fisik secara matematis. Sangat akurat untuk posisi kepala dan kedipan mata, tetapi buta terhadap "niat" (konteks).
2. **SigLIP 2 (Vision-Language):** AI Semantik berbasis VLM (Vision-Language Model). Memahami "niat" dari bahasa tubuh dan tatapan, tetapi tidak presisi dalam menghitung derajat miringnya kepala.

Skor akhir didapatkan dengan rumus penggabungan berbobot (*Ratio Scoring*):
```python
hybrid_score = (α × sigmoid(siglip_score)) + (β × landmark_score)
```

**Kenapa menggunakan Sigmoid pada keluaran SigLIP?**
Keluaran murni dari model SigLIP berupa *logits* (skor mentah yang bisa bernilai negatif atau positif tak hingga). Agar dapat digabungkan dengan skor MediaPipe (yang memiliki batas pasti 0.0 hingga 1.0), nilai *logits* SigLIP dimasukkan ke dalam fungsi `Sigmoid(x) = 1 / (1 + e^-x)`. Ini memampatkan skor menjadi probabilitas 0% hingga 100% yang mulus, sehingga *hybrid scoring* berjalan seimbang tanpa salah satu AI mendominasi.

**Rasio Bobot Default per Label:**

| Label | α (SigLIP) | β (Landmark) | Alasan & Pembagian Tugas (*Ratio Justification*) |
|---|---|---|---|
| Boredom | 0.50 | 0.50 | MediaPipe sangat kuat mendeteksi kepala menunduk (pose), SigLIP kuat mendeteksi aura mengantuk/kosong. |
| Engagement | 0.50 | 0.50 | MediaPipe mendeteksi kepala tegak, SigLIP mendeteksi "tatapan fokus ke monitor". |
| Confusion | 0.50 | 0.50 | MediaPipe mendeteksi kerutan dahi fisik, SigLIP memvalidasi niat "sedang berpikir keras". |
| Frustration | 0.50 | 0.50 | MediaPipe mendeteksi gestur fisik ekstrem (facepalm/otot tegang), SigLIP menangkap *stress/anger*. |

---

### 4. Strategi Prompting SigLIP 2

Kualitas SigLIP sangat bergantung pada deskripsi bahasa (*Prompt Engineering*). Prompt dirancang secara spesifik untuk menghindari kata *overlap* (tumpang tindih) yang membingungkan AI.

#### Confusion Prompts
Fokus utama: Berpikir keras, bingung, tidak yakin.
*   **EN:** *"a face of a student with a thinking expression, pursed lips, and furrowed eyebrows"*
    *   **ID:** *"wajah siswa dengan ekspresi berpikir, bibir mengerucut, dan alis berkerut"*
    *   **Alasan:** Spesifik memisahkan 'berpikir' dari 'marah'.
*   **EN:** *"a face of a student scratching their head feeling puzzled and confused"*
    *   **ID:** *"wajah siswa yang menggaruk kepalanya merasa kebingungan"*
    *   **Alasan:** Secara eksplisit menyebutkan "garuk kepala" agar SigLIP bisa mem-backup MediaPipe saat tangan gagal terdeteksi utuh.

#### Frustration Prompts
Fokus utama: Stres berat, kemarahan, tekanan mental berlebihan.
*   **EN:** *"a face of a student looking extremely angry and stressed with a clenched jaw"*
    *   **ID:** *"wajah siswa yang terlihat sangat marah dan stres dengan rahang mengeras"*
    *   **Alasan:** Menghapus kata abu-abu seperti *"strained"* (tegang) yang sering disalahartikan sebagai tegang memikirkan tugas (*Confusion*). Diganti dengan *"extremely angry/stressed"* yang mutlak.
*   **EN:** *"a face of a student resting their head on their hand looking completely stressed out"*
    *   **ID:** *"wajah siswa yang menopang kepalanya di tangan terlihat sangat stres"*
    *   **Alasan:** Mendefinisikan gestur *facepalm/topang dagu* secara harfiah.

Dengan pemisahan leksikal ini, SigLIP tidak lagi kebingungan membedakan orang yang mengerutkan dahi karena mikir (*Confusion*) dan orang yang mengerutkan dahi karena frustrasi (*Frustration*).

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
