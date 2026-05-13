# Labeler Emosi SigLIP2

Aplikasi desktop untuk melabeli emosi siswa pada video secara manual maupun semi-otomatis. Menggunakan **SigLIP2** (Google VLM) dikombinasikan dengan **MediaPipe FaceLandmarker** (geometri wajah 3D) sebagai Hybrid Scoring.

---

## Daftar Isi

1. [Ciri-ciri Setiap Emosi (Panduan Anotasi)](#1-ciri-ciri-setiap-emosi-panduan-anotasi)
2. [Prasyarat & Instalasi](#2-prasyarat--instalasi)
3. [Menjalankan Aplikasi](#3-menjalankan-aplikasi)
4. [Panduan Penggunaan Aplikasi](#4-panduan-penggunaan-aplikasi)
5. [Cara Kerja Sistem Scoring](#5-cara-kerja-sistem-scoring)
6. [Konfigurasi (.env)](#6-konfigurasi-env)
7. [Struktur Kode](#7-struktur-kode)
8. [File Output](#8-file-output)
9. [FAQ & Troubleshooting](#9-faq--troubleshooting)

---

## 1. Ciri-ciri Setiap Emosi (Panduan Anotasi)

> Ini adalah bagian paling penting untuk labeler. Baca sebelum mulai anotasi.

Sistem menggunakan **4 label biner** yang bersifat **multi-label** (lebih dari satu boleh aktif bersamaan).

---

### BOREDOM (Kebosanan) — Label 0

**Kapan beri label 1 (positif)?**

| Tanda Visual | Keterangan |
|---|---|
| **Kelopak mata berat / setengah tertutup** | Mata tidak terbuka penuh, terlihat mengantuk |
| **Menguap** | Mulut terbuka lebar (tidak harus sampai 100% — sudah cukup jika terlihat menguap) |
| **Kepala noleh ke samping** | Tidak memandang ke depan/kamera, kepala miring ≥8° |
| **Tatapan mata ke samping** | Bola mata terlihat melihat ke kiri/kanan, bukan ke depan |
| **Ekspresi datar/kosong** | Tidak ada reaksi, otot wajah rileks total, tatapan hampa |
| **Kepala menengadah** | Kepala mendongak ke atas (sering disertai mata mengantuk) |

**Kapan beri label 0 (negatif)?**
- Wajah masih menghadap depan meskipun ekspresi biasa
- Mata terbuka normal meski tidak tersenyum
- Hanya sekilas noleh (bukan konsisten sepanjang frame)

**Catatan sistem:** Landmark menggunakan `gaze_dev` (deviasi gaze 2D) sebagai sinyal utama. Dead zone 5°, penuh di 25°. Menguap, mata berat, dan kepala mendongak sebagai sinyal pendukung (maks 0.70 dari nilai penuh) agar siswa yang hanya menguap tapi masih menghadap depan tidak langsung mendapat skor 1.0.

---

### ENGAGEMENT (Keterlibatan) — Label 1

**Kapan beri label 1 (positif)?**

| Tanda Visual | Keterangan |
|---|---|
| **Mata terbuka penuh menatap lurus** | Memandang ke depan/layar/kamera secara langsung |
| **Kepala tegak menghadap depan** | Yaw kepala < 10°, tidak menoleh |
| **Respons wajah aktif** | Mengangguk, alis sedikit terangkat, senyum tipis tanda tertarik |
| **Ekspresi segar dan hadir** | Terlihat sadar dan memperhatikan, bukan hanya duduk diam |

**Kapan beri label 0 (negatif)?**
- Kepala menoleh, terlihat melamun
- Mata melirik ke arah lain secara dominan
- Terlihat mengantuk atau tatapan kosong

**Catatan sistem:** Engagement menggunakan metrik `gaze_dev` yang sama dengan Boredom — keduanya **inversely related**. Semakin tinggi deviasi gaze, semakin tinggi Boredom dan semakin rendah Engagement. Gate Engagement mulai turun dari gaze_dev ≥ 5° dan mencapai nol di ≥ 17°.

---

### CONFUSION (Kebingungan) — Label 2

**Kapan beri label 1 (positif)?**

| Tanda Visual | Keterangan |
|---|---|
| **Alis berkerut/turun** | Dahi mengerut, kedua alis turun ke tengah (bukan satu sisi) |
| **Alis bagian dalam naik** | Bagian tengah alis terangkat membentuk pola "∧" atau "V terbalik" |
| **Mata menyipit / melirik ke atas** | Squinting saat berusaha memahami, atau bola mata melirik ke atas saat berpikir |
| **Mulut sedikit terbuka** | Mulut menganga tipis (BUKAN menguap lebar — batas: mulut tidak terbuka >40% penuh) |
| **Bibir mengerucut (pucker)** | Bibir sedikit maju/mengerut saat berpikir keras |
| **Kepala miring/mendongak sedikit** | Kepala miring ke samping atau sedikit mendongak saat berpikir |
| **Tangan menggaruk kepala** | Tangan di atas garis mata (area dahi/kepala) |

**Kapan beri label 0 (negatif)?**
- Ekspresi netral tanpa alis berkerut
- Wajah terlihat datar meski sedang diam
- Hanya satu ciri ringan tanpa kombinasi

**PENTING — Perbedaan Confusion vs Frustration:**
Confusion = **berpikir keras** (masih sabar, ingin mengerti). Frustration = **stres/marah** (sudah menyerah, kesal). Jika alis berkerut tapi wajah masih "tenang", itu Confusion. Jika alis berkerut sambil mengernyit hidung atau bibir ditekan keras, itu Frustration.

**Catatan sistem:** `browInnerUp` (alis dalam naik) hanya aktif jika bersamaan dengan sinyal mata (lirik atas, lookUp) atau kepala (pitch). Ini mencegah false positive pada siswa yang bentuk alisnya secara natural melengkung ke atas.

---

### FRUSTRATION (Frustrasi) — Label 3

**Kapan beri label 1 (positif)?**

| Tanda Visual | Keterangan |
|---|---|
| **Kombinasi otot tegang** | HARUS ada minimal 2 dari: alis turun keras + bibir ditekan + mata menyipit tajam + pipi menegang |
| **Mengernyit hidung (nose sneer)** | Hidung berkerut seperti ekspresi jijik/marah — sinyal terkuat, satu ini sudah cukup |
| **Mata tertutup rapat/dipaksa** | Bukan mengantuk — tapi menutup mata karena frustrasi (eye squint tajam) |
| **Bibir ditekan keras (lip press)** | Bibir merapatkan bersama secara kuat, bukan sekedar mulut tertutup biasa |
| **Tangan menutupi wajah** | Facepalm, tangan di pipi/hidung/mulut, atau menopang kepala dengan telapak tangan |
| **Ekspresi marah/stres ekstrem** | Bukan hanya "berpikir", tapi terlihat kesal, mau menangis, atau sangat lelah mental |

**Kapan beri label 0 (negatif)?**
- Hanya satu otot yang sedikit tegang (misalnya hanya bibir rapat tapi wajah lainnya rileks)
- Alis berkerut tapi tidak ada otot lain yang tegang → itu Confusion, bukan Frustration
- Tangan menyentuh wajah tapi bukan facepalm/menutupi (misalnya hanya memegang pipi santai)

**Catatan sistem:** Frustration menggunakan **SUM logic** (bukan MAX) — satu sinyal saja tidak cukup. Kecuali `noseSneer` (mengernyit hidung) yang sangat jarang terjadi secara natural, sehingga satu signal ini sudah dianggap kuat.

---

### Bolehkah Multi-Label?

**Ya.** Contoh kombinasi yang valid:
- `Confusion=1, Engagement=1` — siswa memperhatikan tapi tampak tidak mengerti
- `Boredom=1, Confusion=0` — siswa noleh, tidak bingung, hanya tidak tertarik
- `Confusion=1, Frustration=1` — siswa sangat bingung sampai terlihat stres
- **Tidak lazim:** `Boredom=1, Engagement=1` (kontradiktif, hindari kecuali frame ambigu)

---

## 2. Prasyarat & Instalasi

**Persyaratan sistem:**
- Python 3.9 atau lebih baru
- GPU NVIDIA (opsional — inferensi juga berjalan di CPU, lebih lambat)

```bash
pip install -r requirements.txt
```

Model SigLIP2 (~400 MB) dan MediaPipe FaceLandmarker (~30 MB) diunduh otomatis saat pertama dijalankan.

---

## 3. Menjalankan Aplikasi

```bash
# Salin dan sesuaikan konfigurasi (opsional)
cp .env.example .env

# Jalankan aplikasi utama
python app.py
```

---

## 4. Panduan Penggunaan Aplikasi

### Membuka Dataset

Klik **Buka Folder** → pilih folder root yang berisi file `.mp4`. Aplikasi akan:
1. Mencari semua file video secara rekursif
2. Membuat folder `hasil_label6/` sebagai direktori output
3. Memuat ulang label yang sudah tersimpan

### Memutar Video

Video diputar otomatis saat dimuat. Klik area video untuk pause/play. Di bawah video terdapat **4 thumbnail** (2×2) yang mewakili 4 titik terdistribusi merata sepanjang video. FPS yang terdeteksi ditampilkan di status bar; jika codec melaporkan nilai tidak valid (> 60 atau ≤ 0), aplikasi otomatis mengestimasi FPS dari timestamp frame.

### Labeling Per-Frame (Manual)

1. Pilih tab label aktif di atas galeri (misal `Confusion`)
2. **Klik kiri pada thumbnail** untuk seek ke posisi frame tersebut
3. **Klik kanan pada thumbnail** untuk toggle label aktif pada frame tersebut
   - Border berwarna = positif (label = 1)
   - Border abu-abu = negatif (label = 0)
4. **Double-klik pada thumbnail** untuk menandai frame sebagai "ditolak" (overlay merah)
5. Jika **≥2 dari 4 frame** positif → label video = 1

### Menggunakan AI (Hybrid Scoring)

- **Proses Video Ini** — jalankan inferensi pada video aktif
- **Batch Semua** — inferensi seluruh dataset di background thread

Bar di kanan tiap label menunjukkan **Hybrid Score** (0.0–1.0). Video yang sudah diproses otomatis di-skip saat batch ulang (berdasarkan `batch_history.json`).

### Toggle Landmark Viz

Aktifkan switch **Viz** di topbar untuk melihat overlay landmark pada thumbnail: face mesh 3D tesselation (putih tipis), skeleton tangan (hijau), per-eye gaze arrow (kuning), blendshape signal bars (kanan), dan emotion score bars (bawah). Berguna untuk debugging mengapa AI memberi skor tertentu.

### Reset Label Video Ini

Tombol **Reset Label Video Ini** (merah) di panel kanan menghapus semua frame annotations dan riwayat AI untuk video aktif, lalu meresetnya ke nol tanpa konfirmasi. Gunakan jika hasil AI perlu diulang dari awal.

### Flag / Reject

Toggle **Flag/Reject** untuk menandai video yang tidak layak dilabeli (kualitas buruk, wajah tidak jelas). Video terflag tidak dimasukkan ke dataset training.

---

## 5. Cara Kerja Sistem Scoring

### Hybrid Score Formula

```
hybrid_score[frame] = α × siglip_score[frame] + β × landmark_score[frame]
label_video = 1  jika avg(hybrid_score) ≥ threshold
```

### SigLIP Scoring (Visual)

SigLIP2 membandingkan gambar wajah dengan prompt teks per label:

```
logit[frame, prompt] → sigmoid(logit + 3.5) → avg per label → siglip_score
```

Bias `+3.5` diperlukan karena logit zero-shot SigLIP sering bernilai negatif; bias ini menggeser kurva ke area sensitivitas 0.2–0.9.

### Landmark Scoring (Geometri Wajah MediaPipe)

Setiap frame dianalisis dengan FaceLandmarker (478 titik) + HandLandmarker. Output per frame: skor 0.0–1.0 per emosi.

#### Boredom & Engagement — Unified `gaze_dev`

Kedua emosi ini diturunkan dari satu metrik tunggal `gaze_dev` (deviasi angular 2D dari arah kamera), sehingga keduanya **inversely related** — tidak bisa tinggi bersamaan.

```
# Komponen horizontal — direction-aware (iris berlawanan yaw = kompensasi)
gaze_h    = yaw + iris_x × 35

# Komponen vertikal — dengan fallback bentuk kelopak (eyeLookDown)
gaze_v    = max(|-pitch + iris_y × 25|, eyeLookDown × 40)
gaze_v_eff = max(0, gaze_v - 15°)    # dead zone 15° untuk squinting & layar di bawah mata

# Tiga floor untuk horizontal — mencegah kompensasi penuh
iris_side  = |iris_x| × 35 × 2.0    # iris_x=0.2 → 14°, iris_x=0.3 → 21°
gaze_h_eff = max(|gaze_h|, iris_side, |yaw|)

gaze_dev   = sqrt(gaze_h_eff² + gaze_v_eff²)

# BOREDOM: dead zone 5°, penuh di 25°
bore_gaze  = clamp((gaze_dev - 5) / 20, 0, 1)

blink_v    = clamp((max(eyeBlinkL, eyeBlinkR) - 0.20) / 0.50, 0, 1)  # dead zone 0.20
yawn_v     = clamp(jawOpen / 0.35, 0, 1)   # aktif jika pitch < 15°
pitch_up_v = clamp((pitch - 20°) / 25, 0, 1)
sig_expr   = max(blink_v, yawn_v, pitch_up_v) × 0.70

bore = clamp(max(bore_gaze, sig_expr) × 0.85 + (bore_gaze + sig_expr) × 0.15, 0, 1)

# ENGAGEMENT: metrik gaze_dev yang sama, inversely related
gate        = clamp(1 - max(0, gaze_dev - 5) / 12, 0, 1)  # 0 di ≥17°
blink_heavy = max(0, max(eyeBlinkL, eyeBlinkR) - 0.50) / 0.50  # hanya >0.50 = droopy
eng         = gate × max(0.30, 1.0 - blink_heavy)
```

> **Kenapa tiga floor untuk `gaze_h_eff`?** (1) `gaze_h` memungkinkan iris mengkompensasi yaw — wajar jika kepala miring dan mata melihat balik ke kamera. (2) `iris_side` mencegah kompensasi penuh: iris yang jelas ke samping tetap berkontribusi walau kepala sedikit mengimbangi. (3) `|yaw|` sebagai floor minimum — kepala miring = gaze_dev minimal sebesar sudut itu, tidak bisa nol.

#### Confusion
```
iris_up_v  = clamp((-iris_y - 0.15) / 0.30, 0, 1)   # pupil ke atas (dead zone 0.15)
look_up_v  = clamp(max(lookUpL, lookUpR) / 0.35, 0, 1)
pitch_cu   = clamp((pitch - 5°) / 15, 0, 1)          # mendongak mulai 5°
brow_dn_v  = clamp(mean(browDownL, browDownR) / 0.23, 0, 1)
co_signal  = max(iris_up_v, look_up_v, pitch_cu)     # sinyal penguat untuk browInnerUp
brow_in_v  = clamp(browInnerUp / 0.30, 0, 1) × clamp(co_signal / 0.25, 0, 1)

jo = jawOpen
jaw_co = bell(0.05–0.25 puncak, 0.40 batas atas)    # mangap sedikit, bukan menguap
pucker_co = clamp(mouthPucker / 0.30, 0, 1)

base_conf = max(brow_dn_v, brow_in_v, iris_up_v, look_up_v, jaw_co, pucker_co, hand_chin)
conf = clamp(base_conf × 0.85 + (pitch_cu + brow_signal) × 0.15, 0, 1)

# Suppression: tangan menutupi wajah (bukan garuk kepala) → kurangi skor confusion
suppression = hand_forehead if hand_chin < 0.5 else 0.0
conf = clamp(conf - suppression, 0, 1)
```

> **Kenapa `browInnerUp` butuh co_signal?** Beberapa siswa secara natural punya alis melengkung (browInnerUp = 1.0 terus). Tanpa co_signal, confusion akan selalu tinggi meski tidak ada ekspresi bingung. Co_signal memastikan ada bukti lain dari mata/iris/kepala sebelum browInnerUp dihitung.

#### Frustration
```
br_fr = clamp(mean(browDownL, browDownR) / 0.40, 0, 1)
ns_fr = clamp(max(noseSneerL, noseSneerR) / 0.20, 0, 1)  # sinyal terkuat
ck_fr = clamp(mean(cheekSquintL, cheekSquintR) / 0.40, 0, 1)
lp_fr = clamp(mean(mouthPressL, mouthPressR) / 0.40, 0, 1)
ey_fr = clamp(mean(eyeSquintL, eyeSquintR) / 0.40, 0, 1)

# SUM logic: noseSneer kuat + kombinasi otot lain
sig_wajah_frus = clamp(ns_fr + (br_fr + lp_fr + ey_fr + ck_fr) / 2.0, 0, 1)

frus = clamp((sig_wajah_frus + hand_forehead) × 0.85 + (cheek + jaw) × 0.15, 0, 1)
```

### Zonasi Tangan

Dalam crop wajah 512×512:

| Zona | Posisi Y di Crop | Interpretasi | Efek |
|---|---|---|---|
| Atas (`y < 0.25`) | Di atas mata / forehead area | Menggaruk kepala | +Confusion |
| Tengah & Bawah (`y 0.25–1.20`) | Menutup wajah / menopang dagu | Facepalm / stres | +Frustration, −Confusion |

### Bobot Hybrid per Label

Nilai saat ini di `.env` (dapat diubah):

| Label | α (SigLIP) | β (Landmark) | Alasan |
|---|---|---|---|
| Boredom | 0.50 | 0.50 | Landmark baik untuk yaw/iris; SigLIP baik untuk ekspresi lelah/kosong |
| Engagement | 0.50 | 0.50 | Gate logic Landmark sangat presisi; SigLIP mendukung konteks visual |
| Confusion | **0.60** | **0.40** | SigLIP lebih baik menangkap ekspresi berpikir halus yang sulit dikuantifikasi |
| Frustration | 0.50 | 0.50 | Seimbang: Landmark untuk gestur tangan, SigLIP untuk aura stres |

### Temporal Restlessness Bonus (Khusus Boredom)

Jika std deviasi yaw kepala ≥3° di 4 frame (kepala bergerak bolak-balik), skor Boredom mendapat bonus maksimum +0.15. Ini menangkap pola tolah-toleh yang tidak terlihat dari satu frame saja.

---

## 6. Konfigurasi (.env)

```env
# Model SigLIP2
SIGLIP_MODEL_ID=google/siglip2-base-patch16-224

# Folder output (relatif terhadap folder dataset)
OUTPUT_DIR=hasil_label6

# Padding crop wajah (0.20 = ketat, 0.40 = longgar)
FACE_CROP_PADDING=0.30

# Bobot hybrid GLOBAL (fallback jika per-label tidak diset)
SIGLIP_WEIGHT=0.50
LANDMARK_WEIGHT=0.50

# Override per-label
BOREDOM_SIGLIP_WEIGHT=0.50
BOREDOM_LANDMARK_WEIGHT=0.50
ENGAGEMENT_SIGLIP_WEIGHT=0.50
ENGAGEMENT_LANDMARK_WEIGHT=0.50
CONFUSION_SIGLIP_WEIGHT=0.60
CONFUSION_LANDMARK_WEIGHT=0.40
FRUSTRATION_SIGLIP_WEIGHT=0.50
FRUSTRATION_LANDMARK_WEIGHT=0.50

# Auto-flag jika frame dengan >1 wajah melebihi threshold ini (default 1)
MULTI_FACE_FRAMES_THRESHOLD=1
```

---

## 7. Struktur Kode

```
Labeler-App-Siglip-2/
├── app.py                  # Entry point — GUI & event orchestration
├── ai_service.py           # Headless REST mode (opsional)
├── requirements.txt
├── .env / .env.example
│
├── core/
│   ├── siglip_model.py     # Singleton loader SigLIP2 (lazy load GPU/CPU)
│   ├── inference.py        # Hybrid scoring: SigLIP × Landmark fusion
│   ├── landmark_analyzer.py # MediaPipe: head pose, iris, blendshapes, hand zones
│   ├── face_detector.py    # BlazeFace crop wajah + return bbox
│   └── README_SIGLIP.md    # Dokumentasi teknis pipeline SigLIP
│
├── ui/
│   ├── constants.py        # LABELS, LABEL_COLORS, DEFAULT_PROMPT_GROUPS
│   ├── left_panel.py       # Panel kiri: video player + galeri 6 frame
│   └── right_panel.py      # Panel kanan: AI score bars, prompt editor, threshold
│
└── utils/
    ├── io.py               # Baca/tulis CSV/JSON annotations
    └── video.py            # Ekstraksi frame + pipeline crop + landmark analysis
```

---

## 8. File Output

Semua output tersimpan di `hasil_label6/` (atau nilai `OUTPUT_DIR` di `.env`):

| File | Format | Isi |
|---|---|---|
| `annotations_bener.csv` | CSV | Label per video (Boredom/Engagement/Confusion/Frustration: 0 atau 1) |
| `frame_annotations.json` | JSON | Label per frame per video (6 frame × 4 label) |
| `batch_history.json` | JSON | Riwayat skor AI: avg_score, siglip_avg, landmark_avg, frame_scores |
| `flagged_videos.csv` | CSV | Daftar video yang di-flag/reject |
| `skipped_videos.json` | JSON | Daftar video yang di-skip |
| `thresholds.json` | JSON | Nilai threshold slider yang terakhir disimpan |
| `cropped_faces/clean/` | JPG | Crop wajah bersih untuk training |
| `cropped_faces/viz/` | JPG | Crop wajah dengan overlay landmark untuk debugging |

---

## 9. FAQ & Troubleshooting

**Q: Kenapa AI skor Confusion selalu rendah?**
A: Pastikan threshold slider Confusion tidak terlalu tinggi (coba turunkan ke 0.40). Skor confusion bergantung pada ekspresi alis dan mata — jika wajah siswa terlihat bingung tapi skor tetap rendah, pertimbangkan untuk melabeli manual.

**Q: Kenapa AI tidak mendeteksi siswa menguap sebagai Boredom?**
A: Sistem mendeteksi menguap (`jawOpen > 0.35`) sebagai sinyal bosan kuat. Namun jika kepala siswa mendongak terlalu jauh (pitch > 15°) saat menguap, atau mulut tidak terbuka cukup lebar, sinyal bisa lemah. Gunakan viz mode untuk cek nilai signal `Yawn`.

**Q: Confusion dan Frustration terdeteksi bersamaan — apakah normal?**
A: Ya, bisa terjadi. Siswa yang sangat bingung sampai stres bisa memiliki kedua emosi aktif. Namun jika sering bersamaan padahal wajah tidak menunjukkan tanda stres, turunkan threshold Frustration.

**Q: Kenapa video di-flag otomatis?**
A: Tiga kondisi auto-flag:
1. Wajah tidak terdeteksi di **semua** 4 frame (kualitas video/pencahayaan buruk)
2. Video terlalu pendek (<4 frame yang bisa diekstrak)
3. Lebih dari 1 frame berisi >1 wajah — bisa diubah via `MULTI_FACE_FRAMES_THRESHOLD` di `.env`

**Q: Berapa lama proses Batch AI?**
A: ~3–8 detik per video dengan GPU, ~15–30 detik per video di CPU.

**Q: Apakah bisa mengubah prompt SigLIP?**
A: Ya. Klik header label di panel kanan untuk expand, lalu edit prompt di textbox. Perubahan langsung berlaku untuk inferensi berikutnya (tidak perlu restart).

**Q: Apa itu `batch_history.json` dan kapan harus di-reset?**
A: Menyimpan skor AI dari setiap video yang sudah diproses. Video yang sudah ada di history akan di-skip saat Batch Semua. Klik **Restart Batch** di panel kanan untuk menghapus history dan memproses ulang semua video dari awal.
