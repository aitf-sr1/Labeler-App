# Labeler Emosi SigLIP2

Aplikasi desktop untuk melabeli emosi siswa pada video secara manual maupun semi-otomatis. Menggunakan **SigLIP2** (Google VLM) dikombinasikan dengan **MediaPipe FaceLandmarker** (geometri wajah 3D) sebagai Hybrid Scoring.

---

## Daftar Isi

1. [Panduan Anotasi — Ciri-ciri Setiap Emosi](#1-panduan-anotasi--ciri-ciri-setiap-emosi)
2. [Prasyarat & Instalasi](#2-prasyarat--instalasi)
3. [Menjalankan Aplikasi](#3-menjalankan-aplikasi)
4. [Panduan Penggunaan UI](#4-panduan-penggunaan-ui)
5. [Cara Kerja Sistem Scoring (Ringkasan)](#5-cara-kerja-sistem-scoring-ringkasan)
6. [Sistem Cache](#6-sistem-cache)
7. [Konfigurasi (.env)](#7-konfigurasi-env)
8. [Struktur Kode](#8-struktur-kode)
9. [File Output](#9-file-output)
10. [FAQ & Troubleshooting](#10-faq--troubleshooting)

### Dokumentasi Lanjutan

| Dokumen | Isi |
|---|---|
| [docs/COMPUTATION.md](docs/COMPUTATION.md) | Panduan teknis lengkap: dari angka mentah MediaPipe → rumus per-emosi → SigLIP → hybrid → prediksi. Dilengkapi contoh angka nyata dan referensi akademis. |
| [docs/RULES_PANEL.md](docs/RULES_PANEL.md) | Cara menggunakan Rules Editor: semua ~50 parameter dijelaskan, panduan "kalau label X bermasalah ubah ini", strategi tuning. |
| [core/README_SIGLIP.md](core/README_SIGLIP.md) | Arsitektur pipeline inferensi: SigLIP2, MediaPipe, sistem cache, recalculate, viz regeneration. |

---

## 1. Panduan Anotasi — Ciri-ciri Setiap Emosi

> Bagian paling penting untuk labeler. Baca sebelum mulai anotasi.

Sistem menggunakan **4 label biner** yang bersifat **multi-label** (lebih dari satu bisa aktif bersamaan).

---

### BOREDOM (Kebosanan) — Label 0

**Definisi:** Siswa tidak memperhatikan konten — tatapan tidak ke layar/materi, atau ada tanda kelelahan mental yang jelas (mengantuk, menguap).

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Kepala noleh ke samping | Tidak menghadap layar/kamera, kepala berputar kiri/kanan |
| Tatapan mata ke samping | Bola mata melihat ke kiri/kanan jauh dari depan |
| Tatapan mata ke kejauhan / kosong | Staring tanpa fokus, "menerawang" |
| Kelopak mata berat / setengah tertutup | Mata tidak terbuka penuh, terlihat mengantuk |
| Menguap | Mulut terbuka lebar karena mengantuk — **hanya valid jika bersamaan dengan gaze deviated atau tidak ke layar** |
| Kepala menengadah | Kepala mendongak ke atas (sering disertai mata mengantuk) |
| Kepala tertunduk ke bawah tanpa ekspresi fokus | Tertunduk pasif, bukan karena ngetik/membaca — ekspresi kosong |

**Kapan beri label 0?**
- Wajah menghadap depan (ke layar) meski ekspresi biasa/datar
- Mata terbuka normal meski tidak tersenyum
- Menguap sekilas tapi langsung kembali menatap layar
- Mata ke bawah karena ngetik atau membaca catatan (itu Engagement, bukan Boredom)

**Catatan sistem:** Signal utama adalah `gaze_dev` (deviasi angular dari arah kamera). Menguap dan mata berat hanya dihitung **jika gaze sudah deviated** atau siswa sedang melirik ke bawah (ngetik). Jika seseorang menguap tapi masih menatap lurus ke layar, sistem tidak langsung beri skor Boredom tinggi. Ini sengaja — menguap sambil tetap fokus bukan boredom murni.

---

### ENGAGEMENT (Keterlibatan) — Label 1

**Definisi:** Siswa aktif memperhatikan konten — tatapan ke layar/materi, atau secara aktif mengerjakan tugas (ngetik, menulis).

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Menatap langsung ke layar/kamera | Tatapan ke depan, mata terbuka, kepala menghadap layar |
| Mata ke bawah sambil ngetik/menulis | Kepala atau mata mengarah ke bawah, **tetapi ekspresi terlihat fokus dan aktif** — bukan kosong |
| Kepala tegak menghadap depan | Yaw kepala kecil, tidak menoleh ke samping |
| Respons wajah aktif | Mengangguk, alis sedikit terangkat, senyum tipis tanda tertarik |
| Ekspresi segar dan hadir | Terlihat sadar dan memperhatikan, bukan duduk pasif atau melamun |
| Kepala sedikit miring tapi mata mengikuti layar | Head pose bukan frontal sempurna, tapi arah pandang jelas ke konten |

**Kapan beri label 0?**
- Kepala menoleh jauh ke samping, terlihat melamun
- Mata melirik ke arah lain secara dominan dan konsisten
- Terlihat mengantuk atau tatapan kosong
- Tertunduk pasif tanpa ekspresi fokus

**Boredom dan Engagement boleh bersamaan (multi-label = 1) hanya jika:** siswa terlihat **ngetik atau membaca catatan ke bawah sambil ada tanda mengantuk** (mata berat, kelopak turun). Ini satu-satunya kondisi wajar keduanya aktif bersamaan. Selain itu, keduanya mestinya bertentangan.

**Catatan sistem:** Engagement diukur dari gaze_dev yang rendah (kepala/iris menghadap ke depan). Jika `lookDown` (blendshape mata lirik ke bawah) tinggi, sistem menginterpretasikan sebagai "ngetik" dan memberi skor Engagement tinggi — selama iris masih cukup centered dan tidak ada tanda disengaged lain.

---

### CONFUSION (Kebingungan) — Label 2

**Definisi:** Siswa sedang memproses informasi yang sulit secara kognitif — masih **attentive dan fokus ke konten**, tapi ekspresi menunjukkan kebingungan atau usaha berpikir keras.

**PENTING — Confusion adalah emosi MURNI WAJAH. Tidak ada keterlibatan tangan.**

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Alis berkerut/turun ke tengah | Dahi mengerut, kedua alis turun — tanda berusaha memahami |
| Alis bagian dalam naik (browInnerUp) | Bagian tengah alis terangkat membentuk pola "∧" saat bingung |
| Mata menyipit saat berpikir | Squinting dengan usaha, bukan mengantuk |
| Mata melirik ke atas | Bola mata bergerak ke atas — tanda sedang mengakses memori/berpikir |
| Mulut sedikit terbuka / mangap tipis | BUKAN menguap lebar — mangap sedikit saat sedang berpikir keras |
| Bibir mengerucut (pucker) | Bibir sedikit maju/mengerut seperti ekspresi "hmm..." |
| Kepala sedikit mendongak | Kepala mendongak saat berpikir keras (pitch naik) |

**Kapan beri label 0?**
- Ekspresi netral tanpa alis berkerut sama sekali
- Wajah datar meski sedang diam
- Hanya satu ciri ringan tanpa kombinasi sinyal lain
- **Ada tangan dekat wajah/kepala → kemungkinan Frustration, bukan Confusion**
- Gaze sangat deviated (noleh jauh) — orang bingung masih fokus ke konten, bukan noleh

**PENTING — Perbedaan Confusion vs Frustration:**

| | Confusion | Frustration |
|---|---|---|
| Ekspresi | Berpikir keras, masih sabar | Stres, kesal, mau menyerah |
| Alis | Berkerut, mungkin bagian dalam naik | Berkerut keras + hidung mengernyit |
| Mulut | Sedikit terbuka / mangap tipis | Bibir ditekan rapat atau rahang tegang |
| Tangan | **Tidak ada** | Tangan di dahi, pipi, menutupi wajah |
| Gaze | Masih ke arah layar/konten | Bisa ke mana saja |

**Catatan sistem:** `browInnerUp` (alis dalam naik) hanya aktif jika bersamaan dengan sinyal mata (lirik atas, lookUp) atau pitch kepala — mencegah false positive pada siswa yang alis-nya secara alami melengkung ke atas. Confusion juga di-gate oleh `attentive_gate`: jika gaze sangat deviated (gaze_dev > 28°), skor Confusion otomatis turun signifikan karena "orang yang bingung masih fokus ke layar". Selain itu, **jika tangan terdeteksi dekat kepala, skor Confusion langsung ditekan** — itu sinyal Frustration.

---

### FRUSTRATION (Frustrasi) — Label 3

**Definisi:** Siswa mengalami tekanan emosional intens — kesal, stres, hampir menyerah, atau sangat lelah secara mental. Sering disertai gestur tangan ke kepala/wajah.

**PENTING — Frustration adalah satu-satunya emosi yang menggunakan sinyal tangan.**

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| **Tangan di dahi / kepala** | Facepalm, memegang kepala, tangan di atas mata — sinyal terkuat |
| **Tangan menutupi wajah** | Menutup pipi, hidung, mulut, atau seluruh wajah |
| **Tangan menopang kepala** | Siku di meja, tangan memegang kepala dari samping atau depan |
| Kombinasi otot tegang | MINIMAL 2 dari: alis turun keras + bibir ditekan rapat + mata menyipit tajam + pipi menegang |
| Mengernyit hidung (nose sneer) | Hidung berkerut seperti ekspresi jijik/marah — salah satu sinyal wajah terkuat |
| Mata tertutup rapat/dipaksa | Bukan mengantuk — menutup mata karena frustrasi (eye squint tajam dan tegang) |
| Bibir ditekan keras (lip press) | Kedua bibir merapatkan secara kuat, garis bibir sangat tipis |
| Rahang tegang (jaw clench) | Rahang terlihat tegang, gigi mengatup |

**Kapan beri label 0?**
- Hanya satu otot wajah yang sedikit tegang tanpa kombinasi
- Alis berkerut tapi tidak ada otot lain tegang dan tidak ada tangan → itu Confusion
- Tangan menyentuh wajah secara sekilas dan santai (bukan facepalm/menopang)
- Tampak hanya "berpikir" tanpa tanda stres

**Catatan sistem:** Formula Frustration menggunakan **weighted-max fusion** — bukan penjumlahan aditif. Tangan (hand_weight=0.65) adalah sinyal primer: `base = max(hand×0.65 + face×0.35, hand, face)`. Satu sinyal lemah saja tidak menginflasi skor. Tangan dari zona manapun dekat kepala (dahi, pipi, dagu, menutupi wajah) menghitung sebagai trigger Frustration. Jika tangan terdeteksi, **skor Confusion otomatis ditekan** untuk mencegah kebingungan antar label.

---

### Bolehkah Multi-Label?

**Ya.** Contoh kombinasi valid:

| Kombinasi | Kapan valid |
|---|---|
| `Confusion=1, Engagement=1` | Siswa memperhatikan tapi tampak tidak mengerti — yang paling umum |
| `Boredom=1, Engagement=0` | Siswa jelas tidak memperhatikan |
| `Confusion=1, Frustration=1` | Siswa sangat bingung sampai mulai stres — ada ekspresi tegang + tanda bingung |
| `Boredom=1, Engagement=1` | **Hanya valid** jika siswa ngetik/membaca ke bawah sambil ada tanda mengantuk (mata berat). Selain kondisi ini, hindari — keduanya bertentangan. |
| `Frustration=1, Engagement=0` | Siswa stres sampai tidak fokus ke konten |

**Hindari:**
- `Confusion=1` jika ada tangan di wajah → pakai `Frustration=1`
- `Boredom=1, Engagement=1` jika tidak ada konteks ngetik/membaca

---

## 2. Prasyarat & Instalasi

**Persyaratan sistem:**
- Python 3.9 atau lebih baru
- GPU NVIDIA (opsional — inferensi juga berjalan di CPU, lebih lambat)

```bash
pip install -r requirements.txt
pip install customtkinter   # UI framework (tidak ada di requirements.txt)
```

Model yang diunduh otomatis saat pertama kali dijalankan:
- SigLIP2 `google/siglip2-base-patch16-224` (~400 MB) via HuggingFace
- MediaPipe FaceLandmarker + HandLandmarker (~50 MB)

---

## 3. Menjalankan Aplikasi

```bash
# Salin dan sesuaikan konfigurasi (opsional)
cp .env.example .env

# Jalankan aplikasi
python app.py
```

---

## 4. Panduan Penggunaan UI

### Membuka Dataset

Klik **Buka Folder** di topbar → pilih folder root yang berisi file `.mp4`. Aplikasi akan:
1. Mencari semua file video secara rekursif
2. Membuat folder output di `{folder_dataset}/{OUTPUT_DIR}/` (default: `hasil_label7/`)
3. Membuat sub-folder `raw_cache/` dan `siglip_cache/` untuk sistem cache
4. Memuat ulang semua data yang sudah tersimpan (annotations, batch history, rules)

### Memutar Video

Video diputar otomatis saat dimuat. Klik area video untuk pause/play. Di bawah video terdapat **galeri frame** yang mewakili titik-titik terdistribusi merata sepanjang video.

### Labeling Per-Frame (Manual)

1. Pilih tab label aktif di atas galeri (misal `Confusion`)
2. **Klik kiri pada thumbnail** → seek video ke posisi frame tersebut
3. **Klik kanan pada thumbnail** → toggle label aktif pada frame tersebut
   - Border berwarna = positif (label = 1)
   - Border abu-abu = negatif (label = 0)
4. **Double-klik pada thumbnail** → tandai frame sebagai "ditolak" (overlay merah)
   - Frame ditolak tidak dihitung dalam prediksi label video
5. Prediksi akhir video berdasarkan **avg_score vs threshold** (bukan majority vote)

### AI Inferensi — Proses Satu Video

Klik **Proses Video Ini** di panel kanan. Proses:
1. Ekstrak & crop wajah dari frame video aktif
2. Analisis landmark MediaPipe per frame (head pose, iris, blendshapes, tangan)
3. Scoring SigLIP2 zero-shot per frame × per prompt
4. Gabungkan ke Hybrid Score
5. Update AI score bars di panel kanan
6. Simpan ke `batch_history.json`, `frame_annotations.json`, dan cache

### AI Inferensi — Batch Semua

Klik **Batch Semua** untuk memproses seluruh dataset di background thread. Video yang sudah ada di `batch_history.json` di-skip otomatis. Status progres tampil di label kecil di bawah tombol.

Klik **Restart Batch** untuk menghapus history dan memproses ulang semua video dari awal.

Klik **Batch Semua** lagi saat berjalan untuk **membatalkan** batch (tombol berubah menjadi "Batalkan").

### Batch Versioning

Di panel kanan, sebelum klik Proses Video Ini atau setelah recalculate:

- **Checkbox "Buat batch baru"** — jika dicentang, hasil batch juga disimpan ke file terpisah (selain `batch_history.json` utama yang selalu ditimpa)
- **Field nama** — isi nama untuk file baru (misal `batch_v2_rules_adjusted`)
  - File batch disimpan sebagai `{nama}.json`
  - File split 2D disimpan ke folder `Label2d/{nama}/` dengan nama yang sama

Jika tidak dicentang, semua hasil hanya ditimpa ke file utama (`batch_history.json` dan `annotations_bener.csv`).

### Toggle Viz Landmark

Aktifkan switch **Viz** di topbar untuk melihat overlay landmark pada thumbnail:
- Face mesh 3D tesselation (garis abu-abu tipis)
- Skeleton tangan (hijau)
- Lingkaran iris per mata (cyan)
- Panah gaze per mata (kuning)
- Signal bars blendshape (sisi kanan)
- Emotion score bars (bawah)

Berguna untuk debugging mengapa AI memberi skor tertentu pada frame tertentu.

### Rules Editor

Klik tombol **Rules** di topbar untuk membuka jendela Rules Editor. Jendela ini memungkinkan pengaturan ulang semua parameter scoring tanpa edit kode.

**Cara pakai:**
- Setiap baris memiliki **slider** (drag kasar) dan **input field** (ketik angka pasti)
- Ketik angka di field → tekan **Enter** atau klik area lain → nilai di-clamp ke range valid
- Klik **Reset Default** untuk mengembalikan semua parameter ke nilai awal
- Klik **Simpan** untuk menyimpan ke `rules.json`
- Klik **Recalculate Batch** untuk menghitung ulang seluruh batch dari cache dengan parameter baru

### Recalculate Batch

Tersedia di Rules Editor. Menghitung ulang prediksi untuk **semua video yang sudah pernah diproses**, menggunakan:
- Raw feature cache (blendshapes, head pose, iris) dari `raw_cache/`
- SigLIP score cache dari `siglip_cache/`
- Parameter rules dan threshold terkini

**Catatan penting:** Setelah mengubah `rules.json` atau `thresholds.json`, **selalu jalankan Recalculate Batch** agar semua video di-recompute secara konsisten dengan parameter baru. Tanpa ini, video lama masih memakai skor dari parameter yang berbeda.

Setelah recalculate:
- `batch_history.json` diperbarui
- `frame_annotations.json` diperbarui
- Viz thumbnail video aktif di-regenerate otomatis

### Split Label 2D

Klik **Split Label 2D** di panel kanan untuk membagi dataset menjadi train/val/test berdasarkan identitas siswa (UUID dari path).

- Splitting dilakukan per **siswa** (bukan per video) — tidak ada kebocoran data antar split
- Rasio split: **80% train / 10% val / 10% test**
- Output: 3 file CSV di folder `Label2d/{nama_batch}/`

### Reset & Flag

- **Reset Label Video Ini** (merah) — hapus semua frame annotations dan riwayat AI untuk video aktif
- **Flag/Reject** di topbar — tandai video sebagai tidak layak dilabeli (kualitas buruk, wajah tidak jelas)

### Keyboard Shortcut

| Tombol | Fungsi |
|---|---|
| `←` / `→` | Video sebelumnya / berikutnya |
| `Space` | Pause / play video |
| `S` | Simpan dan lanjut ke video berikutnya |

---

## 5. Cara Kerja Sistem Scoring (Ringkasan)

Sistem menggunakan **Hybrid Scoring**: menggabungkan dua sumber sinyal per frame, lalu rata-rata dari seluruh frame untuk keputusan akhir.

```
                    ┌─ SigLIP2 (visual)  ─────────────────────────────┐
Video → N frame →  │  logit → sigmoid(logit + bias) → siglip_score    │
                    │                                                   ├─► hybrid = siglip_w × siglip + land_w × land
                    └─ MediaPipe (geometri) ──────────────────────────┘
                       yaw, pitch, iris, blendshapes, tangan → landmark_score

avg(hybrid_score semua frame) → vs threshold → prediction video (0/1)
```

### Bobot Hybrid per Label (rules.json)

| Label | siglip_w | land_w | Alasan |
|---|---|---|---|
| Boredom | 0.30 | 0.70 | Gaze MediaPipe sangat reliable untuk deteksi noleh |
| Engagement | 0.20 | 0.80 | Gaze MediaPipe sangat reliable, dominan landmark |
| Confusion | 0.85 | 0.15 | SigLIP lebih andal baca ekspresi wajah halus |
| Frustration | 0.50 | 0.50 | Balanced: SigLIP baca ekspresi, landmark deteksi tangan |

### Formula Landmark per Emosi

**Boredom:**
- Signal utama: `gaze_dev` (deviasi angular 2D kepala+iris dari kamera)
- Signal pendukung: mata berat (eyeBlink), menguap (jawOpen), kepala mendongak
- **expr_gate:** signal pendukung (blink/yawn) hanya aktif jika `gaze_dev > threshold` ATAU mata melirik ke bawah (ngetik). Jika gaze lurus ke depan, ekspresi saja tidak cukup trigger Boredom.

**Engagement:**
- Inversely proportional dengan gaze_dev — semakin tinggi deviasi gaze, semakin rendah Engagement
- Dikurangi jika ada droopy eyes parah (eyeBlink > 0.5)

**Confusion:**
- **Murni ekspresi wajah** — alis berkerut (`browDown`), alis dalam naik (`browInnerUp`), iris melirik atas, mulut mangap tipis, bibir mengerucut
- **attentive_gate:** di-gate oleh gaze_dev — confusion menurun signifikan jika gaze sangat deviated (orang bingung masih fokus ke konten)
- **Hand suppression:** jika tangan terdeteksi dekat kepala (zona manapun), skor Confusion ditekan langsung

**Frustration:**
- Sinyal tangan (zona dahi ATAU pipi/dagu) sebagai **primer** (weight=0.65)
- Sinyal wajah (nose sneer, brow down, mouth press, eye squint, cheek squint) sebagai **suplemen** (weight=0.35)
- Formula: `base = max(hand×0.65 + face×0.35, hand_alone, face_alone)` — weighted-max, bukan aditif
- Single cue yang sangat kuat tetap bisa trigger (tangan facepalm kuat tidak perlu ekspresi wajah)

### Threshold Aktif (thresholds.json)

| Label | Threshold |
|---|---|
| Boredom | 0.28 |
| Engagement | 0.24 |
| Confusion | 0.34 |
| Frustration | 0.26 |

Threshold ini dibandingkan dengan `avg_score` (rata-rata hybrid score semua frame) untuk menentukan prediksi video (0 atau 1).

> Untuk penjelasan lengkap setiap rumus dengan contoh angka nyata, lihat **[docs/COMPUTATION.md](docs/COMPUTATION.md)**.

---

## 6. Sistem Cache

Untuk mendukung Recalculate Batch tanpa re-run model, aplikasi menyimpan dua jenis cache per video:

### Raw Feature Cache (`raw_cache/`)

Disimpan saat **Proses Video Ini** atau **Batch Semua** pertama kali. Berisi output MediaPipe per frame:

```json
{
  "video_rel": "kelas1/siswa/video.mp4",
  "generated_at": "2025-01-01T12:00:00",
  "frames": [
    {
      "frame_idx": 0,
      "face_found": true,
      "yaw": 5.2,
      "pitch": -2.1,
      "iris_x": 0.12,
      "iris_y": -0.05,
      "blendshapes": { "browDownLeft": 0.12, "jawOpen": 0.08, "...": "..." },
      "hand_forehead": 0.0,
      "hand_chin": 0.0
    }
  ]
}
```

`hand_forehead` = proporsi titik tangan di zona dahi (y ∈ [-0.20, 0.25) dari crop).
`hand_chin` = proporsi titik tangan di zona pipi/dagu (y ∈ [0.25, 1.20] dari crop).

### SigLIP Score Cache (`siglip_cache/`)

Berisi skor SigLIP2 mentah (sebelum hybrid combine) per frame per label:

```json
{
  "video_rel": "kelas1/siswa/video.mp4",
  "frames": [
    { "frame_idx": 0, "siglip_scores": [0.31, 0.62, 0.28, 0.19] }
  ]
}
```

Nama file cache: path video dengan separator `/` diganti `__`, tanpa ekstensi.

---

## 7. Konfigurasi (.env)

```env
# Model SigLIP2 (dari HuggingFace)
SIGLIP_MODEL_ID=google/siglip2-base-patch16-224

# Folder output (relatif terhadap folder dataset yang dibuka)
OUTPUT_DIR=hasil_label7

# Padding crop wajah — 0.30 = ketat, 0.60 = longgar, 0.80 = sangat longgar
FACE_CROP_PADDING=0.60

# Auto-flag video jika jumlah frame dengan >1 wajah melebihi threshold ini
MULTI_FACE_FRAMES_THRESHOLD=4
```

> Bobot hybrid (siglip_w, land_w) dan semua parameter scoring diatur melalui **Rules Editor** di UI, bukan via `.env`. File `rules.json` di dalam folder output menyimpan konfigurasi tersebut.

---

## 8. Struktur Kode

```
Labeler-App-Siglip-2/
├── app.py                    # Entry point — GUI & event orchestration
├── ai_service.py             # Headless REST mode (opsional, via FastAPI)
├── main.py                   # Launcher alternatif
├── requirements.txt
├── .env / .env.example
│
├── core/
│   ├── siglip_model.py       # Singleton loader SigLIP2 (lazy load, GPU/CPU)
│   ├── inference.py          # Hybrid scoring: SigLIP × Landmark fusion
│   ├── landmark_analyzer.py  # MediaPipe: head pose, iris, blendshapes, tangan, viz
│   ├── face_detector.py      # BlazeFace crop wajah 512×512 + return bbox
│   ├── rules.py              # DEFAULT_RULES + load_rules / save_rules
│   ├── recalculate.py        # Recalculate batch dari raw_cache + siglip_cache
│   └── README_SIGLIP.md      # Dokumentasi teknis pipeline inferensi
│
├── ui/
│   ├── constants.py          # LABELS, LABEL_COLORS, DEFAULT_PROMPT_GROUPS
│   ├── left_panel.py         # Panel kiri: video player + galeri frame
│   ├── right_panel.py        # Panel kanan: AI score bars, prompt, threshold, split
│   └── rules_panel.py        # Toplevel: editor semua parameter rules
│
├── utils/
│   ├── io.py                 # Baca/tulis CSV/JSON annotations
│   └── video.py              # Ekstraksi frame, crop, landmark pipeline, simpan cache
│
└── docs/
    ├── RULES_PANEL.md        # Dokumentasi lengkap Rules Editor & semua parameter
    └── COMPUTATION.md        # Panduan teknis: dari angka MediaPipe ke prediksi akhir
```

---

## 9. File Output

Semua output tersimpan di `{folder_dataset}/{OUTPUT_DIR}/` (default: `hasil_label7/`):

| File / Folder | Format | Isi |
|---|---|---|
| `annotations_bener.csv` | CSV | Label per video (0 atau 1) per 4 label |
| `frame_annotations.json` | JSON | Label per frame per video (N frame × 4 label + rejected flag) |
| `batch_history.json` | JSON | Riwayat skor AI: avg_score, siglip_avg, landmark_avg, frame_scores, threshold |
| `{nama}.json` | JSON | Salinan batch history dengan nama kustom (jika "Buat batch baru" dicentang) |
| `flagged_videos.csv` | CSV | Daftar video yang di-flag/reject |
| `skipped_videos.json` | JSON | Daftar video yang di-skip |
| `thresholds.json` | JSON | Nilai threshold per label yang terakhir disimpan |
| `rules.json` | JSON | Parameter scoring landmark + hybrid weights (dari Rules Editor) |
| `raw_cache/{safe_name}.json` | JSON | Raw MediaPipe features per frame per video |
| `siglip_cache/{safe_name}.json` | JSON | SigLIP scores per frame per video |
| `cropped_faces/clean/` | JPG | Crop wajah bersih 512×512 untuk training |
| `cropped_faces/viz/` | JPG | Crop wajah dengan overlay landmark + emotion scores |
| `Label2d/{nama}/train.csv` | CSV | Split train (80%) berdasarkan UUID siswa |
| `Label2d/{nama}/val.csv` | CSV | Split val (10%) berdasarkan UUID siswa |
| `Label2d/{nama}/test.csv` | CSV | Split test (10%) berdasarkan UUID siswa |

---

## 10. FAQ & Troubleshooting

**Q: Kenapa AI skor Confusion selalu rendah?**
A: Confusion sekarang murni ekspresi wajah (tanpa tangan). Pastikan wajah menunjukkan alis berkerut, mangap tipis, atau lirik atas. Jika tangan terdeteksi dekat kepala, skor Confusion ditekan otomatis. Coba turunkan threshold Confusion di `thresholds.json` (dari 0.34 ke 0.30).

**Q: Kenapa ada tangan di frame tapi tidak terdeteksi Frustration?**
A: Dua kemungkinan: (1) tangan tidak cukup proporsional di crop — MediaPipe membutuhkan tangan cukup jelas terlihat di area crop wajah; (2) skor wajah rendah dan tangan score juga rendah — formula weighted-max masih membutuhkan minimal salah satu sinyal cukup kuat. Coba turunkan threshold Frustration (misal ke 0.22).

**Q: Kenapa orang yang jelas menatap ke depan masih dapat Boredom?**
A: Kemungkinan video belum di-Recalculate setelah perubahan rules. Klik **Recalculate Batch** di Rules Editor. Formula terbaru memiliki `expr_gate` yang mencegah blink/yawn trigger Boredom saat gaze lurus ke depan.

**Q: Kenapa gallery frame hanya highlight 1 frame sebagai Boredom padahal harusnya lebih?**
A: Highlight frame di gallery menggunakan `frame_preds` dari cache lama. Prediksi video final (`avg_score vs threshold`) bisa berbeda. Klik **Proses Video Ini** untuk refresh semua nilai dengan threshold saat ini.

**Q: Confusion dan Frustration terdeteksi bersamaan — apakah normal?**
A: Bisa terjadi jika siswa sangat bingung sampai stres (ada ekspresi tegang + tanda kebingungan). Namun sejak formula terbaru, kehadiran tangan dekat kepala menekan skor Confusion — jadi co-occurrence seharusnya berkurang. Jika masih sering, naikkan threshold Frustration.

**Q: Boredom dan Engagement muncul bersamaan — apakah salah?**
A: Tidak selalu salah. Valid jika siswa ngetik/membaca ke bawah sambil terlihat mengantuk (mata berat + lookDown tinggi). Tidak valid jika tidak ada konteks ngetik. Jika terlalu sering terjadi, naikkan threshold Boredom (misal dari 0.28 ke 0.32).

**Q: Kenapa viz thumbnail tidak berubah setelah Recalculate?**
A: Seharusnya berubah otomatis. Pastikan video sudah pernah diproses setidaknya sekali sebelumnya (agar cache ada). Jika cache belum ada, lakukan **Proses Video Ini** atau **Batch Semua** terlebih dahulu.

**Q: Setelah ganti rules dan simpan, apakah langsung berpengaruh?**
A: **Simpan** di Rules Editor menyimpan rules ke `rules.json`. Inferensi berikutnya akan pakai rules baru. Untuk video yang sudah pernah diproses, **wajib jalankan Recalculate Batch** — tanpa ini, label lama tidak terupdate dan dataset menjadi tidak konsisten.

**Q: Berapa lama proses Batch AI?**
A: ~3–8 detik per video dengan GPU, ~15–30 detik per video di CPU. Recalculate jauh lebih cepat (~0.1 detik per video) karena tidak re-run model besar.

**Q: Apakah bisa mengubah prompt SigLIP?**
A: Ya. Klik header label di panel kanan untuk expand, lalu edit prompt di textbox. Perubahan langsung berlaku untuk inferensi berikutnya. Untuk efek ke seluruh dataset, jalankan ulang Batch Semua (perlu re-run SigLIP karena cache SigLIP terikat ke prompt yang dipakai saat itu).

**Q: UUID idx di Split Label 2D itu apa?**
A: Indeks komponen path (0-based) yang mengidentifikasi siswa. Misal path `clips/batch-1/siswa_abc/sesi/video.mp4`: komponen ke-2 = `siswa_abc`. Set UUID idx = 2 untuk split per siswa. Default: 2.

**Q: Kapan harus Restart Batch vs Recalculate?**
A: **Recalculate** — gunakan setelah mengubah rules/threshold. Cepat, tidak re-run model, pakai cache yang ada. **Restart Batch** — gunakan jika ingin re-run model dari awal (misal setelah mengubah prompt SigLIP). Menghapus semua history dan cache. Jauh lebih lambat.
