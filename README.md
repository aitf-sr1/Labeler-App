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

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Kelopak mata berat / setengah tertutup | Mata tidak terbuka penuh, terlihat mengantuk |
| Menguap | Mulut terbuka lebar |
| Kepala noleh ke samping | Tidak memandang ke depan/kamera, kepala miring |
| Tatapan mata ke samping | Bola mata melihat ke kiri/kanan, bukan ke depan |
| Ekspresi datar/kosong | Tidak ada reaksi, otot wajah rileks total |
| Kepala menengadah | Kepala mendongak ke atas (sering disertai mata mengantuk) |

**Kapan beri label 0?**
- Wajah masih menghadap depan meskipun ekspresi biasa
- Mata terbuka normal meski tidak tersenyum
- Hanya sekilas noleh (bukan konsisten sepanjang frame)

**Catatan sistem:** Landmark menggunakan `gaze_dev` (deviasi angular 2D dari arah kamera) sebagai sinyal utama. Menguap, mata berat, dan kepala mendongak sebagai sinyal pendukung (maks 0.70 dari nilai penuh) agar siswa yang hanya menguap tapi masih menghadap depan tidak langsung mendapat skor 1.0.

---

### ENGAGEMENT (Keterlibatan) — Label 1

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Mata terbuka penuh menatap lurus | Memandang ke depan/layar/kamera secara langsung |
| Kepala tegak menghadap depan | Yaw kepala kecil, tidak menoleh |
| Respons wajah aktif | Mengangguk, alis sedikit terangkat, senyum tipis tanda tertarik |
| Ekspresi segar dan hadir | Terlihat sadar dan memperhatikan, bukan hanya duduk diam |

**Kapan beri label 0?**
- Kepala menoleh, terlihat melamun
- Mata melirik ke arah lain secara dominan
- Terlihat mengantuk atau tatapan kosong

**Catatan sistem:** Engagement dan Boredom menggunakan metrik `gaze_dev` yang sama — keduanya **inversely related**. Semakin tinggi deviasi gaze → Boredom naik, Engagement turun.

---

### CONFUSION (Kebingungan) — Label 2

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Alis berkerut/turun | Dahi mengerut, kedua alis turun ke tengah |
| Alis bagian dalam naik | Bagian tengah alis terangkat membentuk pola "∧" |
| Mata menyipit / melirik ke atas | Squinting saat berusaha memahami, atau bola mata melirik ke atas |
| Mulut sedikit terbuka | Mangap tipis (BUKAN menguap lebar) |
| Bibir mengerucut (pucker) | Bibir sedikit maju/mengerut saat berpikir keras |
| Kepala miring/mendongak sedikit | Kepala miring ke samping atau sedikit mendongak saat berpikir |
| Tangan menggaruk kepala | Tangan di atas garis mata (area dahi/kepala) |

**Kapan beri label 0?**
- Ekspresi netral tanpa alis berkerut
- Wajah terlihat datar meski sedang diam
- Hanya satu ciri ringan tanpa kombinasi

**PENTING — Perbedaan Confusion vs Frustration:**
Confusion = **berpikir keras** (masih sabar, ingin mengerti). Frustration = **stres/marah** (sudah menyerah, kesal). Jika alis berkerut tapi wajah masih "tenang", itu Confusion. Jika alis berkerut sambil mengernyit hidung atau bibir ditekan keras, itu Frustration.

**Catatan sistem:** `browInnerUp` (alis dalam naik) hanya aktif jika bersamaan dengan sinyal mata (lirik atas, lookUp) atau kepala (pitch). Ini mencegah false positive pada siswa yang bentuk alisnya secara natural melengkung ke atas.

---

### FRUSTRATION (Frustrasi) — Label 3

**Kapan beri label 1?**

| Tanda Visual | Keterangan |
|---|---|
| Kombinasi otot tegang | HARUS ada minimal 2 dari: alis turun keras + bibir ditekan + mata menyipit tajam + pipi menegang |
| Mengernyit hidung (nose sneer) | Hidung berkerut seperti ekspresi jijik/marah — sinyal terkuat |
| Mata tertutup rapat/dipaksa | Bukan mengantuk — menutup mata karena frustrasi (eye squint tajam) |
| Bibir ditekan keras (lip press) | Bibir merapatkan bersama secara kuat |
| Tangan menutupi wajah | Facepalm, tangan di pipi/hidung/mulut, atau menopang kepala |
| Ekspresi marah/stres ekstrem | Bukan hanya "berpikir", tapi terlihat kesal, mau menangis, atau sangat lelah mental |

**Kapan beri label 0?**
- Hanya satu otot yang sedikit tegang tanpa kombinasi
- Alis berkerut tapi tidak ada otot lain yang tegang → itu Confusion
- Tangan menyentuh wajah tapi bukan facepalm/menutupi

**Catatan sistem:** Frustration menggunakan **SUM logic** (bukan MAX) — satu sinyal saja tidak cukup. Kecuali `noseSneer` yang sangat jarang terjadi secara natural.

---

### Bolehkah Multi-Label?

**Ya.** Contoh kombinasi yang valid:
- `Confusion=1, Engagement=1` — siswa memperhatikan tapi tampak tidak mengerti
- `Boredom=1, Confusion=0` — siswa noleh, tidak bingung, hanya tidak tertarik
- `Confusion=1, Frustration=1` — siswa sangat bingung sampai terlihat stres
- **Tidak lazim:** `Boredom=1, Engagement=1` — kontradiktif, hindari kecuali frame benar-benar ambigu

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
2. Membuat folder output di `{folder_dataset}/{OUTPUT_DIR}/` (default: `hasil_label6/`)
3. Membuat sub-folder `raw_cache/` dan `siglip_cache/` untuk sistem cache
4. Memuat ulang semua data yang sudah tersimpan (annotations, batch history, rules)

### Memutar Video

Video diputar otomatis saat dimuat. Klik area video untuk pause/play. Di bawah video terdapat **6 thumbnail** yang mewakili 6 titik terdistribusi merata sepanjang video.

### Labeling Per-Frame (Manual)

1. Pilih tab label aktif di atas galeri (misal `Confusion`)
2. **Klik kiri pada thumbnail** → seek video ke posisi frame tersebut
3. **Klik kanan pada thumbnail** → toggle label aktif pada frame tersebut
   - Border berwarna = positif (label = 1)
   - Border abu-abu = negatif (label = 0)
4. **Double-klik pada thumbnail** → tandai frame sebagai "ditolak" (overlay merah)
   - Frame ditolak tidak dihitung dalam voting prediksi label video
5. Jika **≥ setengah frame valid** positif → label video = 1

### AI Inferensi — Proses Satu Video

Klik **Proses Video Ini** di panel kanan. Proses:
1. Ekstrak & crop wajah dari 6 frame video aktif
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
- Face mesh 3D tesselation (garis putih tipis)
- Skeleton tangan (hijau)
- Lingkaran iris per mata (cyan)
- Panah gaze per mata (kuning)
- Signal bars blendshape (sisi kanan)
- Emotion score bars (bawah)

Berguna untuk debugging mengapa AI memberi skor tertentu pada frame tertentu. Gambar viz diperbarui otomatis setiap kali rules diubah dan recalculate selesai.

### Rules Editor

Klik tombol **Rules** di topbar untuk membuka jendela Rules Editor (Toplevel terpisah). Jendela ini memungkinkan pengaturan ulang semua ~50 parameter scoring tanpa edit kode.

Lihat [docs/RULES_PANEL.md](docs/RULES_PANEL.md) untuk dokumentasi lengkap semua parameter.

**Cara pakai:**
- Setiap baris memiliki **slider** (drag kasar) dan **input field** (ketik angka pasti)
- Ketik angka di field → tekan **Enter** atau klik area lain → nilai di-clamp ke range valid dan slider ikut bergerak
- Klik **Reset Default** untuk mengembalikan semua parameter ke nilai awal
- Klik **Simpan** untuk menyimpan ke `rules.json` (berlaku untuk inferensi berikutnya)
- Klik **Recalculate Batch** untuk menghitung ulang seluruh batch dari cache dengan parameter baru

### Recalculate Batch

Tersedia di Rules Editor. Fitur ini menghitung ulang prediksi untuk **semua video yang sudah pernah diproses**, menggunakan:
- Raw feature cache (blendshapes, head pose, iris) yang tersimpan di `raw_cache/`
- SigLIP score cache yang tersimpan di `siglip_cache/`
- Parameter rules dan threshold terkini

**Catatan:** Hanya video yang memiliki kedua file cache yang bisa di-recalculate. Video yang di-skip (belum pernah diproses) tetap di-skip.

Setelah recalculate:
- `batch_history.json` diperbarui
- `frame_annotations.json` diperbarui
- Viz thumbnail video aktif di-regenerate otomatis dengan skor baru
- AI score bars di panel kanan diperbarui

### Split Label 2D

Klik **Split Label 2D** di panel kanan untuk membagi dataset menjadi train/val/test berdasarkan identitas siswa (UUID dari path).

- Splitting dilakukan per **siswa** (bukan per video) sehingga tidak ada kebocoran data antar split
- Rasio split: **80% train / 10% val / 10% test**
- Output: 3 file CSV (`train.csv`, `val.csv`, `test.csv`) di folder `Label2d/{nama_batch}/`
- **UUID idx** — indeks komponen path yang menjadi identifier siswa (default: 2, yaitu komponen ke-3 dari path video). Sesuaikan dengan struktur folder dataset.

Contoh: jika path video adalah `dataset/kelas1/siswa_abc/sesi1/video.mp4` dan UUID ada di komponen ke-3 (`siswa_abc`), set UUID idx = 3.

### Reset & Flag

- **Reset Label Video Ini** (merah) — hapus semua frame annotations dan riwayat AI untuk video aktif, reset ke nol. Tanpa konfirmasi.
- **Flag/Reject** di topbar — tandai video sebagai tidak layak dilabeli (kualitas buruk, wajah tidak jelas). Video terflag tidak dimasukkan ke dataset training dan dicatat di `flagged_videos.csv`.

### Keyboard Shortcut

| Tombol | Fungsi |
|---|---|
| `←` / `→` | Video sebelumnya / berikutnya |
| `Space` | Pause / play video |
| `S` | Simpan dan lanjut ke video berikutnya |

---

## 5. Cara Kerja Sistem Scoring (Ringkasan)

Sistem menggunakan **Hybrid Scoring**: menggabungkan dua sumber sinyal per frame, lalu voting mayoritas dari 6 frame untuk keputusan akhir.

```
                    ┌─ SigLIP2 (visual)  ─────────────────────────────┐
Video → 6 frame →  │  logit → sigmoid(logit + bias) → siglip_score    │
                    │                                                   ├─► hybrid_score = α·siglip + β·land
                    └─ MediaPipe (geometri) ──────────────────────────┘
                       yaw, pitch, iris, blendshapes → landmark_score

hybrid_score per frame → threshold → frame_pred (0/1)
vote dari 6 frame → prediction video (0/1)
```

**Empat label yang dihitung:**
- **Boredom** — dari `gaze_dev` (deviasi pandangan dari kamera) + sinyal menguap/mata ngantuk
- **Engagement** — inverse dari gaze_dev; gate turun ke nol saat kepala menoleh jauh
- **Confusion** — dari alis berkerut, lirik atas, mangap sedikit, bibir mengerucut; butuh kombinasi sinyal
- **Frustration** — SUM logic: noseSneer + kombinasi otot tegang + gestur tangan facepalm

Semua parameter (threshold sinyal, bobot blend, bobot α/β) dapat diubah lewat **Rules Editor** tanpa edit kode.

> Untuk penjelasan lengkap setiap rumus dengan contoh angka nyata step-by-step, lihat **[docs/COMPUTATION.md](docs/COMPUTATION.md)**.
>
> Untuk cara mengatur parameter agar skor label membaik, lihat **[docs/RULES_PANEL.md](docs/RULES_PANEL.md)**.
>
> Untuk arsitektur pipeline dan sistem cache, lihat **[core/README_SIGLIP.md](core/README_SIGLIP.md)**.

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
      "iris_img_x": 0.01,
      "iris_img_y": 0.02,
      "blendshapes": { "browDownLeft": 0.12, "jawOpen": 0.08, ... },
      "hand_forehead": 0.0,
      "hand_chin": 0.0
    },
    ...
  ]
}
```

### SigLIP Score Cache (`siglip_cache/`)

Berisi skor SigLIP2 mentah (sebelum hybrid combine) per frame per label:

```json
{
  "video_rel": "kelas1/siswa/video.mp4",
  "generated_at": "2025-01-01T12:00:00",
  "frames": [
    { "frame_idx": 0, "siglip_scores": [0.31, 0.62, 0.28, 0.19] },
    ...
  ]
}
```

Nama file cache: `rel_path` dengan separator `/` dan `\` diganti `__`, tanpa ekstensi.
Contoh: `kelas1__siswa_abc__sesi1__video.json`

---

## 7. Konfigurasi (.env)

```env
# Model SigLIP2 (dari HuggingFace)
SIGLIP_MODEL_ID=google/siglip2-base-patch16-224

# Folder output (relatif terhadap folder dataset yang dibuka)
OUTPUT_DIR=hasil_label6

# Padding crop wajah — 0.30 = ketat, 0.60 = longgar, 0.80 = sangat longgar
FACE_CROP_PADDING=0.60

# Auto-flag video jika jumlah frame dengan >1 wajah melebihi threshold ini
# Set ke angka besar (misal 16) untuk menonaktifkan auto-flag
MULTI_FACE_FRAMES_THRESHOLD=4
```

> **Catatan:** Bobot hybrid (siglip_w, land_w) dan parameter scoring kini diatur melalui **Rules Editor** di UI, bukan via `.env`. File `rules.json` di dalam folder output menyimpan konfigurasi tersebut.

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
│   ├── left_panel.py         # Panel kiri: video player + galeri 6 frame
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

Semua output tersimpan di `{folder_dataset}/{OUTPUT_DIR}/` (default: `hasil_label6/`):

| File / Folder | Format | Isi |
|---|---|---|
| `annotations_bener.csv` | CSV | Label per video (0 atau 1) per 4 label |
| `frame_annotations.json` | JSON | Label per frame per video (6 frame × 4 label + rejected flag) |
| `batch_history.json` | JSON | Riwayat skor AI: avg_score, siglip_avg, landmark_avg, frame_scores, threshold |
| `{nama}.json` | JSON | Salinan batch history dengan nama kustom (jika "Buat batch baru" dicentang) |
| `flagged_videos.csv` | CSV | Daftar video yang di-flag/reject |
| `skipped_videos.json` | JSON | Daftar video yang di-skip |
| `thresholds.json` | JSON | Nilai threshold slider per label yang terakhir disimpan |
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
A: Coba turunkan threshold slider Confusion (misal ke 0.35–0.40). Bisa juga buka Rules Editor dan turunkan `brow_dn_th` atau `brow_in_th` untuk Confusion.

**Q: Kenapa viz thumbnail tidak berubah setelah Recalculate?**
A: Seharusnya berubah otomatis. Pastikan video sudah diproses setidaknya sekali sebelumnya (agar cache ada). Jika cache belum ada, lakukan "Proses Video Ini" atau "Batch Semua" terlebih dahulu.

**Q: Setelah ganti rules dan simpan, apakah langsung berpengaruh?**
A: Klik **Simpan** di Rules Editor menyimpan rules ke `rules.json`. Inferensi berikutnya (Proses Video Ini / Batch Semua) akan pakai rules baru. Untuk video yang sudah pernah diproses, gunakan **Recalculate Batch** agar tidak perlu re-run model.

**Q: Confusion dan Frustration terdeteksi bersamaan — apakah normal?**
A: Ya, bisa terjadi. Siswa yang sangat bingung sampai stres bisa memiliki kedua emosi aktif. Jika sering bersamaan padahal wajah tidak menunjukkan tanda stres, naikkan threshold Frustration.

**Q: Kenapa video di-flag otomatis?**
A: Tiga kondisi auto-flag:
1. Wajah tidak terdeteksi di sebagian besar frame
2. Video terlalu pendek (< 6 frame yang bisa diekstrak)
3. Jumlah frame dengan > 1 wajah melebihi `MULTI_FACE_FRAMES_THRESHOLD` di `.env`

**Q: Berapa lama proses Batch AI?**
A: ~3–8 detik per video dengan GPU, ~15–30 detik per video di CPU. Recalculate jauh lebih cepat (~0.1 detik per video) karena tidak re-run model.

**Q: Apakah bisa mengubah prompt SigLIP?**
A: Ya. Klik header label di panel kanan untuk expand, lalu edit prompt di textbox. Perubahan langsung berlaku untuk inferensi berikutnya.

**Q: UUID idx di Split Label 2D itu apa?**
A: Indeks komponen path (0-based) yang mengidentifikasi siswa. Misal path `dataset/kelas/siswa_abc/video.mp4`: komponen ke-0 = `dataset`, ke-1 = `kelas`, ke-2 = `siswa_abc`. Set UUID idx = 2 untuk split per siswa. Default: 2.

**Q: Apa itu `batch_history.json` dan kapan harus di-reset?**
A: Menyimpan skor AI dari setiap video yang sudah diproses. Video yang sudah ada di history di-skip saat Batch Semua. Klik **Restart Batch** untuk menghapus history dan memproses ulang semua video dari awal.
