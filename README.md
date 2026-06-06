# Labeler App — Pelabelan Emosi Pembelajaran

Aplikasi desktop untuk melabeli emosi siswa (Boredom, Engagement, Confusion, Frustration) pada video secara manual maupun semi-otomatis.

Disebut **"Labeler App"** (bukan lagi "Labeler SigLIP") karena tidak mengandalkan satu model saja — keputusan label menggabungkan **banyak unsur sinyal**:

| Unsur | Peran |
|---|---|
| **SigLIP2** (Google VLM) | Membaca ekspresi wajah halus dari crop wajah 224×224 (visual) |
| **MediaPipe FaceLandmarker** | Geometri wajah 3D: arah kepala (yaw/pitch/roll), iris/gaze, 52 blendshape |
| **MediaPipe HandLandmarker** | Deteksi tangan di wajah (hand-over-face) → cue Confusion/Frustration |
| **MediaPipe FaceLandmarker** | Satu-satunya sumber AU: blendshape→AU dengan normalisasi baseline-relative (stretch agresif AU4, per-person calibration) |
| **Kalibrasi per-orang** | Frame netral per siswa (Bosch 2023) sebagai baseline AU |

Keempat-empatnya digabung lewat **Hybrid Scoring** (lihat [§5](#5-cara-kerja-sistem-scoring-ringkasan)). Setiap mekanisme punya dasar paper — lihat [docs/ACADEMIC_BASIS.md](docs/ACADEMIC_BASIS.md).

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
| [docs/ACADEMIC_BASIS.md](docs/ACADEMIC_BASIS.md) | **Dasar akademis** tiap mekanisme: verbatim quote dari paper + penjelasan (FACS/AU, gaze, tangan, kalibrasi per-orang, co-occurrence). |
| [docs/ALUR_METODE.md](docs/ALUR_METODE.md) | Alur metode end-to-end: sinyal apa → emosi apa → paper mana, beserta bobotnya. |
| [docs/COMPUTATION.md](docs/COMPUTATION.md) | Panduan teknis lengkap: dari angka mentah MediaPipe → rumus per-emosi → SigLIP → hybrid → prediksi. Dilengkapi contoh angka nyata dan referensi akademis. |
| [docs/PANDUAN_ANOTASI_MANUAL.md](docs/PANDUAN_ANOTASI_MANUAL.md) | Panduan anotator manusia: ciri tiap emosi, tabel gaze, tabel suppress/co-occurrence, cue tangan. |
| [docs/DESIGN_RATIONALE.md](docs/DESIGN_RATIONALE.md) | Alasan desain tiap bobot (kenapa angka segini), kode vs. paper. |
| [docs/RULES_PANEL.md](docs/RULES_PANEL.md) | Cara menggunakan Rules Editor: semua ~50 parameter dijelaskan, panduan "kalau label X bermasalah ubah ini", strategi tuning. |
| [core/README_SIGLIP.md](core/README_SIGLIP.md) | Arsitektur pipeline inferensi: SigLIP2, MediaPipe-only AU pipeline, kalibrasi per-orang, sistem cache, recalculate, viz. |

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

### Mode Label Semi Manual

Aktifkan switch **Label Semi Manual** di panel kanan untuk mengaktifkan mode edit manual. Tanpa mode ini, hasil AI tidak bisa diubah — semua aksi edit (klik kanan, double-klik, toggle Flag/Reject) di-block.

Panel statistik menampilkan dua kolom terpisah:
- **Hasil AI** — dihitung dari `frame_annotations.json` (tidak terpengaruh perubahan manual)
- **Hasil Manual** — dihitung dari `manual_labels.json` (berisi override manual)

### Labeling Per-Frame (Manual)

> Memerlukan mode **Label Semi Manual** aktif.

1. Pilih tab label aktif di atas galeri (misal `Confusion`)
2. **Klik kiri pada thumbnail** → seek video ke posisi frame tersebut
3. **Klik kanan pada thumbnail** → toggle label aktif pada frame tersebut
   - Border berwarna = positif (label = 1)
   - Border abu-abu = negatif (label = 0)
4. **Double-klik pada thumbnail** → tandai frame sebagai "ditolak" (overlay merah)
   - Frame ditolak tidak dihitung dalam prediksi label video
   - Status ditolak disimpan di `manual_labels.json` — tidak mengubah hasil AI
5. Prediksi akhir video berdasarkan **vote mayoritas frame** dengan threshold per-frame

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

Klik tab **Rules** di atas galeri frame untuk membuka Rules Editor — editor mengambil alih area galeri secara penuh. Klik **Kembali** (atau **Simpan**) untuk kembali ke galeri.

**Cara pakai:**
- Setiap baris memiliki **slider** (drag kasar) dan **input field** (ketik angka pasti)
- Ketik angka di field → tekan **Enter** atau klik area lain → nilai di-clamp ke range valid
- Klik **Reset Default** untuk mengembalikan semua parameter ke nilai awal
- Klik **Simpan** untuk menyimpan ke `rules.json` tanpa recalculate
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

Sistem menggunakan **Hybrid Scoring**: menggabungkan dua jalur sinyal per frame, lalu rata-rata dari seluruh frame untuk keputusan akhir.

```
                    ┌─ SigLIP2 (visual) ──────────────────────────────┐
                    │  logit → sigmoid(logit + bias) → siglip_score    │
Video → N frame →  │                                                   ├─► hybrid = (siglip_w×siglip + land_w×land) / (siglip_w + land_w)
                    │  LANDMARK_score = gabungan dari:                 │
                    │   • MediaPipe Face: yaw/pitch/roll, iris/gaze    │
                    │   • MediaPipe Hand: tangan di wajah (HoF)        │
                    │   • Action Unit FACS: py-feat (utama) /          │
                    └─    blendshape→AU (fallback), baseline-normalized┘

avg(hybrid_score semua frame) → vs threshold → prediction video (0/1)
```

> **Penting:** `landmark_score` **bukan cuma geometri MediaPipe** — ia membungkus **Action Unit FACS** (py-feat/blendshape) + **sinyal tangan**. Inilah alasan app ini bukan sekadar "SigLIP labeler".

### Action Unit (AU FACS) — jantung deteksi Confusion & Frustration

Confusion/Frustration tidak bisa dibaca dari arah kepala saja; keduanya butuh **otot wajah halus**. Sistem memetakannya ke **Action Unit FACS** (Craig et al. 2008):

| AU | Otot | Dipakai untuk |
|---|---|---|
| AU1 + AU2 | inner+outer brow raise | **Frustration** (Craig 2008: co-occurrence 100%) — `brow_raise_direct_w=0.65` |
| AU4 + AU14 | brow lowerer + dimpler | **Frustration PRIMER** (Grafsgaard 2013) — `face_weight=0.60` |
| AU4 + AU7 | brow lowerer + lid tightener | **Confusion** (Craig 2008: 95%/78%) — `au7_alone_w=0.78`, `au4_au7_co_w=0.50` |
| AU25 + AU26 | lips part + jaw drop | **Confusion** (Namba 2024 "thinking face") — `mouth_open_conf_w=0.25` |
| AU12 | lip corner (smile) | **gate** Confusion (lintas-emosi, bukan sinyal positif) |
| AU43 | eye closure | **Boredom** — `blink_direct_w=0.45` |

**Dua sumber AU (dengan baseline-normalization):**
1. **py-feat** — detektor yang dilatih langsung pada FACS (sumber **utama**, jalan di GPU). Berada di venv terpisah `.venv-pyfeat`, dipanggil lewat subprocess.
2. **Blendshape MediaPipe → AU** — fallback bila py-feat tidak tersedia.

AU dinormalisasi terhadap baseline: `intensity = clamp((raw − neutral) / (active − neutral), 0, 1)`. Bila siswa punya **frame netral** yang ditandai (kalibrasi per-orang, Bosch 2023), `neutral` diambil dari frame itu; jika tidak, dipakai baseline populasi.

### Bobot Hybrid per Label (rules.json → `hybrid.siglip_w` / `hybrid.land_w`)

| Label | siglip_w | land_w | Alasan |
|---|---|---|---|
| Boredom | 0.25 | 0.75 | gaze (geometrik) + AU43 dominan; SigLIP kecil (tired/vacant look) |
| Engagement | 0.50 | 0.50 | **HOLISTIK** — Whitehill 2014: tak ada AU tunggal, "static pixels" → SigLIP tertinggi |
| Confusion | 0.35 | 0.65 | AU4+AU7 (py-feat) primer + SigLIP jaring pengaman (oklusi tangan) |
| Frustration | 0.30 | 0.70 | AU1+2/AU4/AU14 (py-feat) + tangan primer + SigLIP gestalt stres |

> **Prinsip:** SigLIP **tertinggi di Engagement** (satu-satunya emosi holistik tanpa AU dominan — Whitehill 2014). Tiga emosi lain punya **AU diskrit** (Craig 2008) → bertumpu **py-feat (AU FACS) + MediaPipe**, tapi tetap diberi **SigLIP** sebagai *cross-check independen* (berguna saat py-feat gagal: oklusi tangan/wajah miring). SigLIP = expression reader tervalidasi (Zhai 2023); D'Mello 2009 memakai *"facial features"* untuk keempat emosi — jadi bobot SigLIP di sini **ada dasarnya**, dan nilainya = kalibrasi empiris (seperti threshold).

`hybrid = (siglip_w×siglip + land_w×land) / (siglip_w + land_w)` — rata-rata berbobot (ternormalisasi), bukan jumlah mentah.

### Formula Landmark per Emosi

**Boredom:**
- Signal utama: `gaze_dev` (deviasi angular 2D kepala+iris dari kamera) — noleh ke atas/samping = bosan.
- Signal pendukung: mata berat (AU43/eyeBlink), menguap (jawOpen).
- **expr_gate:** signal pendukung hanya aktif jika `gaze_dev` cukup besar. Gaze **ke bawah** (lihat keyboard/menulis) TIDAK dihitung bosan (Sümer 2021: head-down = taking notes).

**Engagement:**
- Inversely proportional dengan gaze_dev — semakin tinggi deviasi gaze, semakin rendah Engagement.
- Gaze ke bawah masih boleh engaged (lihat keyboard). Dikurangi jika ada droopy eyes parah.

**Confusion:**
- **AU brow + lid:** AU4 (brow lowerer) dan **AU7 standalone** (`au7_alone_w=0.78`) + co-occurrence AU4+AU7 (`au4_au7_co_w=0.50`).
- **Mulut "thinking face":** AU25+AU26 (`mouth_open_conf_w=0.25`, Namba 2024).
- **Tangan (HoF):** `max(hand_one,hand_two) × hand_conf_w=0.40` **menambah** Confusion (Behera 2020: HoF↑ saat difficulty↑). *Catatan: ini menggantikan "hand suppression" desain lama — tangan kini cue positif, bukan penekan.*
- **AU12 smile gate:** senyum lebar meredam Confusion sebagian (floor 0.30, tidak men-nol-kan — "questioning smile" boleh co-occur).
- Boleh **co-occur dengan Engagement** (D'Mello: productive confusion) — tidak ada suppress Conf↔Eng.

**Frustration:**
- **AU1+AU2** (inner+outer brow raise) sebagai sinyal langsung — `brow_raise_direct_w=0.65` (Craig 2008: 100%).
- **AU4+AU14** (brow lowerer + dimpler) sebagai **PRIMER** — `face_weight=0.60` (Grafsgaard 2013; dinaikkan dari sekunder).
- **2 tangan** di wajah — `hand_two × hand_frus_w=0.30` (Grafsgaard 2013b: self-efficacy rendah). Cue LEMAH.
- Formula: weighted-max antar sinyal — single cue kuat tetap bisa trigger.

### Threshold (thresholds.json)

Threshold dibandingkan dengan `avg_score` (rata-rata hybrid semua frame) untuk menentukan prediksi video (0/1). **Default 0.50 per label**, dapat diatur per-label lewat slider **Threshold** di panel kanan dan disimpan ke `thresholds.json`.

| Label | Default | Catatan kalibrasi |
|---|---|---|
| Boredom | 0.50 | turunkan bila boredom under-detected |
| Engagement | 0.50 | — |
| Confusion | 0.50 | dirancang agar cue tangan saja (land 0.34) belum memicu di sekitar ~0.35 |
| Frustration | 0.50 | dirancang agar 2-tangan saja (land 0.255) belum memicu di sekitar ~0.45 |

> Untuk penjelasan lengkap setiap rumus dengan contoh angka nyata, lihat **[docs/COMPUTATION.md](docs/COMPUTATION.md)**, dan dasar paper tiap mekanisme di **[docs/ACADEMIC_BASIS.md](docs/ACADEMIC_BASIS.md)**.

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
      "hand_one": 0.0,
      "hand_two": 0.0
    }
  ]
}
```

`hand_one` = 1.0 jika terdeteksi tepat 1 tangan di area wajah (count-based). Cue LEMAH Confusion via `max(hand_one,hand_two)*hand_conf_w=0.40` — Behera 2020: HoF ↑ saat difficulty ↑ + Mahmoud 2011: "unsure".
`hand_two` = 1.0 jika terdeteksi ≥2 tangan di area wajah. Cue LEMAH Confusion (HoF) + cue LEMAH Frustration (`hand_frus_w=0.30`) — Grafsgaard 2013b: hand-to-face ↔ self-efficacy rendah.

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
│   ├── landmark_analyzer.py  # MediaPipe: head pose, iris, blendshape, tangan, scoring, viz
│   ├── action_units.py       # Blendshape/py-feat → AU FACS, baseline-normalized
│   ├── pyfeat_worker.py      # Worker py-feat (subprocess di .venv-pyfeat, GPU) — detektor AU utama
│   ├── pyfeat_client.py      # Client cross-venv ke pyfeat_worker (stdin/stdout JSON)
│   ├── face_detector.py      # BlazeFace crop wajah + return bbox
│   ├── rules.py              # DEFAULT_RULES + load_rules / save_rules
│   ├── recalculate.py        # Recalculate batch dari raw_cache + siglip_cache
│   └── README_SIGLIP.md      # Dokumentasi teknis pipeline inferensi
│
├── ui/
│   ├── constants.py          # LABELS, LABEL_COLORS, DEFAULT_PROMPT_GROUPS
│   ├── left_panel.py         # Panel kiri: video player, galeri frame (+ marker frame netral)
│   ├── right_panel.py        # Panel kanan: statistik, AI bars, prompt, threshold, kartu kalibrasi
│   └── rules_panel.py        # RulesContent (embedded di gallery) + RulesPanel (compat wrapper)
│
├── utils/
│   ├── io.py                 # Baca/tulis CSV/JSON annotations + thresholds
│   ├── person_neutral.py     # Kalibrasi per-orang: simpan/baca frame netral (person_neutrals.json)
│   └── video.py              # Ekstraksi frame, crop, landmark pipeline, simpan cache
│
└── docs/
    ├── ACADEMIC_BASIS.md     # Dasar akademis + verbatim quote tiap mekanisme
    ├── ALUR_METODE.md        # Alur sinyal → emosi → paper
    ├── COMPUTATION.md        # Panduan teknis: dari angka MediaPipe ke prediksi akhir
    ├── DESIGN_RATIONALE.md   # Alasan tiap bobot (kode vs paper)
    ├── PANDUAN_ANOTASI_MANUAL.md  # Panduan anotator manusia
    └── RULES_PANEL.md        # Dokumentasi lengkap Rules Editor & semua parameter

(py-feat berjalan di venv terpisah `.venv-pyfeat` — torch+CUDA, di luar pohon ini)
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
| `manual_labels.json` | JSON | Override label manual per frame (mode Label Semi Manual) — terpisah dari hasil AI |
| `flagged_videos.csv` | CSV | Daftar video yang di-flag/reject |
| `skipped_videos.json` | JSON | Daftar video yang di-skip |
| `thresholds.json` | JSON | Nilai threshold per label yang terakhir disimpan |
| `rules.json` | JSON | Parameter scoring landmark + hybrid weights (dari Rules Editor) |
| `person_neutrals.json` | JSON | Kalibrasi per-orang: AU baseline frame netral per siswa (`{uuid: {AU..., _video, _frame}}`) — disimpan di folder dataset |
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
