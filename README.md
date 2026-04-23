# Labeler Emosi SigLIP2

Aplikasi desktop untuk melabeli emosi pada video secara manual maupun semi-otomatis menggunakan model SigLIP2 dari Google. Dibangun untuk kebutuhan anotasi emosi siswa.

## Daftar Isi

1. [Deskripsi](#deskripsi)
2. [Prasyarat](#prasyarat)
3. [Instalasi](#instalasi)
4. [Menjalankan Aplikasi](#menjalankan-aplikasi)
5. [Panduan Penggunaan](#panduan-penggunaan)
6. [Alur Kerja Kode](#alur-kerja-kode)
7. [File Output](#file-output)
8. [Struktur Kode](#struktur-kode)
9. [FAQ](#faq)

## Deskripsi

Aplikasi ini melabeli video dengan 4 kelas emosi. Setiap label bernilai `0` (tidak ada) atau `1` (ada).

| Label | Arti |
|---|---|
| Boredom | Siswa terlihat bosan |
| Engagement | Siswa terlihat fokus / terlibat |
| Confusion | Siswa terlihat bingung |
| Frustration | Siswa terlihat frustrasi |

Ada tiga mode pelabelan:

- **Manual** — nilai tiap label ditentukan secara manual oleh labeler
- **Semi-otomatis** — AI memberikan saran awal, labeler melakukan koreksi
- **Batch AI** — AI memproses semua video secara berurutan di background thread

## Prasyarat

- Python 3.9 atau lebih baru
- pip
- GPU NVIDIA opsional — inferensi juga bisa berjalan di CPU

## Instalasi

```bash
pip install -r requirements.txt
```

Download model SigLIP2 pertama kali membutuhkan sekitar 400 MB. Pastikan koneksi internet tersedia saat pertama kali menjalankan aplikasi.

## Menjalankan Aplikasi

```bash
python app.py
```

## Panduan Penggunaan

### Membuka Dataset

Klik tombol **Buka Folder** di pojok kiri atas, lalu pilih folder root dataset yang berisi file `.mp4`. Aplikasi akan mencari semua file video secara rekursif, membuat folder `hasil_label6/` sebagai direktori output, dan memuat ulang data yang sudah pernah disimpan secara otomatis.

### Memutar Video

Video diputar otomatis saat pertama dimuat. Klik area video untuk pause atau play. Slider di bawah video digunakan untuk berpindah posisi. Di bawah slider terdapat 16 thumbnail yang mewakili titik-titik distribusi merata sepanjang video.

### Labeling Manual

Panel kanan berisi tombol `0` dan `1` untuk setiap label emosi. Untuk labeling per-frame:

1. Pilih tab label aktif (misalnya `Engagement`) di atas galeri frame
2. Klik kanan pada frame untuk toggle — border berwarna berarti positif, abu berarti negatif
3. Jika 8 atau lebih dari 16 frame ditandai positif, nilai label video otomatis berubah menjadi `1`

### Menggunakan AI

Panel kanan bawah berisi kontrol inferensi SigLIP.

- **Proses Video Ini** — inferensi dijalankan pada video yang sedang ditampilkan
- **Batch Semua** — inferensi dijalankan pada seluruh dataset secara berurutan, dapat dihentikan sewaktu-waktu
- Bar di samping tiap label menunjukkan confidence score dari model

Video yang sudah pernah diproses akan di-skip secara otomatis berdasarkan `batch_history.json`. Prompt dan threshold dapat disesuaikan langsung di panel.

### Navigasi Antar Video

| Aksi | Keterangan |
|---|---|
| Save & Next atau panah kanan | Simpan label saat ini dan lanjut ke video berikutnya |
| Prev atau panah kiri | Kembali ke video sebelumnya |
| Skip | Lewati video tanpa menyimpan label |
| Kolom "Loncat ke" + Go | Lompat ke nomor video tertentu |
| Spasi | Play / Pause |

### Flag dan Reject Video

Aktifkan toggle **Flag / Reject** di bar bawah untuk menandai video sebagai bermasalah (blur, salah clip, tidak bisa dinilai), lalu klik Save & Next. Video tersebut dihapus dari `annotations_bener.csv` dan dicatat di `flagged_videos.csv`. Saat kembali ke video yang sudah di-flag, toggle aktif kembali secara otomatis.

## Alur Kerja Kode

### 1. Startup

```
python app.py
    -> VideoLabelerApp.__init__()
        -> inisialisasi variabel state (video_files, annotations_data, dll)
        -> _build_ui() -> topbar, LeftPanel, RightPanel, bottombar
        -> bind keyboard shortcut (spasi, panah kiri/kanan)
```

### 2. Membuka Folder

```
Klik "Buka Folder"
    -> open_folder()
        -> filedialog memilih folder
        -> tentukan path semua file output di hasil_label6/
        -> glob mencari semua .mp4 secara rekursif
        -> _load_data()
            -> load_annotations()       -> baca annotations_bener.csv
            -> load_flagged()           -> baca flagged_videos.csv
            -> load_frame_annotations() -> baca frame_annotations.json
            -> load_batch_history()     -> baca batch_history.json
            -> load_skipped()           -> baca skipped_videos.json
        -> load_video(index=0)
```

### 3. Memuat Video

```
load_video(index)
    -> buka file .mp4 dengan cv2.VideoCapture
    -> baca total_frames dan fps
    -> restore label dari annotations_data jika sudah pernah dilabeli
    -> restore status flag dari flagged_data
    -> refresh_frame_gallery()
        -> prepare_cropped_frames()
            -> cek cache di cropped_faces/
            -> jika belum ada: extract_16_frames() -> crop_face() per frame -> simpan .jpg
            -> return 16 PIL.Image
        -> render 16 thumbnail di LeftPanel
        -> update vote bar dan AI score bar di RightPanel
    -> toggle_play() -> mulai update_frame() loop
```

### 4. Loop Playback

```
update_frame() dipanggil setiap 15ms via root.after()
    -> hitung target frame dari waktu yang sudah berlalu (time-based)
    -> cap.read() frame tersebut
    -> show_video_frame() -> tampilkan ke canvas video
    -> root.after(15, update_frame) -> jadwalkan ulang
```

### 5. Menyimpan Label

```
Klik "Save & Next" atau tekan panah kanan
    -> save_and_next()
        -> save_current_state()
            -> baca nilai label_vars (Boredom, Engagement, Confusion, Frustration)
            -> jika flag aktif:
                -> tambah ke flagged_data, hapus dari annotations_data
            -> jika flag tidak aktif:
                -> simpan [b, e, c, f] ke annotations_data
            -> save_annotations()       -> tulis ulang annotations_bener.csv
            -> save_flagged()           -> tulis ulang flagged_videos.csv
            -> save_frame_annotations() -> tulis ulang frame_annotations.json
        -> current_index += 1
        -> load_video(index baru)
```

### 6. Inferensi AI (Satu Video)

```
Klik "Proses Video Ini"
    -> _proses_satu()
        -> ambil prompts dan thresholds dari RightPanel
        -> jalankan worker() di background thread
            -> prepare_cropped_frames() -> 16 PIL.Image
            -> run_siglip_on_frames()
                -> gabung semua prompt jadi satu batch teks
                -> model menghasilkan logits [16 frames x N teks]
                -> normalisasi min-max per frame
                -> hitung avg_score dan voting per label
                -> return prediksi per label
            -> _apply_siglip_result() -> tulis ke frame_annotations + batch_history
            -> update_ui() via root.after() -> update tombol label + score bar
```

### 7. Inferensi AI (Batch Semua Video)

```
Klik "Batch Semua"
    -> _toggle_batch() -> _proses_semua()
        -> jalankan worker() di background thread
            -> untuk setiap video:
                -> jika sudah ada di batch_history -> skip
                -> prepare_cropped_frames()
                -> run_siglip_on_frames()
                -> _apply_siglip_result()
                -> save_annotations() dan save_frame_annotations() per video
                -> update status di UI via root.after()
            -> selesai: kembalikan tombol ke state normal
```

## File Output

Semua hasil disimpan di subfolder `hasil_label6/` di dalam folder dataset yang dibuka.

```
folder_dataset/
└── hasil_label6/
    ├── annotations_bener.csv    — Label utama tiap video
    ├── flagged_videos.csv       — Daftar video yang di-flag
    ├── frame_annotations.json  — Label per-frame (16 frame per video)
    ├── batch_history.json       — Riwayat hasil inferensi AI
    ├── skipped_videos.json      — Daftar video yang di-skip
    └── cropped_faces/           — Cache crop wajah
```

Format `annotations_bener.csv`:

```
UUID, Video_Asli, Clip_Name, File_Path, Boredom, Engagement, Confusion, Frustration
```

Format `flagged_videos.csv`:

```
UUID, Video_Asli, Clip_Name, File_Path
```

## Struktur Kode

```
siglip_microservice/
├── app.py                 — Entry point utama
├── main.py                — Entry point FastAPI
├── ai_service.py          — Fungsi inferensi untuk REST API
├── requirements.txt
├── ui/
│   ├── constants.py       — Label, warna, dan prompt default
│   ├── left_panel.py      — Video player dan galeri frame
│   └── right_panel.py     — Kontrol label, AI, dan prompt editor
├── core/
│   ├── siglip_model.py    — Singleton loader model SigLIP2
│   ├── face_detector.py   — Deteksi dan crop wajah (MediaPipe)
│   └── inference.py       — Inferensi zero-shot
└── utils/
    ├── io.py              — Baca/tulis CSV dan JSON
    └── video.py           — Ekstraksi frame dari video
```

Detail teknis model SigLIP2 dan pipeline inferensi tersedia di [README_SIGLIP.md](core/README_SIGLIP.md).

## FAQ

**Apakah progress hilang kalau aplikasi ditutup?**

Tidak. Data disimpan ke disk setiap kali Save & Next ditekan.

**Kenapa tidak ada file `flagged_videos.csv`?**

Di versi sebelumnya flag hanya tersimpan di memori dan hilang saat aplikasi ditutup. Bug ini sudah diperbaiki — flag kini langsung ditulis ke disk.

**Kenapa inferensi AI lambat?**

Model SigLIP2 cukup besar. Tanpa GPU, inferensi di CPU membutuhkan sekitar 10–30 detik per video. Gunakan mode Batch agar proses berjalan di background.

**Apa bedanya Skip dan Flag?**

Skip melewati video sementara dan video masih bisa dilabeli nanti. Flag menandai video sebagai ditolak secara permanen dan mengeluarkannya dari data training.

**Bolehkah folder `cropped_faces/` dihapus?**

Boleh. Folder tersebut hanya berisi cache crop wajah dan akan dibuat ulang secara otomatis. Tidak ada data labeling yang hilang.

**Bagaimana cara mereset label satu video?**

Set semua label ke `0`, pastikan Flag tidak aktif, lalu klik Save & Next.
