# Panduan Setup — Menyiapkan & Menjalankan Labeler App

Dokumen ini menjelaskan **semua yang perlu disiapkan** untuk menjalankan aplikasi,
dari nol, lengkap dengan perintah `pip` yang tinggal disalin.

---

## 1. Prasyarat sistem

| Kebutuhan | Keterangan |
|---|---|
| **Python 3.10 – 3.12** | Disarankan 3.11. Cek: `python3 --version` |
| **OS** | Linux / Windows / macOS (dikembangkan & diuji di Linux) |
| **RAM** | Minimal 8 GB (model SigLIP + MediaPipe) |
| **GPU (opsional)** | NVIDIA + CUDA → inferensi SigLIP jauh lebih cepat. CPU tetap bisa, hanya lebih lambat. |
| **Koneksi internet** | Sekali saja, untuk mengunduh model `google/siglip2-base-patch16-224` (otomatis saat pertama jalan). |
| **Tkinter** | Biasanya sudah ada. Jika error `No module named _tkinter` di Linux: `sudo apt install python3-tk`. |

---

## 2. Membuat virtual environment + memasang library

Dari dalam folder aplikasi (`Labeler-App-Siglip-2/`):

```bash
# 1) buat & aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) (GPU NVIDIA saja) pasang PyTorch sesuai CUDA Anda DULU, contoh CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    (CPU saja? lewati langkah ini — torch versi CPU ikut terpasang di langkah 3)

# 3) pasang semua sisanya sekaligus
pip install -r requirements.txt
```

Selesai. Semua library yang dibutuhkan aplikasi sudah terpasang.

> Model SigLIP **tidak** perlu diunduh manual — `transformers` mengunduhnya otomatis
> ke cache (`~/.cache/huggingface`) saat inferensi pertama. MediaPipe membawa
> model FaceLandmarker/HandLandmarker-nya sendiri.

---

## 3. Konfigurasi `.env`

Salin contoh dan sesuaikan:

```bash
cp .env.example .env
```

Isi yang penting:

| Variabel | Arti |
|---|---|
| `OUTPUT_DIR` | Folder tujuan semua output (CSV, cropped_faces, cache). Boleh absolut. |
| `SIGLIP_MODEL_ID` | Default `google/siglip2-base-patch16-224`. |
| `BLENDSHAPE_SOURCE` | `mediapipe` (default) atau `mp_blendshapes`. |
| `PREPROCESS_WORKERS` | Jumlah thread CPU untuk ekstraksi frame (turunkan bila RAM kecil). |
| `SIGLIP_BATCH_VIDEOS` | Berapa video digabung per forward-pass GPU. |

---

## 4. Menjalankan aplikasi

```bash
source .venv/bin/activate
python app.py
```

Lalu klik **Buka Folder** dan pilih folder dataset video. Panduan pemakaian UI ada
di [README.md](../README.md) bagian "Panduan Penggunaan UI".

---

## 5. (Opsional) Augmentasi LivePortrait — panel "LP Transform"

Fitur augmentasi ekspresi (lihat [LP_TRANSFORM.md](LP_TRANSFORM.md)) memakai
[LivePortrait](https://github.com/KwaiVGI/LivePortrait), yang butuh **versi library
berbeda** (numpy 1.26 + torch 2.3). Karena itu ia dipasang di **environment terpisah**
(`.venv-lp`) agar tidak bentrok dengan `.venv` aplikasi.

Ringkas langkahnya:

```bash
# di folder proyek (yang berisi 4-Create/)
python3 -m venv .venv-lp
source .venv-lp/bin/activate
# clone LivePortrait ke 4-Create/LivePortrait lalu pasang requirements-nya:
#   git clone https://github.com/KwaiVGI/LivePortrait 4-Create/LivePortrait
#   pip install -r 4-Create/LivePortrait/requirements.txt
#   (unduh bobot HF: KwaiVGI/LivePortrait → 4-Create/LivePortrait/pretrained_weights)
deactivate
```

Lalu di `4-Create/.env` set:

```
LIVEPORTRAIT_PYTHON=/path/ke/.venv-lp/bin/python
LP_DRIVING_DIR=/path/ke/4-Create/refrensi
LP_EXTRA_FLAGS=--flag-crop-driving-video --driving-option expression-friendly --no-flag-relative-motion
```

Aplikasi menjalankan LivePortrait lewat **worker persisten** (`4-Create/lp_worker.py`)
memakai `LIVEPORTRAIT_PYTHON` — model di-load sekali, dipakai ulang (cepat). Bila
worker gagal start, aplikasi otomatis fallback ke mode subprocess sekali-jalan.

---

## 6. Masalah umum

| Gejala | Solusi |
|---|---|
| `No module named customtkinter` | `pip install -r requirements.txt` belum dijalankan / venv belum aktif. |
| `No module named _tkinter` (Linux) | `sudo apt install python3-tk`. |
| SigLIP lambat / pakai CPU | Pasang torch versi CUDA (langkah 2) dan pastikan GPU terdeteksi: `python -c "import torch; print(torch.cuda.is_available())"`. |
| Unduhan model lambat di run pertama | Normal — model di-cache setelah unduhan pertama. |
| Panel LP Transform error / lambat | Pastikan `.venv-lp` + `LIVEPORTRAIT_PYTHON` benar; cek log `4-Create/lp_worker.log`. |
