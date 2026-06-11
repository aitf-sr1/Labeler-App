# Panel LP Transform — Augmentasi Ekspresi di Dalam Aplikasi

Panel ini menambah sampel kelas minoritas (**Confusion**, **Frustration**, dll.) dengan
**mengubah ekspresi wajah** memakai [LivePortrait](https://github.com/KwaiVGI/LivePortrait),
tanpa keluar dari aplikasi. Identitas orang tetap; hanya ekspresinya yang diganti mengikuti
**video acuan (driving)**.

Ada **dua sumber** yang bisa diaugmentasi:
1. **Frame dari dataset video** yang sedang dilabeli (orang/UUID sudah ada di dataset).
2. **Dataset wajah baru** (mis. foto open-source) — tiap foto dianggap **orang baru**, untuk
   menambah ragam orang agar model tidak overfit.

Karena hasil augmentasi memakai UUID orang yang sama (atau orang baru yang unik), ia hanya
masuk split **train** — tidak bocor ke val/test.

---

## Alur singkat (end-to-end)

```
            ┌─ Sumber A: frame video dataset (tandai "LP Transform" di galeri)
 SUMBER ────┤
            └─ Sumber B: dataset wajah baru (folder foto, tiap foto = orang baru)
                                   │
 DRIVING  ── folder video acuan ekspresi (Confusion1.mp4, Frustration1.mp4, …)
                                   │
 PROSES   ── Proses Frame Ini  |  Batch Semua  |  Proses Wajah Terpilih
                                   │   (worker LivePortrait persisten — model load sekali)
 PILIH    ── geser slider hasil → tandai frame index yang dipakai (ekspresi konsisten)
                                   │
 TINJAU   ── grid + pemeriksa BESAR → Deteksi AI (bandingkan vs label manual) → tolak/terima
                                   │
 BUAT     ── pilih komposisi (Tanpa LP / Dengan LP / LP + Dataset Baru) → Buat Dataset
 DATASET     (output Label2d_merged_{komposisi}/, non-destruktif, bisa Undo)
```

---

## 1. Membuka panel

Di galeri (panel kiri) klik tab **LP Transform**, atau tombol **Buka Panel ▸** di bar
"Frame LP Transform". Tiap frame galeri punya tombol **LP Transform** (amber) untuk menandai
frame netral yang ingin diubah. Tombol **Hapus Semua Tanda** di header panel menghapus semua
tanda sekaligus.

Bagian **SUMBER** menampilkan UUID + frame yang sedang aktif; **◀ Sebelumnya / Berikutnya ▶**
berpindah antar tanda. Sumber selalu mengikuti video yang sedang dilihat.

---

## 2. Folder video driving (acuan ekspresi)

Driving **berbasis folder**. Beri nama file sesuai **NAMA EMOSI LENGKAP + nomor urut**:

```
refrensi/
  Confusion1.mp4    Confusion2.mp4
  Frustration1.mp4
  Boredom1.mp4
  Engagement1.mp4
```

Klik **Pindai** untuk mendeteksi. Per emosi pilih **Semua** atau **satu** video via dropdown.
Folder default dari `LP_DRIVING_DIR` (default `4-Create/refrensi`).

---

## 3. Memproses

1. Pilih satu/lebih **emosi** (pill).
2. **Proses Frame Ini** — generate satu video hasil dari frame sumber aktif, untuk dicoba &
   dipilih frame terbaiknya. **Hasil di-cache**: kalau pindah frame lalu kembali, hasilnya
   muncul lagi tanpa proses ulang.
3. **Batch Semua** — proses **semua** frame bertanda dengan **satu** video driving; frame index
   yang Anda tandai dipakai untuk semua (ekspresi konsisten). Ganti driving → Batch lagi.
4. **Batal** menghentikan; **Reset** mengosongkan pilihan & hasil (file tidak dihapus).

Saat proses berjalan, muncul **indikator loading beranimasi** supaya jelas tidak hang.

### Kecepatan — model di-load sekali
LivePortrait dijalankan oleh **worker persisten** (`4-Create/lp_worker.py`, env `.venv-lp`):
model di-load **sekali** lalu dipakai ulang. Bila worker gagal, otomatis fallback ke subprocess
sekali-jalan. Akses worker diserialisasi + diberi timeout → tanpa race/hang yang membekukan UI.

---

## 4. Pratinjau & pilih frame

Tiga gambar bersisian **SUMBER | DRIVING | HASIL** (frame & UUID sinkron). Overlay landmark
(viz) hanya muncul bila saklar **Viz** di topbar aktif, dihitung dengan penundaan singkat agar
menggeser slider tetap mulus. Tiap kolom punya tombol **Unduh Gambar**.

Geser slider video hasil, lalu **＋ Tandai Frame** (pilih frame tertentu, mis. 2, 4, 8) atau
**Tandai Merata** (N dari "Jumlah target"). **Simpan Frame Tertanda** menyimpan ke dataset.
Frame index yang ditandai juga dipakai saat **Batch Semua** dan **Proses Wajah Terpilih**.

**Kunci posisi proporsional** (checkbox, default aktif): kalau **video driving diganti** lalu
diproses ulang, frame yang sudah ditandai **otomatis dipetakan ulang** ke posisi proporsional
di video baru — **jumlah frame tetap sama, hanya letaknya yang menyesuaikan** panjang/timing
driving baru. Contoh: tandai 2 frame di driving 10-frame (posisi ~22% & ~89%) → ganti ke driving
50-frame → otomatis jadi 2 frame di ~frame 11 & ~44. Saat Batch, tiap video hasil memetakan
fraksi ini ke index-nya sendiri, jadi konsisten walau tiap driving beda panjang. Matikan
checkbox bila ingin menandai dari nol tiap ganti driving.

Hasil disimpan ke `{OUTPUT_DIR}/augmented/liveportrait_app/{uuid}/{emosi}/...jpg`.

---

## 5. Dataset wajah baru (tiap foto = orang baru)

Untuk menambah ragam orang (mis. dataset wajah open-source) agar model belajar deteksi emosi
lebih umum / tidak overfit:

1. Pilih **folder** berisi foto wajah → **Pindai**. Muncul grid thumbnail.
2. **Centang** foto yang ingin diproses (klik thumbnail; ada **Pilih Semua** / **Kosongkan**).
3. Pastikan **emosi**, **driving**, dan **frame index** sudah diset seperti alur biasa.
4. **Proses Wajah Terpilih** → tiap foto diberi UUID baru (`newface-…`) = **orang baru**,
   lalu diaugmentasi dan disimpan seperti hasil LP biasa.

---

## 6. Tinjau & label (cek sebelum buat dataset)

- **Muat / Refresh** menyiapkan **seluruh** gambar hasil (ribuan pun cepat — thumbnail dimuat
  per-halaman di latar, tidak sekaligus).
- **Navigasi (untuk ribuan gambar):** pemeriksa besar punya **◀ Sebelumnya / Berikutnya ▶**,
  kotak **Loncat** ke nomor tertentu, dan penunjuk posisi **"12 / 1340"**. Grid thumbnail
  **berhalaman** (◀ Halaman / Halaman ▶) — klik thumbnail untuk lompat ke gambar itu di pemeriksa.
  Jadi tidak perlu scroll satu-satu mencari di antara ribuan hasil.
- Klik **1×** thumbnail → tampil **BESAR** di pemeriksa. Di sana ditampilkan **dua hal**:
  - **Deteksi AI** (chip, read-only) — hasil SigLIP+MediaPipe.
  - **Label final manual** (pill, bisa diklik) — yang dipakai saat buat dataset.
  Ini membantu memastikan tidak ada emosi lain selain yang diinginkan.
- **Deteksi AI Semua** mengisi chip AI untuk semua hasil (disimpan di `lp_ai_labels.json`,
  terpisah dari label final manual `lp_labels.json`). Berjalan **inkremental** — gambar yang
  sudah berlabel AI dilewati, jadi menjalankan ulang cepat.
- Klik **2×** gambar = **tolak/terima** (merah = ditolak, tidak ikut dataset). **Buang Ditolak →
  _trash** memindahkan gambar yang ditolak ke folder `_trash/` (BUKAN hapus permanen — bisa
  dipulihkan manual bila berubah pikiran).
- **Filter** tampilan: Semua / Diterima / Ditolak / **AI != target** / per-emosi — supaya
  pemeriksaan ribuan hasil terarah (mis. langsung lihat yang dicurigai salah).
- **Auto-Tolak (AI != target)**: tandai tolak otomatis semua hasil yang menurut deteksi AI
  **tidak mengandung emosi targetnya** (jalankan "Deteksi AI Semua" dulu). QA ribuan gambar
  dalam satu klik; ada konfirmasi jumlah, tidak menghapus file, dan tiap gambar bisa
  di-batal-tolak lagi. Setelahnya pakai filter "Ditolak" untuk meninjau keputusan AI.
- **Statistik Dataset**: tampilkan distribusi label per emosi — `asli` (Label2d train) +
  `LP diterima` (yang akan masuk dataset, termasuk berapa dari wajah baru) + `ditolak`.
  Pakai ini untuk menilai apakah kelas minoritas (Confusion/Frustration) sudah seimbang
  sebelum Buat Dataset.

---

## 7. Buat dataset (pilih komposisi) & Undo

Pilih **komposisi** lewat bullet/radio, lalu **Buat Dataset**:

| Komposisi | Isi |
|---|---|
| **Tanpa LP** | Dataset asli saja (tanpa augmentasi). |
| **Dengan LP** | Asli + hasil LP dari frame video dataset. |
| **LP + Dataset Wajah Baru** | Asli + hasil LP video + hasil LP dataset wajah baru. |

Output **non-destruktif** ke `Label2d_merged_{komposisi}/` (mis. `Label2d_merged_lp/`):
train = asli + sintetik, val/test disalin apa adanya, kolom `synthetic` (1 = hasil LP).
`Label2d` asli tidak diubah. **Undo / Hapus** menghapus folder `Label2d_merged_*` (asli aman).

Tiga komposisi disimpan terpisah → mudah membandingkan (mis. cek apakah augmentasi membantu
atau malah overfit).

---

## File yang dihasilkan (di bawah `OUTPUT_DIR`)

| File / folder | Isi |
|---|---|
| `augmented/liveportrait_app/{uuid}/{emosi}/*.jpg` | Frame hasil generate |
| `lp_labels.json` | Label final manual per gambar (dipakai saat buat dataset) |
| `lp_ai_labels.json` | Hasil deteksi AI per gambar (pembanding) |
| `lp_review.json` | Daftar gambar yang ditolak |
| `Label2d_merged_{base,lp,lp_new}/` | Dataset gabungan per komposisi |
| `augment_marks.json` → `lp_transform_frames` | Frame sumber video yang ditandai LP |

## Kode terkait

| File | Peran |
|---|---|
| `ui/lp_panel.py` | Seluruh UI panel LP Transform |
| `app.py` (metode `_lp_*`) | Proses, worker persisten, label AI, dataset wajah, buat/undo dataset |
| `4-Create/lp_worker.py` | Worker LivePortrait persisten (model load sekali) |
| `4-Create/.env` (`LP_DRIVING_*`, `LP_DRIVING_DIR`, `LP_FACES_DIR`, `LP_EXTRA_FLAGS`) | Konfigurasi |
