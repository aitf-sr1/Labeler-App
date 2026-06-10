# Panel LP Transform — Augmentasi Ekspresi di Dalam Aplikasi

Panel ini menambah sampel kelas minoritas (**Confusion** & **Frustration**, atau emosi lain)
dengan cara **mengubah ekspresi wajah** sebuah frame netral memakai
[LivePortrait](https://github.com/KwaiVGI/LivePortrait), tanpa keluar dari aplikasi.

Identitas siswa tetap (wajah orang yang sama), hanya ekspresinya yang diganti mengikuti
sebuah **video acuan (driving)**. Karena frame hasil punya UUID orang yang sama, ia ikut
split **train** (tidak bocor ke val/test).

> Ringkas: *pilih frame netral → pilih video acuan ekspresi → generate → pilih frame
> hasil → cek & label → merge ke dataset.*

---

## 1. Membuka panel

Di galeri frame (panel kiri), klik tab **LP Transform** di kanan atas, atau tombol
**Buka Panel ▸** pada bar "Frame LP Transform".

Tiap frame di galeri punya tombol **LP Transform** (amber). Klik untuk menandai frame
netral mana yang ingin diubah ekspresinya. Tombol Prev/Next di panel berpindah antar
tanda. Sumber yang sedang aktif selalu mengikuti video yang sedang dilihat.

---

## 2. Folder video driving (acuan ekspresi)

Driving **berbasis folder**, bukan path tetap. Taruh video acuan di satu folder; nama
file menentukan **emosi + urutan**:

```
refrensi/
  confuse1.mp4      → Confusion
  confuse2.mp4      → Confusion
  frustration1.mp4  → Frustration
  boredom1.mp4      → Boredom
  engagement1.mp4   → Engagement
```

Kata kunci nama file: `confuse/bingung`, `frustrat/frustrasi`, `bored/bosan`,
`engag/antusias`. Klik **Pindai** untuk mendeteksi. Per emosi bisa pilih **Semua** video
atau **satu** video tertentu lewat dropdown. Folder default diatur lewat `LP_DRIVING_DIR`
(default: `4-Create/refrensi`).

Untuk menambah variasi: cukup taruh file baru dengan nomor berbeda dan urut
(`confuse3.mp4`, dst), lalu Pindai lagi.

---

## 3. Memproses

1. Pilih satu/lebih **emosi** (pill).
2. **Proses Frame Ini** — generate satu video hasil dari frame sumber aktif memakai video
   driving terpilih, untuk dicoba dan dipilih frame terbaiknya secara interaktif.
3. **Batch Semua** — proses **semua** frame bertanda LP dengan **satu** video driving.
   Frame index yang Anda tandai di scrubber dipakai untuk semua video — karena driving-nya
   sama, ekspresi pada frame ke-N akan konsisten di semua hasil. Ganti video driving →
   Batch lagi.
4. **Batal** — hentikan proses berjalan. **Reset** — kosongkan pilihan & hasil (file tidak
   dihapus).

### Kecepatan: model di-load sekali

Model LivePortrait dijalankan oleh **worker persisten** (`4-Create/lp_worker.py`) di
`.venv-lp`. Model di-load **sekali** lalu dipakai ulang untuk semua job, sehingga tidak
ada lag "load model" tiap frame. Bila worker gagal start, aplikasi otomatis fallback ke
mode subprocess sekali-jalan (lebih lambat, tetapi tetap jalan). Worker dimatikan rapi
saat aplikasi ditutup. Akses worker diserialisasi (satu job pada satu waktu) dan diberi
timeout, sehingga tidak ada race condition / hang yang membekukan UI.

---

## 4. Pratinjau & unduh

Tiga gambar besar bersisian: **SUMBER | DRIVING | HASIL**, dengan UUID dan nomor frame
yang sinkron. Overlay landmark (viz) hanya muncul bila saklar **Viz** di topbar aktif, dan
dihitung dengan penundaan singkat agar menggeser slider tetap mulus. Tiap kolom punya
tombol **Unduh Gambar** untuk menyimpan frame yang sedang tampil.

---

## 5. Memilih frame hasil

Geser slider video hasil, lalu:

- **＋ Tandai Frame** — tandai frame yang sedang tampil.
- **Tandai Merata** — tandai N frame tersebar merata (N dari "Jumlah target").
- **Simpan Frame Tertanda** / **Simpan Frame Ini Saja** — simpan ke dataset.

Frame yang ditandai juga dipakai sebagai index ekstraksi saat **Batch Semua**.

Hasil disimpan ke:
`{OUTPUT_DIR}/augmented/liveportrait_app/{uuid}/{emosi}/...jpg`

---

## 6. Tinjau & label (cek sebelum merge)

Sebelum merge, periksa hasil agar tidak ada label yang tak diinginkan:

- **Muat / Refresh** — tampilkan semua gambar hasil (thumbnail dimuat di latar agar UI
  tidak nge-lag). Klik **1x** thumbnail → tampil besar di pemeriksa.
- **Label Semua (AI)** — label semua hasil otomatis dengan SigLIP + MediaPipe (penggaris
  yang sama dengan menu utama). Berguna untuk mendeteksi bila frame ternyata mengandung
  emosi lain selain yang diinginkan.
- **Label manual** — di pemeriksa, klik pill emosi untuk menyetel label final per gambar
  (dengan mutual exclusion Bore↔Eng, Conf↔Frus).
- **Tolak / terima** — klik **2x** gambar (di grid atau pemeriksa). Merah = ditolak, tidak
  ikut merge. **Hapus Ditolak** menghapus file yang ditolak dari disk.

Label final disimpan di `lp_labels.json`; status tolak di `lp_review.json`.

---

## 7. Merge & Undo

- **Merge ke Dataset** — gabungkan gambar hasil yang **diterima + berlabel** ke split
  **train** dataset. Output **non-destruktif** ke folder baru `Label2d_lp_merged/`
  (val/test disalin apa adanya). Ditambah kolom `synthetic` (1 = hasil LP, 0 = asli).
  `Label2d` asli tidak diubah. Bila `Label2d` belum ada, split dibuat otomatis dulu.
- **Undo Merge** — hapus folder `Label2d_lp_merged/`. Karena merge non-destruktif, undo
  aman dan tidak menyentuh data asli.

---

## File yang dihasilkan (di bawah `OUTPUT_DIR`)

| File / folder | Isi |
|---|---|
| `augmented/liveportrait_app/{uuid}/{emosi}/*.jpg` | Frame hasil generate |
| `lp_labels.json` | Label final per gambar hasil (untuk merge) |
| `lp_review.json` | Daftar gambar yang ditolak |
| `Label2d_lp_merged/` | Hasil merge (train asli + sintetik, val/test disalin) |
| `augment_marks.json` → `lp_transform_frames` | Frame sumber yang ditandai LP |

## Kode terkait

| File | Peran |
|---|---|
| `ui/lp_panel.py` | Seluruh UI panel LP Transform |
| `app.py` (metode `_lp_*`) | Proses, worker persisten, label, merge/undo |
| `4-Create/lp_worker.py` | Worker LivePortrait persisten (model load sekali) |
| `4-Create/.env` (`LP_DRIVING_*`, `LP_EXTRA_FLAGS`) | Konfigurasi driving & flag LP |
