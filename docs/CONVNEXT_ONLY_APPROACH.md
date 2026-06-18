# Arah Baru: AI-assist Berbasis ConvNeXt (Tanpa SigLIP, Tahan Distorsi)

Branch `feat/convnext-only-no-siglip`. Dokumen rancangan. **Belum mengubah
pipeline scoring yang ada** agar branch lain tidak rusak; ini peta jalannya.

## Masalah pendekatan sekarang
1. **SigLIP2** (zero-shot vision-language) berat, lambat, dan bukan dilatih khusus
   untuk 4 state belajar. Model ConvNeXt yang sudah dilatih sendiri lebih akurat
   untuk tugas ini (F1 macro 0.824 di test sendiri).
2. **AU eksplisit via MediaPipe FaceLandmarker/blendshape sering gagal** saat ada
   distorsi kamera (wajah tidak terdeteksi) sehingga fitur AU tidak andal di lapangan.

## Keputusan
| Komponen | Sekarang | Diganti jadi |
|----------|----------|--------------|
| Skor emosi | SigLIP2 + landmark/AU hybrid + rules | **ConvNeXt ONNX** (model yang sudah dilatih) langsung |
| Pembeda Conf/Frus | AU alis dari MediaPipe | **Frame-difference** (selisih antar-frame, pixel-level) |
| Crop wajah | MediaPipe BlazeFace | **SCRFD/RetinaFace (InsightFace)** atau YOLO-face (lebih tahan distorsi) |
| Baseline per-orang | AU deviasi dari netral (butuh landmark) | opsional: appearance-conditioned (crop netral), bukan AU |

## Mengapa frame-difference valid (ringkas; detail di FRAMEDIFF_BASIS.md)
Confusion vs Frustration dibedakan oleh **arah gerak alis** (Craig 2008: AU4 turun
vs AU1+AU2 naik). FACS sendiri mendefinisikan AU sebagai **gerakan otot** (Bartlett
1999: *"quantifying facial movement in terms of component actions"*), dan ekspresi
spontan dibedakan oleh **dynamics of the movement** (Bartlett 2006). Frame-difference
adalah manifestasi pixel dari gerakan itu, **tidak butuh landmark** (tahan distorsi),
jadi tetap "melihat dari AU" secara sah. Engagement tetap dari pixel statis
(Whitehill 2014: *"static pixels, not the motion per se"*), maka model memakai input
**dwi-aliran: appearance (RGB) + frame-difference**. Semua kutipan verbatim &
tervalidasi ke PDF ada di `docs/ACADEMIC_BASIS.md` dan `docs/FRAMEDIFF_BASIS.md`.

## Model & training
Kode latih model dwi-aliran sudah ada di repo training:
`3-Training/Lightweight-Model/convnextv2/`:
- `model_framediff.py`   ConvNeXtV2 `in_chans=6` (3 appearance + 3 difference)
- `dataset_framediff.py` loader 6-kanal (referensi selisih = frame pasangan)
- `train_framediff.py`   training mandiri (tak menyentuh train.py lama)
- `FRAMEDIFF_BASIS.md`   dasar akademis (verbatim)
Hasil: `runs/framediff/local/` (`metrics.csv` per-epoch + `checkpoints/best.pth`).

## Langkah integrasi ke app (belum dilakukan, agar tak merusak)
1. Tambah `core/convnext_model.py` (loader ONNX) + `core/convnext_scoring.py`
   (crop -> 6-kanal appearance+diff -> ONNX -> 4 prob -> threshold per-label).
2. Ganti detektor crop ke SCRFD/RetinaFace (InsightFace sudah ada di repo LivePortrait).
3. Beri flag di UI: pilih "ConvNeXt" vs "SigLIP" (transisi aman, bisa dibandingkan).
4. Setelah terverifikasi, pensiunkan jalur SigLIP.

> Catatan: langkah 1-4 sengaja belum dieksekusi otomatis karena menyentuh inti
> scoring; dilakukan terkontrol agar branch `main` dan lainnya tetap aman.
