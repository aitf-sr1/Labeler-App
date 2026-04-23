# SigLIP2 — Zero-Shot Emotion Inference Pipeline

Dokumen ini menjelaskan dasar teknis dan alasan desain di balik penggunaan SigLIP2 sebagai model inferensi emosi dalam pipeline anotasi.

## Daftar Isi

1. [Model SigLIP2](#model-siglip2)
2. [Alasan Pemilihan SigLIP](#alasan-pemilihan-siglip)
3. [Perbedaan SigLIP dan CLIP: Sigmoid vs Softmax](#perbedaan-siglip-dan-clip-sigmoid-vs-softmax)
4. [Pipeline Inferensi Zero-Shot](#pipeline-inferensi-zero-shot)
5. [Normalisasi Min-Max dan Alasannya](#normalisasi-min-max-dan-alasannya)
6. [Face Detection: MediaPipe BlazeFace](#face-detection-mediapipe-blazeface)
7. [Alur Lengkap: Video ke Label](#alur-lengkap-video-ke-label)
8. [Parameter Inferensi](#parameter-inferensi)
9. [Keterbatasan](#keterbatasan)

## Model SigLIP2

SigLIP2 (Sigmoid Language-Image Pre-training, versi 2) adalah model Vision-Language yang dikembangkan oleh Google DeepMind. Model menerima pasangan gambar dan teks, kemudian menghasilkan skor kesesuaian antar keduanya tanpa perlu fine-tuning untuk tugas spesifik — pendekatan ini disebut **zero-shot inference**.

Model yang digunakan dalam pipeline ini:

```
google/siglip2-base-patch16-224
```

Spesifikasi: arsitektur base, patch size 16×16 piksel, resolusi input 224×224 piksel. Model ini menerima batch gambar dan batch teks secara bersamaan, lalu menghasilkan matriks logit berukuran `[n_images × n_texts]`.

## Alasan Pemilihan SigLIP

Dataset yang digunakan tidak menyediakan label emosi per-frame yang bisa digunakan untuk melatih classifier dari awal. SigLIP dipilih karena memungkinkan inferensi hanya berdasarkan deskripsi teks tanpa data berlabel.

| Pertimbangan | Penjelasan |
|---|---|
| Zero-shot capable | Tidak membutuhkan data training berlabel |
| Fleksibel | Prompt dapat diubah kapan saja tanpa retraining |
| Pretrained pada skala besar | Dilatih pada miliaran pasangan gambar-teks |
| Multi-label friendly | Satu gambar bisa memiliki skor tinggi untuk beberapa label sekaligus |

**Alternatif yang dipertimbangkan:**

- **DeepFace / FER** — hanya mengenali 6–7 ekspresi dasar (happy, sad, angry, dll), tidak sesuai dengan label emosi yang dibutuhkan
- **CLIP** — pendahulu SigLIP dengan fungsi loss yang berbeda (lihat bagian berikutnya)
- **Fine-tuned classifier** — membutuhkan data berlabel yang tidak tersedia di fase anotasi awal

## Perbedaan SigLIP dan CLIP: Sigmoid vs Softmax

Perbedaan arsitektural paling mendasar antara SigLIP dan CLIP terletak pada fungsi loss yang digunakan selama pre-training.

**CLIP menggunakan InfoNCE loss dengan softmax.**
Setiap gambar dalam batch bersaing dengan seluruh teks dalam batch yang sama. Skor akhir bersifat relatif:

```
skor = softmax(logits)   # nilai selalu berjumlah 1 dalam satu baris
```

Konsekuensinya, jika semua teks tidak relevan dengan gambar, salah satu teks tetap mendapat skor tertinggi karena softmax memaksa distribusi probabilitas. Ini tidak sesuai untuk skenario multi-label atau pertanyaan absolut seperti "apakah gambar ini menunjukkan boredom?".

**SigLIP menggunakan sigmoid loss.**
Setiap pasangan gambar-teks dievaluasi secara independen tanpa bersaing dengan pasangan lain:

```
skor = sigmoid(logit)    # nilai antara 0 dan 1 per pasangan, independen
```

Satu gambar dapat memiliki skor tinggi untuk beberapa label sekaligus. Ini sesuai dengan sifat emosi dalam konteks pembelajaran yang bersifat multi-label — seorang siswa bisa terlihat sekaligus bosan dan bingung.

## Pipeline Inferensi Zero-Shot

Untuk setiap video, inferensi menerima input berupa 16 frame (PIL.Image, resolusi 512×512 setelah crop wajah), 4 grup prompt teks (satu grup per label, masing-masing 6 deskripsi), dan 4 nilai threshold.

**Langkah 1: Persiapan batch teks**

Semua prompt dari semua label digabung menjadi satu list flat. Index posisi tiap label dalam list dicatat untuk dipakai kemudian.

```python
# 4 label × 6 prompt = 24 teks total
all_texts    = [prompt_boredom_1, ..., prompt_frustration_6]
group_indices = [[0-5], [6-11], [12-17], [18-23]]
```

**Langkah 2: Forward pass**

```python
inputs = processor(text=all_texts, images=pil_images, padding="max_length")
logits_per_image = model(**inputs).logits_per_image  # [16, 24]
```

Setiap elemen `logits[i][j]` adalah nilai mentah yang merepresentasikan kesesuaian frame ke-i dengan teks ke-j.

**Langkah 3: Normalisasi min-max per frame**

```python
logits_min  = logits.min(dim=1, keepdim=True).values
logits_max  = logits.max(dim=1, keepdim=True).values
norm_logits = (logits - logits_min) / (logits_max - logits_min + 1e-8)
```

Alasan normalisasi ini dijabarkan di bagian berikutnya.

**Langkah 4: Agregasi skor per label**

Untuk setiap label, rata-ratakan skor dari semua prompt yang termasuk dalam label tersebut, untuk setiap frame:

```python
# contoh untuk Boredom (prompt di index 0–5)
scores = [norm_logits[f][0:6].mean().item() for f in range(16)]
```

Hasilnya adalah 16 skor per label, masing-masing dalam rentang 0–1.

**Langkah 5: Voting dan prediksi akhir**

```python
avg_score   = mean(scores)
vote_pos    = sum(1 for s in scores if s >= threshold)
frame_preds = [1 if s >= threshold else 0 for s in scores]
prediction  = 1 if avg_score >= threshold else 0
```

Prediksi akhir (`prediction`) ditentukan dari `avg_score`. Voting per-frame hanya digunakan untuk visualisasi di galeri.

**Output per label:**

```python
{
    "prediction":   int,          # 0 atau 1
    "avg_score":    float,        # 0.0 – 1.0
    "vote_pos":     int,          # jumlah frame positif
    "vote_neg":     int,          # jumlah frame negatif
    "frame_scores": list[float],  # skor tiap frame
    "frame_preds":  list[int],    # prediksi tiap frame
}
```

## Normalisasi Min-Max dan Alasannya

Meskipun SigLIP dilatih dengan sigmoid, logit mentah yang dihasilkan model memiliki skala yang tidak konsisten antar-video. Untuk satu video logit bisa berkisar antara –20 hingga +20, untuk video lain antara –5 hingga +5. Jika sigmoid diterapkan langsung pada logit, threshold yang sama tidak akan berperilaku konsisten di seluruh dataset.

Softmax juga tidak cocok karena memaksa semua skor berjumlah 1. Jika semua 24 prompt tidak relevan untuk suatu frame, softmax tetap mendistribusikan probabilitas dan menghasilkan skor yang menyesatkan.

Normalisasi min-max per-frame memberikan interpretasi yang lebih stabil: skor 1.0 berarti prompt tersebut paling cocok di antara semua prompt untuk frame tersebut, dan skor 0.0 berarti paling tidak cocok. Threshold berperilaku konsisten untuk semua video.

Pendekatan ini bukan yang paling tepat secara teoritis, tetapi secara empiris memberikan hasil yang lebih dapat dikontrol dan diinterpretasikan oleh labeler.

## Face Detection: MediaPipe BlazeFace

Sebelum dikirim ke SigLIP, setiap frame di-crop pada area wajah. Ini penting karena SigLIP dilatih pada gambar umum — membatasi input ke area wajah mengurangi konteks yang tidak relevan seperti background dan pakaian.

**Model yang digunakan:** MediaPipe BlazeFace Short Range, model deteksi wajah ringan yang dikembangkan Google untuk mobile. Diunduh otomatis ke `~/.cache/siglip_labeler/` pada pertama kali dijalankan.

**Proses crop (`crop_face` di `core/face_detector.py`):**

```
frame BGR
    -> konversi ke RGB
    -> deteksi wajah dengan BlazeFace
    -> jika wajah ditemukan:
        -> pilih bounding box terbesar
        -> tambahkan padding 60% dari ukuran bounding box
        -> square crop di sekitar wajah, koordinat di-clamp ke batas frame
    -> jika tidak ada wajah:
        -> center crop 80% dari sisi terpendek frame
    -> return crop BGR
```

Padding 60% dipilih agar seluruh kepala dan sebagian leher masuk dalam crop — postur kepala dan arah pandang membawa informasi emosi yang tidak kalah penting dibanding ekspresi wajah.

Fallback ke center crop digunakan agar frame yang tidak terdeteksi wajahnya (blur, siswa berbalik, pencahayaan rendah) tidak dibuang. Center crop memberikan estimasi posisi wajah terbaik untuk video conference format yang umumnya menempatkan subjek di tengah frame.

Hasil crop disimpan sebagai file JPEG 512×512 di `hasil_label6/cropped_faces/`. Pada video yang sudah pernah diproses, cache ini digunakan langsung tanpa menjalankan ulang deteksi wajah.

## Alur Lengkap: Video ke Label

```
file .mp4
    |
extract_16_frames()
    ambil frame di posisi 0/16, 1/16, ..., 15/16 dari total durasi
    output: 16 frame BGR
    |
crop_face() per frame (MediaPipe BlazeFace)
    detect wajah -> square crop dengan padding 60%
    resize ke 512x512 -> simpan ke disk (cache)
    output: 16 PIL.Image RGB
    |
run_siglip_on_frames()
    processor() -> tokenize 24 prompt + encode 16 gambar -> tensor
    forward pass -> logits [16 x 24]
    normalisasi min-max per frame -> norm_logits [16 x 24]
    untuk setiap label:
        rata-rata norm_logits dari 6 prompt label tersebut per frame
        voting dan avg_score vs threshold
    output: prediksi per label
    |
_apply_siglip_result()
    tulis frame_preds ke frame_annotations
    tulis seluruh hasil ke batch_history
    save_batch_history() -> disk
    |
annotations_bener.csv + frame_annotations.json
```

## Parameter Inferensi

**Prompt**

Deskripsi teks untuk setiap label dapat diubah di panel kanan UI pada textbox "Positive prompt". Nilai default tersimpan di `ui/constants.py`.

Karakteristik prompt yang baik:
- Menggambarkan kondisi yang terlihat secara visual, bukan kondisi internal
- Spesifik dan tidak ambigu antar label
- Variasi deskripsi yang beragam meningkatkan robustness

**Threshold**

Slider threshold mengontrol sensitivitas prediksi. Nilai default 0.50.

- Threshold tinggi (≥ 0.7): prediksi lebih konservatif, lebih banyak label `0`
- Threshold rendah (≤ 0.3): prediksi lebih sensitif, risiko false positive lebih tinggi

Setiap label dapat memiliki threshold berbeda. Label yang jarang muncul seperti Frustration dapat menggunakan threshold lebih rendah untuk meningkatkan recall.

## Keterbatasan

SigLIP2 adalah model vision-language generalis, bukan model yang dilatih khusus untuk deteksi emosi dalam konteks pembelajaran. Beberapa hal yang perlu diperhatikan:

- Akurasi sangat bergantung pada kualitas dan spesifisitas prompt
- Emosi subtle seperti boredom ringan lebih sulit dideteksi dibanding ekspresi yang ekspresif
- Frame tanpa wajah terdeteksi menggunakan fallback center crop yang menurunkan akurasi
- Inferensi di CPU membutuhkan 10–30 detik per video tergantung hardware

Hasil inferensi sebaiknya diverifikasi secara manual sebelum digunakan sebagai ground truth untuk training model berikutnya.
