# Cara Kerja Perhitungan: Panduan Teknis Lengkap

> **Navigasi:** [README utama](../README.md) · [Rules Editor](RULES_PANEL.md) · [Arsitektur Pipeline](../core/README_SIGLIP.md)

Dokumen ini menjelaskan **dari angka mentah MediaPipe hingga prediksi akhir**, langkah demi langkah dengan contoh angka nyata. Referensi akademis untuk setiap komponen ada di bagian [Referensi & Sumber](#referensi--sumber) di akhir dokumen. Cocok untuk memahami mengapa skor tertentu muncul, atau untuk men-debug hasil yang tidak masuk akal.

---

## Daftar Isi

1. [Output Mentah MediaPipe — Apa yang Keluar dari Model](#1-output-mentah-mediapipe--apa-yang-keluar-dari-model)
2. [Head Pose: Rotation Matrix → Yaw & Pitch](#2-head-pose-rotation-matrix--yaw--pitch)
3. [Iris Offset: Posisi Pupil Relatif ke Mata](#3-iris-offset-posisi-pupil-relatif-ke-mata)
4. [Blendshapes: 52 Koefisien Otot Wajah](#4-blendshapes-52-koefisien-otot-wajah)
5. [Deteksi Tangan & Zonasi](#5-deteksi-tangan--zonasi)
6. [Gaze Deviation — Metrik Dasar Boredom & Engagement](#6-gaze-deviation--metrik-dasar-boredom--engagement)
7. [Scoring BOREDOM — Langkah demi Langkah](#7-scoring-boredom--langkah-demi-langkah)
8. [Scoring ENGAGEMENT — Langkah demi Langkah](#8-scoring-engagement--langkah-demi-langkah)
9. [Scoring CONFUSION — Langkah demi Langkah](#9-scoring-confusion--langkah-demi-langkah)
10. [Scoring FRUSTRATION — Langkah demi Langkah](#10-scoring-frustration--langkah-demi-langkah)
11. [SigLIP Scoring — Dari Teks + Gambar ke Angka](#11-siglip-scoring--dari-teks--gambar-ke-angka)
12. [Hybrid Combine — Gabungkan SigLIP dan Landmark](#12-hybrid-combine--gabungkan-siglip-dan-landmark)
13. [Temporal Restlessness Bonus (Boredom)](#13-temporal-restlessness-bonus-boredom)
14. [Threshold → Voting → Prediksi Akhir](#14-threshold--voting--prediksi-akhir)
15. [Membaca Debug Log](#15-membaca-debug-log)
16. [Contoh Lengkap Satu Frame](#16-contoh-lengkap-satu-frame)

---

## 1. Output Mentah MediaPipe — Apa yang Keluar dari Model

Ketika `analyze_frame(frame_bgr)` dipanggil pada satu frame 512×512, MediaPipe mengembalikan objek `LandmarkResult` yang berisi:

```
LandmarkResult:
  face_found     = True
  yaw            = +12.30     # kepala menoleh kanan 12.3°
  pitch          = -3.50      # kepala sedikit nunduk 3.5°
  iris_x         = +0.2100    # pupil kanan 21% dari center ke tepi mata (kanan)
  iris_y         = -0.0500    # pupil sedikit ke atas 5%
  iris_img_x     = +0.0640    # pusat kedua iris ada di 56.4% dari kiri frame
  iris_img_y     = -0.0120    # pusat iris sedikit di atas tengah frame
  left_iris_px   = (198, 201) # koordinat pixel iris kiri di frame 512×512
  right_iris_px  = (312, 203) # koordinat pixel iris kanan
  blendshapes    = {           # 52 nilai, contoh sebagian:
    "browDownLeft":    0.1200,
    "browDownRight":   0.1400,
    "browInnerUp":     0.0500,
    "eyeBlinkLeft":    0.0800,
    "eyeBlinkRight":   0.0900,
    "eyeLookDownLeft": 0.1100,
    "eyeLookDownRight":0.1000,
    "eyeLookUpLeft":   0.0200,
    "eyeLookUpRight":  0.0300,
    "eyeSquintLeft":   0.1500,
    "eyeSquintRight":  0.1600,
    "jawOpen":         0.0600,
    "mouthSmileLeft":  0.0800,
    "mouthSmileRight": 0.0900,
    "mouthPucker":     0.0400,
    "mouthPressLeft":  0.0300,
    "mouthPressRight": 0.0400,
    "noseSneerLeft":   0.0200,
    "noseSneerRight":  0.0100,
    "cheekSquintLeft": 0.1000,
    "cheekSquintRight":0.1100,
    ...
  }
  hand_forehead  = 0.0000   # tidak ada tangan di zona atas wajah
  hand_chin      = 0.0000   # tidak ada tangan di zona bawah
```

Semua ini tersimpan juga di `raw_cache/{safe_name}.json` untuk keperluan recalculate.

---

## 2. Head Pose: Rotation Matrix → Yaw & Pitch

MediaPipe FaceLandmarker menghasilkan **4×4 facial transformation matrix** yang mewakili orientasi kepala di ruang 3D.

```python
# Contoh matrix (disederhanakan):
matrix = [
  [+0.9816,  +0.0000, -0.1908, 0],   # baris 0
  [+0.0000,  +1.0000,  0.0000, 0],   # baris 1
  [+0.1908,  +0.0000, +0.9816, 0],   # baris 2
  [...]
]
```

Dari 3×3 rotasi (pojok kiri atas matrix), kode mengekstrak sudut Euler:

```python
R = matrix[:3, :3]
sy = sqrt(R[0,0]² + R[1,0]²)       # = sqrt(0.9816² + 0²) = 0.9816

pitch = atan2(-R[2,0], sy)          # = atan2(-0.1908, 0.9816) = -10.9° → ~-11°
yaw   = atan2(R[1,0], R[0,0])      # = atan2(0, 0.9816) = 0°
```

**Konvensi tanda:**
- `yaw` + = kepala menoleh ke **kanan**
- `yaw` - = kepala menoleh ke **kiri**
- `pitch` + = kepala **mendongak** ke atas
- `pitch` - = kepala **nunduk** ke bawah

**Contoh situasi nyata:**

| Posisi kepala | yaw | pitch |
|---|---|---|
| Lurus ke depan | ≈ 0° | ≈ 0° |
| Noleh kanan 30° | ≈ +30° | ≈ 0° |
| Nunduk lihat keyboard | ≈ 0° | ≈ −20° |
| Mendongak ke langit-langit | ≈ 0° | ≈ +25° |
| Noleh kiri sambil nunduk | ≈ −25° | ≈ −15° |

---

## 3. Iris Offset: Posisi Pupil Relatif ke Mata

Iris offset mengukur **ke mana pupil melihat relatif ke struktur mata itu sendiri** — bukan ke frame gambar. Ini penting karena frame sudah di-crop mengikuti wajah, sehingga posisi iris dalam frame hampir selalu di tengah.

### Cara Menghitungnya

Untuk mata kiri (indices sama untuk mata kanan dengan nilai berbeda):

```
Landmark yang dipakai:
  468 = iris kiri (tengah pupil)
  133 = sudut dalam mata kiri (dekat hidung)
   33 = sudut luar mata kiri (dekat pelipis)
  159 = tepi atas kelopak
  145 = tepi bawah kelopak
```

```python
# Posisi pixel dari koordinat ternormalisasi (× ukuran frame 512):
iris_px  = lms[468].x * 512 = 0.387 * 512 = 198.1 px
iris_py  = lms[468].y * 512 = 0.392 * 512 = 200.7 px

inner_px = lms[133].x * 512 = 0.411 * 512 = 210.4 px   # sudut dalam (hidung)
outer_px = lms[33].x  * 512 = 0.340 * 512 = 174.1 px   # sudut luar (pelipis)
top_py   = lms[159].y * 512 = 0.374 * 512 = 191.5 px
bot_py   = lms[145].y * 512 = 0.409 * 512 = 209.4 px

# Pusat geometri mata:
cx = (inner_px + outer_px) / 2 = (210.4 + 174.1) / 2 = 192.25 px
cy = (top_py + bot_py) / 2     = (191.5 + 209.4) / 2 = 200.45 px

# Setengah lebar/tinggi mata (radius):
hw = |inner_px - outer_px| / 2 = |210.4 - 174.1| / 2 = 18.15 px
hh = |top_py - bot_py| / 2     = |191.5 - 209.4| / 2 = 8.95 px

# Offset ternormalisasi:
iris_x_left = (198.1 - 192.25) / 18.15 = +0.322   # pupil kanan dari center mata
iris_y_left = (200.7 - 200.45) / 8.95  = +0.028   # pupil sedikit ke bawah
```

Proses sama untuk mata kanan, lalu rata-rata:
```python
iris_x = clip((iris_x_left + iris_x_right) / 2, -1, 1)
iris_y = clip((iris_y_left + iris_y_right) / 2, -1, 1)
```

**Interpretasi nilai `iris_x`:**

| iris_x | Arti |
|---|---|
| −1.0 | Pupil di ujung paling kiri mata (lirik kiri ekstrem) |
| −0.3 | Pupil sedikit ke kiri dari center |
| 0.0 | Pupil di center mata (menatap lurus) |
| +0.3 | Pupil sedikit ke kanan dari center |
| +1.0 | Pupil di ujung paling kanan mata (lirik kanan ekstrem) |

**Interpretasi nilai `iris_y`:**

| iris_y | Arti |
|---|---|
| −1.0 | Pupil di paling atas (lirik ke atas maksimum) |
| −0.15 | Pupil sedikit ke atas (ambang Confusion) |
| 0.0 | Pupil di center vertikal |
| +1.0 | Pupil di paling bawah (lirik ke bawah) |

### iris_img_x vs iris_x — Perbedaan Penting

```python
# iris_x: posisi pupil RELATIF KE SUDUT MATA (dipakai untuk scoring)
iris_x = (iris_px - center_mata_x) / radius_mata_x

# iris_img_x: posisi pupil RELATIF KE PUSAT FRAME (tidak dipakai untuk scoring)
iris_img_x = (rata_rata_posisi_iris_dalam_frame) - 0.5
```

Karena frame sudah di-crop mengikuti wajah, `iris_img_x` selalu ≈ 0 dan **tidak dipakai** dalam scoring. Hanya `iris_x` yang relevan.

---

## 4. Blendshapes: 52 Koefisien Otot Wajah

MediaPipe FaceLandmarker menghasilkan **52 blendshape coefficient** yang masing-masing merepresentasikan gerakan otot wajah tertentu. Semua nilai antara **0.0** (otot rileks) dan **1.0** (otot berkontraksi penuh).

### Blendshapes yang Dipakai dalam Scoring

| Nama Blendshape | Arti Fisik | Dipakai untuk |
|---|---|---|
| `browDownLeft` / `Right` | Alis kiri/kanan turun (mengernyit) | Confusion, Frustration |
| `browInnerUp` | Alis bagian tengah naik (pola "∧") | Confusion |
| `eyeBlinkLeft` / `Right` | Kelopak mata menutup | Boredom (mata ngantuk) |
| `eyeLookDownLeft` / `Right` | Bola mata bergerak ke bawah | Boredom (fallback gaze_v) |
| `eyeLookUpLeft` / `Right` | Bola mata bergerak ke atas | Confusion |
| `eyeSquintLeft` / `Right` | Mata menyipit (otot tegang, bukan mengantuk) | Frustration |
| `jawOpen` | Rahang turun (mulut terbuka) | Boredom (menguap), Confusion (mangap), Frustration |
| `mouthSmileLeft` / `Right` | Sudut bibir naik (senyum) | Penalti Confusion |
| `mouthPucker` | Bibir maju/mengerut | Confusion |
| `mouthPressLeft` / `Right` | Bibir ditekan keras | Frustration |
| `noseSneerLeft` / `Right` | Hidung berkerut (ekspresi jijik/marah) | Frustration (sinyal terkuat) |
| `cheekSquintLeft` / `Right` | Pipi menegang ke atas | Frustration |

### Blendshapes yang Tidak Dipakai (ada di output tapi diabaikan)

Contoh: `mouthLeft`, `mouthRight`, `mouthRollLower`, `mouthLowerDownLeft`, `tongueOut`, dll. Blendshape ini ada di output MediaPipe tapi tidak berkontribusi ke skor emosi di sistem ini.

### Contoh Nilai untuk Berbagai Ekspresi

```
Ekspresi netral:
  browDownLeft = 0.05, browInnerUp = 0.08, jawOpen = 0.02, eyeBlink = 0.05

Sedang menguap (Boredom):
  jawOpen = 0.65, eyeBlinkLeft = 0.35, eyeBlinkRight = 0.30

Wajah bingung:
  browDownLeft = 0.38, browDownRight = 0.35, browInnerUp = 0.42,
  eyeLookUpLeft = 0.31, jawOpen = 0.18

Wajah frustrasi:
  browDownLeft = 0.52, noseSneerLeft = 0.35, mouthPressLeft = 0.48,
  eyeSquintLeft = 0.44, eyeSquintRight = 0.41
```

---

## 5. Deteksi Tangan & Zonasi

Tangan dideteksi dari **full frame video asli** (sebelum crop wajah), lalu koordinatnya di-remap ke ruang crop 512×512.

### Remap Koordinat dari Full Frame ke Crop

```
Full frame: 1920 × 1080 px
Face bbox dari BlazeFace: x1=400, y1=150, x2=700, y2=550
  → face_w = 300 px, face_h = 400 px

Titik tangan di full frame: x_full = 580 px, y_full = 350 px

Konversi ke koordinat relatif dalam crop wajah (0.0–1.0):
  px_crop = (580 - 400) / 300 = 0.600
  py_crop = (350 - 150) / 400 = 0.500
  → koordinat dalam crop: (0.60, 0.50) = bagian tengah-kanan crop wajah
```

### Zonasi Tangan dalam Crop 512×512

```
Frame crop 512×512 dibagi vertikal:

  y = 0.00  ┌─────────────────────┐
            │  ZONA ATAS (0–0.25) │ ← tangan di sini: menggaruk kepala
  y = 0.25  ├─────────────────────┤    → hand_chin (Confusion)
            │  ZONA TENGAH        │ ← tangan di sini: menutupi/menopang wajah
  y = 0.55  ├─────────────────────┤    → hand_mid_bot (Frustration)
            │  ZONA BAWAH         │ ← tangan di sini: menopang dagu/leher
  y = 1.20  └─────────────────────┘    → hand_mid_bot (Frustration)
```

### Cara Menghitung Skor Tangan

```python
# Contoh: 1 tangan dengan 21 titik, semua terdeteksi
pts = [(0.51, 0.08), (0.53, 0.11), ..., (0.48, 0.22)]  # 21 titik

pts_top     = jumlah titik dengan y antara -0.20 dan 0.25 = 8 titik
pts_mid     = jumlah titik dengan y antara  0.25 dan 0.55 = 10 titik
pts_bot     = jumlah titik dengan y antara  0.55 dan 1.20 = 3 titik
centered    = jumlah titik dengan x antara  0.05 dan 0.95 = 20 titik (>5, lolos filter)

hand_top     = clamp(8 / 5, 0, 1)          = clamp(1.60, 0, 1) = 1.0
hand_mid_bot = clamp((10 + 3) / 5, 0, 1)   = clamp(2.60, 0, 1) = 1.0
```

Threshold `/5` artinya: **5 titik tangan di suatu zona sudah cukup untuk skor 1.0**. Skala linear dari 0 (tidak ada titik) ke 1.0 (≥5 titik).

Penyimpanan di LandmarkResult:
```python
# Penamaan "terbalik" — ini historis, konsisten di seluruh kode:
lr.hand_forehead = hand_mid_bot   # zona TENGAH/BAWAH → Frustration trigger
lr.hand_chin     = hand_top       # zona ATAS → Confusion trigger
```

Filter keamanan: jika `centered < 5` (terlalu sedikit titik yang ada di dalam frame crop), semua skor tangan di-reset ke 0 untuk menghindari false detection dari tangan yang lewat sekilas di pinggir frame.

---

## 6. Gaze Deviation — Metrik Dasar Boredom & Engagement

`gaze_dev` adalah satu angka yang mengukur "seberapa jauh pandangan siswa dari arah kamera". Nilainya dalam satuan derajat (°).

### Langkah Perhitungan Lengkap

**Input (contoh angka):**
```
yaw     = +12.30°   (kepala 12.3° ke kanan)
pitch   = -3.50°    (kepala sedikit nunduk)
iris_x  = +0.210    (pupil 21% ke kanan dari center mata)
iris_y  = -0.050    (pupil sedikit ke atas)

Parameter default:
  scale_h      = 35.0
  scale_v      = 25.0
  iris_side_mult = 2.0
  v_dead_zone  = 15.0
```

**Langkah 1: Gaze Horizontal**
```
gaze_h = yaw + iris_x × scale_h
       = 12.30 + 0.210 × 35.0
       = 12.30 + 7.35
       = +19.65°

Interpretasi: kepala menoleh 12.3° ke kanan, PLUS pupil juga geser
ke kanan 7.35° → total pandangan melenceng 19.65° ke kanan.
```

**Langkah 2: Gaze Vertikal**
```
gaze_v_raw = -pitch + iris_y × scale_v
           = -(-3.50) + (-0.050) × 25.0
           = 3.50 - 1.25
           = 2.25°

look_down_v = (eyeLookDownLeft + eyeLookDownRight) / 2 = 0.11

gaze_v = max(|gaze_v_raw|, look_down_v × 40)
       = max(|2.25|, 0.11 × 40)
       = max(2.25, 4.40)
       = 4.40°

gaze_v_eff = max(0, gaze_v - v_dead_zone)
           = max(0, 4.40 - 15.0)
           = 0.0°

→ gaze_v_eff = 0 karena gerakan vertikal 4.4° masih jauh dari
  dead zone 15° (kompensasi untuk layar yang ada di bawah level mata).
```

**Langkah 3: Tiga Floor untuk Horizontal**
```
iris_side = |iris_x| × scale_h × iris_side_mult
          = |0.210| × 35.0 × 2.0
          = 0.210 × 70.0
          = 14.70°

gaze_h_eff = max(|gaze_h|, iris_side, |yaw|)
           = max(19.65, 14.70, 12.30)
           = 19.65°

Tiga floor ini diperlukan karena:
  1. gaze_h = 19.65° → bisa jadi gabungan yaw + iris yang "wajar"
  2. iris_side = 14.70° → floor: pupil yang jelas ke samping tetap dihitung
     bahkan jika sebagian dikompensasi oleh yaw kepala
  3. |yaw| = 12.30° → floor minimum: kepala miring = pandangan melenceng
     minimal sebesar itu, tidak bisa nol
```

**Langkah 4: gaze_dev Final**
```
gaze_dev = sqrt(gaze_h_eff² + gaze_v_eff²)
         = sqrt(19.65² + 0.0²)
         = sqrt(386.12)
         = 19.65°

Karena gaze_v_eff = 0, deviasi murni dari komponen horizontal.
```

**Interpretasi gaze_dev:**

| gaze_dev | Situasi umum |
|---|---|
| 0–5° | Menatap lurus ke depan / kamera |
| 5–15° | Sedikit menyamping — masih bisa engaged |
| 15–25° | Jelas menoleh — potensi Boredom |
| >25° | Noleh jauh — Boredom sangat mungkin |

---

## 7. Scoring BOREDOM — Langkah demi Langkah

**Melanjutkan dari contoh di atas:** `gaze_dev = 19.65°`

### Langkah 1: bore_gaze dari gaze_dev

```python
gaze_dead_zone = 5.0
gaze_range     = 20.0

bore_gaze = clamp((gaze_dev - gaze_dead_zone) / gaze_range, 0, 1)
          = clamp((19.65 - 5.0) / 20.0, 0, 1)
          = clamp(14.65 / 20.0, 0, 1)
          = clamp(0.733, 0, 1)
          = 0.733

→ Gaze jauh (19.65°), bore_gaze 0.73
```

### Langkah 2: Sinyal Ekspresi Pendukung

```python
# eyeBlink (mata ngantuk)
blink_dead_zone = 0.20
blink_range     = 0.50

max_blink = max(eyeBlinkLeft, eyeBlinkRight) = max(0.08, 0.09) = 0.09

blink_v = clamp((0.09 - 0.20) / 0.50, 0, 1)
        = clamp(-0.11 / 0.50, 0, 1)
        = clamp(-0.22, 0, 1)
        = 0.0   ← mata tidak mengantuk (0.09 < dead zone 0.20)

# Menguap (jawOpen)
yawn_threshold = 0.35
jawOpen = 0.06

yawn_v = clamp(0.06 / 0.35, 0, 1) if pitch < 15 else 0.0
       = clamp(0.171, 0, 1)    # pitch = -3.5 < 15, aktif
       = 0.171   ← mulut sedikit terbuka, sedikit berkontribusi

# Pitch mendongak (pitch_up_v untuk nilai >20°)
pitch_up_v = clamp((pitch - 20) / 25, 0, 1)
           = clamp((-3.50 - 20) / 25, 0, 1)
           = clamp(-1.14, 0, 1)
           = 0.0   ← kepala tidak mendongak

sig_expr_weight = 0.70
sig_expr = max(blink_v, yawn_v, pitch_up_v) × sig_expr_weight
         = max(0.0, 0.171, 0.0) × 0.70
         = 0.171 × 0.70
         = 0.120
```

### Langkah 3: Blend Final

```python
blend_a = 0.85
blend_b = 0.15

base_bore = max(bore_gaze, sig_expr)
          = max(0.733, 0.120)
          = 0.733

bore = clamp(base_bore × blend_a + (bore_gaze + sig_expr) × blend_b, 0, 1)
     = clamp(0.733 × 0.85 + (0.733 + 0.120) × 0.15, 0, 1)
     = clamp(0.623 + 0.853 × 0.15, 0, 1)
     = clamp(0.623 + 0.128, 0, 1)
     = 0.751

→ BOREDOM = 0.751 (tinggi — kepala menoleh jauh)
```

---

## 8. Scoring ENGAGEMENT — Langkah demi Langkah

Engagement menggunakan `gaze_dev` yang sama, tapi sebagai **gate** (mematikan Engagement jika gaze terlalu besar).

**Input:** `gaze_dev = 19.65°`

```python
tegak_dead_zone = 5.0
tegak_range     = 12.0

gate = clamp(1 - max(0, gaze_dev - tegak_dead_zone) / tegak_range, 0, 1)
     = clamp(1 - max(0, 19.65 - 5.0) / 12.0, 0, 1)
     = clamp(1 - 14.65 / 12.0, 0, 1)
     = clamp(1 - 1.221, 0, 1)
     = clamp(-0.221, 0, 1)
     = 0.0

# → gate = 0.0 karena gaze_dev (19.65°) melampaui dead_zone + range = 17°

blink_heavy_th  = 0.50
blink_heavy_min = 0.30

max_blink = 0.09   (dari atas)
blink_heavy = max(0, max_blink - blink_heavy_th) / blink_heavy_th
            = max(0, 0.09 - 0.50) / 0.50
            = max(0, -0.41) / 0.50
            = 0.0

eng = gate × max(blink_heavy_min, 1.0 - blink_heavy)
    = 0.0 × max(0.30, 1.0 - 0.0)
    = 0.0 × 1.0
    = 0.0

→ ENGAGEMENT = 0.000 (nol karena gate = 0 — kepala menoleh terlalu jauh)
```

**Contoh siswa yang lurus:** `gaze_dev = 4°`
```
gate = clamp(1 - max(0, 4 - 5) / 12, 0, 1) = clamp(1 - 0, 0, 1) = 1.0
eng  = 1.0 × max(0.30, 1.0) = 1.0 × 1.0 = 1.0
→ ENGAGEMENT = 1.000 (penuh — gaze 4° masih dalam dead zone 5°)
```

---

## 9. Scoring CONFUSION — Langkah demi Langkah

**Input baru (contoh siswa bingung):**
```
iris_y       = -0.22   (pupil ke atas)
pitch        = +7.00°  (kepala sedikit mendongak — berpikir)
browDownLeft = 0.38, browDownRight = 0.35
browInnerUp  = 0.42
eyeLookUpLeft = 0.28, eyeLookUpRight = 0.31
jawOpen      = 0.18
mouthSmile   = 0.05, 0.06
mouthPucker  = 0.25
gaze_h_eff   = 5.0°   (kepala relatif lurus)
hand_chin    = 0.0
hand_forehead = 0.0
```

### Langkah 1: Sinyal Mata ke Atas

```python
iris_up_dead_zone = 0.15
iris_up_range     = 0.30

iris_up_v = clamp((-iris_y - 0.15) / 0.30, 0, 1)
          = clamp((-(-0.22) - 0.15) / 0.30, 0, 1)
          = clamp((0.22 - 0.15) / 0.30, 0, 1)
          = clamp(0.07 / 0.30, 0, 1)
          = clamp(0.233, 0, 1)
          = 0.233   ← iris sedikit ke atas, sudah melewati dead zone

look_up_threshold = 0.35
look_up_v = clamp(max(0.28, 0.31) / 0.35, 0, 1)
          = clamp(0.31 / 0.35, 0, 1)
          = clamp(0.886, 0, 1)
          = 0.886   ← eyeLookUp kuat!
```

### Langkah 2: Pitch kepala (mendongak saat berpikir)

```python
pitch_start = 5.0
pitch_range = 15.0

pitch_cu = clamp((7.00 - 5.0) / 15.0, 0, 1)
         = clamp(2.0 / 15.0, 0, 1)
         = clamp(0.133, 0, 1)
         = 0.133   ← kepala baru mulai mendongak

co_signal = max(iris_up_v, look_up_v, pitch_cu)
          = max(0.233, 0.886, 0.133)
          = 0.886   ← sinyal penguat untuk browInnerUp
```

### Langkah 3: Alis (browDown & browInnerUp)

```python
brow_dn_th = 0.23

brow_dn_v = clamp((mean(0.38, 0.35)) / 0.23, 0, 1)
          = clamp(0.365 / 0.23, 0, 1)
          = clamp(1.587, 0, 1)
          = 1.0   ← alis turun sangat kuat (melebihi threshold)

# browInnerUp butuh co_signal agar tidak selalu aktif
brow_in_th      = 0.30
brow_in_co_gate = 0.25

brow_in_raw = 0.42
brow_in_v   = clamp(0.42 / 0.30, 0, 1) × clamp(co_signal / 0.25, 0, 1)
            = clamp(1.40, 0, 1)         × clamp(0.886 / 0.25, 0, 1)
            = 1.0                        × clamp(3.544, 0, 1)
            = 1.0                        × 1.0
            = 1.0   ← browInnerUp kuat DAN ada co_signal kuat dari lookUp
```

### Langkah 4: Sinyal Mulut

```python
jawOpen = 0.18

# Bell curve: 0 di bawah jaw_start, naik ke 1 di jaw_peak, turun ke 0 di jaw_end
jaw_start = 0.05, jaw_peak = 0.25, jaw_end = 0.40

0.05 < 0.18 <= 0.25 → path naik:
jaw_val_conf = (0.18 - 0.05) / (0.25 - 0.05)
             = 0.13 / 0.20
             = 0.65

smile_raw = max(0.05, 0.06) = 0.06
smile_pen = max(0, 0.06 - smile_penalty_th)
          = max(0, 0.06 - 0.15)
          = 0.0   (senyum di bawah threshold penalti)

jaw_co = clamp(0.65 - 0.0 × 1.5, 0, 1) = 0.65

# mouthPucker (langsung, tanpa gate)
pucker_th = 0.30

sig_brow_conf = max(brow_dn_v, brow_in_v) = max(1.0, 1.0) = 1.0

pucker_co = clamp(0.25 / 0.30, 0, 1) = clamp(0.833, 0, 1) = 0.833
```

### Langkah 5: base_conf dan blend

```python
sig_mata_conf = max(iris_up_v, look_up_v) = max(0.233, 0.886) = 0.886

base_conf = max(sig_brow_conf, sig_mata_conf, jaw_co, pucker_co, hand_chin)
          = max(1.0, 0.886, 0.65, 0.833, 0.0)
          = 1.0

conf = clamp(base_conf × 0.85 + (pitch_cu + sig_brow_conf) × 0.15, 0, 1)
     = clamp(1.0 × 0.85 + (0.133 + 1.0) × 0.15, 0, 1)
     = clamp(0.85 + 1.133 × 0.15, 0, 1)
     = clamp(0.85 + 0.170, 0, 1)
     = clamp(1.020, 0, 1)
     = 1.0
```

### Langkah 6: Suppression Tangan

```python
# Suppression dari tangan (tangan di zona tengah wajah membatalkan Confusion)
suppression = hand_forehead if hand_chin < 0.5 else 0.0
            = 0.0   (tidak ada tangan)

conf = clamp(1.0 - 0.0, 0, 1) = 1.0

→ CONFUSION = 1.000 (sangat tinggi — banyak sinyal kuat)
```

---

## 10. Scoring FRUSTRATION — Langkah demi Langkah

**Input contoh (ekspresi frustrasi):**
```
browDownLeft = 0.52, browDownRight = 0.49
noseSneerLeft = 0.35, noseSneerRight = 0.28
cheekSquintLeft = 0.41, cheekSquintRight = 0.38
mouthPressLeft = 0.48, mouthPressRight = 0.45
eyeSquintLeft = 0.44, eyeSquintRight = 0.41
jawOpen      = 0.08
mouthSmile   = 0.02
hand_forehead = 0.0
```

### Langkah 1: Sinyal Individual

```python
brow_dn_th = 0.40
br_fr = clamp(mean(0.52, 0.49) / 0.40, 0, 1)
      = clamp(0.505 / 0.40, 0, 1)
      = clamp(1.2625, 0, 1)
      = 1.0

nose_sneer_th = 0.20
ns_fr = clamp(max(0.35, 0.28) / 0.20, 0, 1)
      = clamp(0.35 / 0.20, 0, 1)
      = clamp(1.75, 0, 1)
      = 1.0   ← noseSneer sangat kuat! (sinyal terkuat Frustration)

cheek_squint_th = 0.40
ck_fr = clamp(mean(0.41, 0.38) / 0.40, 0, 1)
      = clamp(0.395 / 0.40, 0, 1)
      = clamp(0.9875, 0, 1)
      = 0.988

mouth_press_th = 0.40
lp_fr = clamp(mean(0.48, 0.45) / 0.40, 0, 1)
      = clamp(0.465 / 0.40, 0, 1)
      = clamp(1.1625, 0, 1)
      = 1.0

eye_squint_th = 0.40
ey_fr = clamp(mean(0.44, 0.41) / 0.40, 0, 1)
      = clamp(0.425 / 0.40, 0, 1)
      = clamp(1.0625, 0, 1)
      = 1.0
```

### Langkah 2: Jaw (rahang)

```python
jaw_start = 0.10, jaw_range = 0.20

jawOpen = 0.08 < jaw_start = 0.10
→ jaw_val_frus = max(0, 0.08 - 0.10) = 0.0
jw_fr = 0.0
```

### Langkah 3: SUM Logic & Smile Penalty

```python
smile_raw = max(0.02, ...) = 0.02
smile_pen = max(0, 0.02 - 0.15) = 0.0  (senyum kecil, tidak ada penalti)

# SUM: noseSneer (1.0) + rata-rata empat sinyal lain
sig_wajah_frus = clamp(ns_fr + (br_fr + lp_fr + ey_fr + ck_fr) / 2.0, 0, 1)
               = clamp(1.0 + (1.0 + 1.0 + 1.0 + 0.988) / 2.0, 0, 1)
               = clamp(1.0 + 3.988 / 2.0, 0, 1)
               = clamp(1.0 + 1.994, 0, 1)
               = clamp(2.994, 0, 1)
               = 1.0

sig_wajah_frus = clamp(1.0 - 0.0 × 1.5, 0, 1) = 1.0
```

### Langkah 4: base_frus & blend

```python
blend_a = 0.85, blend_b = 0.15

base_frus = clamp(sig_wajah_frus + hand_forehead, 0, 1)
          = clamp(1.0 + 0.0, 0, 1)
          = 1.0

frus = clamp(base_frus × 0.85 + (ck_fr + jw_fr) × 0.15, 0, 1)
     = clamp(1.0 × 0.85 + (0.988 + 0.0) × 0.15, 0, 1)
     = clamp(0.85 + 0.1482, 0, 1)
     = clamp(0.9982, 0, 1)
     = 0.998

→ FRUSTRATION = 0.998 (hampir maksimum)
```

---

## 11. SigLIP Scoring — Dari Teks + Gambar ke Angka

SigLIP bekerja dengan membandingkan **representasi gambar** dan **representasi teks** di ruang vektor yang sama.

### Langkah 1: Persiapan Input

```
Prompt Boredom (6 kalimat):
  "a face of a student with heavy droopy eyelids..."
  "a face of a student yawning widely..."
  "a face of a student with a completely blank..."
  ...

6 frame gambar crop wajah 512×512, di-resize ke 224×224 oleh processor

→ Processor menghasilkan:
  pixel_values: tensor [6, 3, 224, 224]    (6 frame, 3 channel RGB, 224×224)
  input_ids:    tensor [24, max_len]        (24 prompt total: 4 label × 6 prompt)
```

### Langkah 2: Forward Pass Model

```python
logits_per_image = model(**inputs).logits_per_image
# Shape: [6, 24]  →  6 frame, 24 teks
# Setiap sel = cosine similarity antara gambar[i] dan teks[j]
```

Contoh logits mentah untuk frame 0 (siswa mengantuk):
```
                      Boredom prompts              Engagement prompts    ...
logits[frame_0] = [-2.15, -1.87, -2.33, -1.96, -2.08, -1.75,   -4.21, -3.87, ...]
                   ↑ prompt "droopy eyelids" vs gambar frame_0

Nilai negatif karena logit zero-shot SigLIP selalu negatif untuk
deskripsi yang tidak terlatih secara supervised.
```

### Langkah 3: Sigmoid + Empirical Bias

```python
EMPIRICAL_BIAS = 3.5

# Untuk Boredom (label 0), ambil 6 kolom Boredom:
group_logits = logits[frame_0, [0,1,2,3,4,5]]
             = [-2.15, -1.87, -2.33, -1.96, -2.08, -1.75]

# Tambah bias:
biased = group_logits + 3.5
       = [1.35, 1.63, 1.17, 1.54, 1.42, 1.75]

# Sigmoid:
probs = sigmoid(biased)
      = [sigmoid(1.35), sigmoid(1.63), ...]
      = [0.794,         0.836,         0.763,  0.824,  0.805,  0.852]

# Rata-rata 6 prompt:
siglip_score_boredom_frame0 = mean([0.794, 0.836, 0.763, 0.824, 0.805, 0.852])
                             = 0.812
```

**Fungsi sigmoid:** `sigmoid(x) = 1 / (1 + e^-x)`

| biased logit | sigmoid(x) |
|---|---|
| −3.0 | 0.047 |
| −1.0 | 0.269 |
|  0.0 | 0.500 |
| +1.0 | 0.731 |
| +1.35 | 0.794 |
| +2.0 | 0.881 |
| +3.0 | 0.953 |

**Kenapa bias +3.5?** Logit SigLIP zero-shot biasanya sekitar −2 sampai −5. Dengan bias +3.5, nilai itu bergeser ke sekitar −1.5 sampai +1.5, yang memberi range sigmoid 0.18–0.82 — range yang bermakna untuk scoring.

Tanpa bias: `sigmoid(-3.5) = 0.030` → semua skor mendekati 0, tidak bisa dibedakan.
Dengan bias: `sigmoid(-3.5 + 3.5) = sigmoid(0) = 0.500` → titik tengah.

### Langkah 4: Per Frame, Per Label

Proses di atas diulang untuk semua 4 label × 6 frame:

```
               Boredom  Engagement  Confusion  Frustration
Frame 0:       0.812     0.342       0.415      0.281
Frame 1:       0.788     0.371       0.398      0.295
Frame 2:       0.824     0.318       0.432      0.267
Frame 3:       0.841     0.295       0.441      0.253
Frame 4:       0.799     0.362       0.421      0.278
Frame 5:       0.776     0.389       0.408      0.312
```

---

## 12. Hybrid Combine — Gabungkan SigLIP dan Landmark

### Bobot dan Normalisasi

```python
# Dari rules (contoh Boredom):
siglip_w_raw = 0.50
land_w_raw   = 0.50
total        = 0.50 + 0.50 = 1.0
siglip_w     = 0.50 / 1.0 = 0.500
land_w       = 0.50 / 1.0 = 0.500
```

Bobot tidak harus berjumlah 1 — sistem normalisasi otomatis:
```python
# Contoh ekstrim: siglip_w=3, land_w=1
total    = 4.0
siglip_w = 3/4 = 0.75
land_w   = 1/4 = 0.25
```

### Perhitungan Hybrid per Frame

```python
# Frame 0, label Boredom:
siglip_score_f0 = 0.812
landmark_score_f0 = 0.989  # dari compute_emotion_scores (contoh di atas)

hybrid_f0 = siglip_w × siglip_score + land_w × landmark_score
          = 0.500 × 0.812 + 0.500 × 0.989
          = 0.406 + 0.4945
          = 0.9005
          → dibulatkan ke 4 desimal: 0.9005
```

Untuk semua 6 frame:
```
Frame 0: 0.500 × 0.812 + 0.500 × 0.989 = 0.9005
Frame 1: 0.500 × 0.788 + 0.500 × 0.971 = 0.8795
Frame 2: 0.500 × 0.824 + 0.500 × 0.981 = 0.9025
Frame 3: 0.500 × 0.841 + 0.500 × 0.988 = 0.9145
Frame 4: 0.500 × 0.799 + 0.500 × 0.975 = 0.8870
Frame 5: 0.500 × 0.776 + 0.500 × 0.962 = 0.8690

avg_score = mean(0.9005, 0.8795, 0.9025, 0.9145, 0.8870, 0.8690)
          = 5.3530 / 6
          = 0.8922
```

---

## 13. Temporal Restlessness Bonus (Boredom)

Bonus ini mendeteksi **kepala yang bergerak bolak-balik antar frame** — sinyal bosan yang tidak bisa dilihat dari satu frame saja.

### Perhitungan

```python
# Kumpulkan nilai yaw dari semua frame yang face_found = True:
yaws = [+12.3, +5.8, −2.1, +8.4, −6.2, +10.1]   # dari 6 frame

import numpy as np
yaw_std = np.std(yaws)
        = std([12.3, 5.8, -2.1, 8.4, -6.2, 10.1])
        = 6.41°

# Parameter bonus:
std_min   = 3.0°    # std_dev harus melebihi ini agar bonus mulai
std_range = 7.0°    # range std untuk mencapai bonus maksimum
bonus_max = 0.15    # bonus paling besar

bonus = min(max((yaw_std - std_min) / std_range, 0.0), 1.0) × bonus_max
      = min(max((6.41 - 3.0) / 7.0, 0.0), 1.0) × 0.15
      = min(max(0.487, 0.0), 1.0) × 0.15
      = 0.487 × 0.15
      = 0.073

# Terapkan ke semua hybrid scores Boredom:
hybrid_scores_after_bonus = [min(s + 0.073, 1.0) for s in hybrid_scores]
                          = [0.9005+0.073, 0.8795+0.073, ...]
                          = [0.9735, 0.9525, 0.9755, 0.9875, 0.9600, 0.9420]
```

```
Debug log: [RESTLESS] yaw_std=6.4° → bonus=0.073
```

### Kapan Bonus Aktif?

| yaw_std | bonus |
|---|---|
| < 3° | 0.000 (kepala diam, tidak ada bonus) |
| 3° | 0.000 (batas bawah) |
| 6.5° | 0.075 |
| 10° | 0.150 (bonus maksimum) |
| > 10° | 0.150 (sudah maksimum) |

---

## 14. Threshold → Voting → Prediksi Akhir

### Threshold per Frame

```python
threshold_boredom = 0.50   # dari slider di panel kanan

frame_preds = [1 if s >= 0.50 else 0 for s in hybrid_scores_final]
            = [1 if 0.9735 >= 0.50,   # frame 0 → 1
               1 if 0.9525 >= 0.50,   # frame 1 → 1
               1 if 0.9755 >= 0.50,   # frame 2 → 1
               1 if 0.9875 >= 0.50,   # frame 3 → 1
               1 if 0.9600 >= 0.50,   # frame 4 → 1
               1 if 0.9420 >= 0.50]   # frame 5 → 1
            = [1, 1, 1, 1, 1, 1]
```

### Voting

```python
# Misalkan frame 2 di-reject (double-klik oleh labeler):
rejected_set = {2}

valid_preds = [p for i, p in enumerate(frame_preds) if i not in rejected_set]
            = [1, 1, 1, 1, 1]   # 5 frame valid (frame 2 di-skip)

n_valid  = 5
vote_pos = sum(valid_preds) = 5
vote_neg = n_valid - vote_pos = 0

# Mayoritas: ≥ setengah frame valid harus positif
prediction = 1 if (n_valid > 0 and vote_pos >= max(1, (n_valid+1)//2)) else 0
           = 1 if (5 > 0 and 5 >= max(1, 3)) else 0
           = 1 if (True and 5 >= 3) else 0
           = 1   ← BOREDOM = 1 (positif)
```

### Ringkasan Output batch_history

```json
{
  "per_label": {
    "0": {
      "prediction":   1,
      "vote_pos":     5,
      "vote_neg":     0,
      "skipped":      1,
      "avg_score":    0.9682,
      "siglip_avg":   0.8065,
      "landmark_avg": 0.9777,
      "threshold":    0.50,
      "frame_scores": [0.9735, 0.9525, 0.9755, 0.9875, 0.9600, 0.9420],
      "frame_preds":  [1, 1, 1, 1, 1, 1]
    }
  }
}
```

---

## 15. Membaca Debug Log

Saat model berjalan, kode mencetak log per frame ke terminal:

```
  [LAND] yaw=+12.3 pitch=-3.5 iris_x=+0.210 iris_y=-0.050 lookDn=0.11 | gH=+19.7° gV=0.0° dev=19.7° | boreGaze=0.97 gate=0.00 | B=0.989 E=0.000 C=0.095 F=0.042
```

| Bagian | Nilai | Arti |
|---|---|---|
| `yaw=+12.3` | +12.3° | Kepala menoleh 12.3° ke kanan |
| `pitch=-3.5` | −3.5° | Kepala sedikit nunduk |
| `iris_x=+0.210` | +0.21 | Pupil 21% ke kanan dari center mata |
| `iris_y=-0.050` | −0.05 | Pupil 5% ke atas |
| `lookDn=0.11` | 0.11 | eyeLookDown rata-rata |
| `gH=+19.7°` | +19.7° | Komponen horizontal gaze (`gaze_h`) |
| `gV=0.0°` | 0.0° | Komponen vertikal efektif (`gaze_v_eff`, sudah dikurangi dead zone) |
| `dev=19.7°` | 19.7° | Total `gaze_dev` = sqrt(19.7²+0²) |
| `boreGaze=0.97` | 0.97 | Komponen gaze untuk Boredom |
| `gate=0.00` | 0.00 | Gate Engagement (0 = Engagement nol karena gaze terlalu jauh) |
| `B=0.989` | 0.989 | Skor Boredom akhir |
| `E=0.000` | 0.000 | Skor Engagement akhir |
| `C=0.095` | 0.095 | Skor Confusion akhir |
| `F=0.042` | 0.042 | Skor Frustration akhir |

Jika Confusion > 0.5, baris tambahan muncul:
```
  [CONF] brow_dn=1.00 brow_in=1.00 iris_up=0.23 look_up=0.89 jaw=0.65 pucker=0.83 pitch_cu=0.13 base=1.00
```

| Nilai | Sinyal |
|---|---|
| `brow_dn=1.00` | alis turun penuh (melebihi threshold 0.23) |
| `brow_in=1.00` | alis dalam naik penuh dengan co_signal kuat |
| `iris_up=0.23` | pupil sedikit ke atas |
| `look_up=0.89` | eyeLookUp kuat |
| `jaw=0.65` | mangap di 65% dari puncak bell curve |
| `pucker=0.83` | bibir mengerucut |
| `pitch_cu=0.13` | kepala baru mulai mendongak |
| `base=1.00` | base_conf = 1.0 sebelum blend |

```
  [HAND-FULL] Terdeteksi 1 tangan, in_crop=14, pts_top=6, pts_mid=5, pts_bot=3
```

| Nilai | Arti |
|---|---|
| `Terdeteksi 1 tangan` | 1 tangan ditemukan di full frame |
| `in_crop=14` | 14 titik dari 21 ada di dalam area crop wajah |
| `pts_top=6` | 6 titik di zona atas (y < 0.25) → Confusion |
| `pts_mid=5` | 5 titik di zona tengah |
| `pts_bot=3` | 3 titik di zona bawah |

```
  [RESTLESS] yaw_std=6.4° → bonus=0.073
```
Muncul hanya jika temporal restlessness bonus aktif (yaw_std ≥ 3°).

---

## 16. Contoh Lengkap Satu Frame

Merangkum semua tahap untuk **satu frame siswa yang noleh sambil sedikit bingung**:

```
Input: frame BGR 512×512

MediaPipe output:
  yaw = +18°, pitch = +6°, iris_x = +0.15, iris_y = -0.12
  browDownLeft=0.28, browDownRight=0.26, browInnerUp=0.35
  eyeLookUpLeft=0.22, eyeLookUpRight=0.25
  jawOpen=0.10, mouthSmile=0.04

GAZE CALCULATION:
  gaze_h = 18 + 0.15×35 = 18 + 5.25 = 23.25°
  gaze_v_raw = -6 + (-0.12)×25 = -6 - 3.0 = -9.0 → |−9.0| = 9.0°
  gaze_v_eff = max(0, 9.0 - 15) = 0°
  iris_side = 0.15×35×2.0 = 10.5°
  gaze_h_eff = max(23.25, 10.5, 18) = 23.25°
  gaze_dev = sqrt(23.25²) = 23.25°

BOREDOM:
  bore_gaze = clamp((23.25-5)/20, 0,1) = clamp(0.913, 0,1) = 0.913
  blink_v = 0, yawn_v = clamp(0.10/0.35,0,1)=0.286
  sig_expr = 0.286×0.70 = 0.200
  bore = clamp(0.913×0.85 + (0.913+0.200)×0.15, 0,1) = clamp(0.776+0.167, 0,1) = 0.943

ENGAGEMENT:
  gate = clamp(1-(23.25-5)/12, 0,1) = clamp(1-1.52, 0,1) = 0.0
  eng = 0.0 × 1.0 = 0.0

CONFUSION:
  iris_up_v = clamp((0.12-0.15)/0.30, 0,1) = clamp(-0.1, 0,1) = 0.0
    (iris_y=-0.12, -iris_y=+0.12, 0.12 < dead zone 0.15 → tidak aktif)
  look_up_v = clamp(max(0.22,0.25)/0.35, 0,1) = clamp(0.714, 0,1) = 0.714
  pitch_cu  = clamp((6-5)/15, 0,1) = clamp(0.067, 0,1) = 0.067
  co_signal = max(0.0, 0.714, 0.067) = 0.714
  brow_dn_v = clamp(mean(0.28,0.26)/0.23, 0,1) = clamp(1.174, 0,1) = 1.0
  brow_in_v = clamp(0.35/0.30,0,1) × clamp(0.714/0.25,0,1) = 1.0 × 1.0 = 1.0
  jaw_co: 0.05 < 0.10 ≤ 0.25 → (0.10-0.05)/(0.25-0.05) = 0.25
  pucker_co = clamp(0/0.30, 0,1) = 0.0  (mouthPucker tidak ada di input)
  base_conf = max(1.0, 0.714, 0.25, 0.0, 0.0) = 1.0
  conf = clamp(1.0×0.85 + (0.067+1.0)×0.15, 0,1) = clamp(1.010, 0,1) = 1.0

SigLIP: (asumsi)
  siglip_boredom_frame = 0.72

HYBRID (α=0.5, β=0.5):
  hybrid = 0.5×0.72 + 0.5×1.0 = 0.36 + 0.50 = 0.86

PREDIKSI (threshold=0.50):
  frame_pred_boredom = 1  (0.86 ≥ 0.50)
  frame_pred_engagement = 0
  frame_pred_confusion = 0
  frame_pred_frustration = 0

DEBUG LOG yang akan tercetak:
  [LAND] yaw=+18.0 pitch=+6.0 iris_x=+0.150 iris_y=-0.120 lookDn=0.00 | gH=+23.2° gV=0.0° dev=23.2° | boreGaze=1.00 gate=0.00 | B=1.000 E=0.000 C=0.000 F=...
```

---

## Ringkasan Formula

```
gaze_dev = sqrt(max(|yaw + iris_x×35|, |iris_x|×70, |yaw|)² + max(0, |gaze_v| - 15)²)

BOREDOM   = blend(max(bore_gaze, sig_expr),  bore_gaze,  sig_expr)
ENGAGEMENT = gate(gaze_dev) × max(0.30, 1-blink_heavy)
CONFUSION  = blend(max_signals, pitch_cu, sig_brow) × (1-hand_suppression)
FRUSTRATION = blend(SUM(ns_fr, br_fr, lp_fr, ey_fr, ck_fr) + hand, ck_fr, jw_fr)

SIGLIP[label][frame] = mean(sigmoid(logits[frame, label_prompts] + 3.5))

HYBRID[frame] = (α × SIGLIP[frame] + β × LANDMARK[frame]) / (α + β)

PREDICTION = 1 jika count(HYBRID[f] ≥ threshold) ≥ n_valid/2
```

---

## Referensi & Sumber

### Model Utama

**SigLIP — Sigmoid Loss for Language Image Pre-Training**
- Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023.
- arXiv: [2303.15343](https://arxiv.org/abs/2303.15343)
- Basis matematis: mengapa dipakai sigmoid (independen per pasang gambar-teks) bukan softmax (bergantung pada seluruh batch)

**SigLIP 2**
- Tschannen et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.*
- arXiv: [2502.08769](https://arxiv.org/abs/2502.08769)
- Model yang dipakai: `google/siglip2-base-patch16-224`

**MediaPipe FaceLandmarker**
- Google LLC. *MediaPipe Face Landmarker.* [developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- Menghasilkan: 478 landmark 3D, 4×4 transformation matrix, 52 blendshape coefficient

**MediaPipe HandLandmarker**
- Google LLC. *MediaPipe Hand Landmarker.* [developers.google.com/mediapipe/solutions/vision/hand_landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- Menghasilkan: 21 landmark 3D per tangan

### Blendshapes — Standar Apple ARKit

52 blendshape yang dihasilkan MediaPipe mengikuti konvensi **Apple ARKit Face Tracking blendshapes**. Setiap nama seperti `browDownLeft`, `jawOpen`, `noseSneerRight` adalah standar yang sama yang dipakai di ARKit dan berbagai 3D avatar pipeline.

- Apple Inc. *ARKit — Tracking and Visualizing Faces.* [developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation)
- Daftar lengkap 52 blendshape dengan deskripsi fisik masing-masing

### FACS — Facial Action Coding System

Blendshape dirancang untuk merepresentasikan **Action Units (AU)** dari FACS — sistem kodifikasi gerakan otot wajah yang dikembangkan oleh Paul Ekman.

- Ekman, P. & Friesen, W.V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement.* Consulting Psychologists Press.
- Ekspresi seperti `browDownLeft` ≈ AU4 (Brow Lowerer), `noseSneerLeft` ≈ AU9 (Nose Wrinkler)

### Head Pose Estimation — Rotation Matrix ke Euler Angles

Transformasi 4×4 matrix → yaw/pitch/roll menggunakan dekomposisi Euler ZYX standar:

- Diebel, J. (2006). *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors.* Stanford University Technical Report. [diebel.com](https://www.diebel.com/attitude/Diebel2006.pdf)
- Rumus yang dipakai: `pitch = atan2(-R[2,0], sy)`, `yaw = atan2(R[1,0], R[0,0])`

Untuk head pose estimation spesifik dari facial landmarks:
- Kazemi, V. & Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* CVPR 2014.

### Iris Tracking & Gaze Estimation

Metode offset iris relatif ke sudut mata (Eye Corner-based):

- Wood, E. et al. (2022). *3D-Aware Facial Landmark Detection via Multi-Task Learning.* Dalam MediaPipe iris tracking pipeline.
- George, A. & Marcel, S. (2019). *OpenGaze: A Toolkit for Open World Gaze Estimation.* — konsep normalisasi iris_x/iris_y relatif ke ukuran mata

### Blink Detection — Eye Aspect Ratio (EAR)

Konsep `eyeBlinkLeft/Right` dari blendshapes serupa dengan Eye Aspect Ratio:

- Soukupova, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* Computer Vision Winter Workshop (CVWW).
- EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|) — rasio keterbukaan kelopak mata

### Emosi & Affect Computing

Konteks teoritis untuk keempat label yang dipakai:

- D'Mello, S.K. & Graesser, A. (2012). *Dynamics of Affective States during Complex Learning.* Learning and Instruction. — boredom, engagement, confusion, frustration dalam konteks belajar
- Pekrun, R. et al. (2002). *Academic Emotions in Students' Self-Regulated Learning and Achievement.* Educational Psychologist. — kerangka teori emosi akademik

### Gaze Deviation & Engagement Detection

- Jaques, N. et al. (2016). *Predicting Students' Disengagement in MOOCs.* EDM 2016. — head pose sebagai proxy engagement
- Vail, A.K. et al. (2015). *Predicting Collaborative Learning Outcomes from Head Pose.* — hubungan yaw kepala dan arah perhatian

---

## Dokumen Terkait

- [README.md](../README.md) — Panduan instalasi, penggunaan UI, dan FAQ
- [RULES_PANEL.md](RULES_PANEL.md) — Cara mengubah parameter scoring dan strategi tuning
- [core/README_SIGLIP.md](../core/README_SIGLIP.md) — Arsitektur pipeline dan sistem cache
