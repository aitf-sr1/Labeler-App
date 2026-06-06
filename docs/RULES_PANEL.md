# Panduan Rules Editor

> **Navigasi:** [README utama](../README.md) · [Perhitungan Teknis](COMPUTATION.md) · [Arsitektur Pipeline](../core/README_SIGLIP.md)

Rules Editor adalah jendela konfigurasi yang memungkinkan kamu mengubah semua parameter scoring landmark dan bobot hybrid tanpa edit kode. Dibuka via tombol **Rules** di topbar aplikasi.

> Untuk memahami **makna matematis** setiap parameter (rumus lengkap, contoh angka, efek perubahan), lihat [docs/COMPUTATION.md](COMPUTATION.md). Dokumen ini fokus pada **cara pakai** dan **strategi tuning**.

---

## Daftar Isi

1. [Cara Membuka & Menggunakan](#1-cara-membuka--menggunakan)
2. [Panduan Cepat — "Saya mau atur label X"](#2-panduan-cepat--saya-mau-atur-label-x)
3. [Referensi Parameter Lengkap](#3-referensi-parameter-lengkap)
   - [Gaze (shared)](#gaze-shared)
   - [Boredom](#boredom)
   - [Engagement](#engagement)
   - [Confusion](#confusion)
   - [Frustration](#frustration)
   - [Hybrid Weights & SigLIP](#hybrid-weights--siglip)
4. [Alur Kerja Recalculate](#4-alur-kerja-recalculate)
5. [Tips & Strategi Tuning](#5-tips--strategi-tuning)

---

## 1. Cara Membuka & Menggunakan

**Buka:** Klik tombol **Rules** di topbar (ungu). Jendela terpisah muncul — bisa dibuka bersamaan dengan aplikasi utama.

**Menutup dan membuka lagi:** Jendela tidak hilang dari memori — klik Rules lagi untuk memunculkan jendela yang sama ke depan.

### Cara Mengubah Nilai

Setiap parameter memiliki dua kontrol:

| Kontrol | Cara pakai |
|---|---|
| **Slider** (bar horizontal) | Drag untuk perubahan kasar dan cepat |
| **Input field** (kotak angka, kanan) | Ketik angka → tekan **Enter** atau klik area lain → nilai otomatis di-clamp ke range valid dan slider ikut bergerak |

Untuk nilai presisi (misal beda 0.01 atau 0.001), gunakan input field — slider terlalu kasar untuk itu.

### Tombol di Bagian Bawah

| Tombol | Fungsi |
|---|---|
| **Reset Default** | Kembalikan semua parameter ke nilai bawaan (DEFAULT_RULES) |
| **Simpan** | Simpan ke `rules.json` di folder output — berlaku untuk inferensi berikutnya |
| **Recalculate Batch** | Hitung ulang seluruh batch dari cache dengan parameter & threshold saat ini |

> Klik **Simpan** dulu sebelum Recalculate — rules yang tampil di UI langsung dipakai tanpa harus disimpan terlebih dahulu, tapi menyimpan memastikan nilai terpersist jika aplikasi ditutup.

---

## 2. Panduan Cepat — "Saya mau atur label X"

### Skor Boredom terlalu tinggi (terlalu banyak false positive)

Kemungkinan penyebab: siswa sering lirik samping sebentar tapi belum tentu bosan.

**Yang diubah:**
1. Naikkan `Gaze Dead Zone` di seksi Boredom (misal 8 → 12) — gaze kecil tidak dianggap bosan
2. Naikkan `Gaze Range` (misal 12 → 18) — butuh gaze lebih ekstrem untuk mencapai skor penuh
3. Turunkan `Expr Weight` (misal 0.70 → 0.50) — kurangi kontribusi sinyal menguap/mata berat
4. Naikkan `Yawn Threshold` (misal 0.35 → 0.50) — butuh mulut lebih terbuka untuk dianggap menguap
5. Di **Hybrid Weights**: naikkan `Landmark` weight relatif terhadap `SigLIP` — landmark lebih ketat dari SigLIP untuk Boredom

---

### Skor Boredom terlalu rendah (miss deteksi)

Kemungkinan penyebab: siswa benar-benar noleh/mengantuk tapi skor rendah.

**Yang diubah:**
1. Turunkan `Gaze Dead Zone` di seksi Boredom (misal 8 → 5)
2. Turunkan `Gaze Range` (misal 12 → 8) — gaze lebih kecil sudah cukup untuk skor tinggi
3. Naikkan `Expr Weight` (misal 0.70 → 0.85) — menguap/mata berat lebih berpengaruh
4. Turunkan `Yawn Threshold` (misal 0.35 → 0.20) — mulut sedikit terbuka sudah dianggap menguap
5. Turunkan threshold slider Boredom di panel kanan (misal 0.50 → 0.40)

---

### Skor Engagement tidak masuk akal (terlalu tinggi/rendah)

Engagement dan Boredom berbagi `gaze_dev`. Mengatur parameter Boredom juga mengubah Engagement secara implisit.

**Parameter khusus Engagement:**
1. `Tegak Dead Zone` — gaze_dev < nilai ini = Engagement penuh (siswa sangat lurus ke depan)
2. `Tegak Range` — range dimana Engagement turun dari 1.0 ke 0.0 saat gaze membesar
3. `Min Engagement` — nilai minimum Engagement bahkan saat mata sangat ngantuk (blink berat)
4. `Heavy Blink Th` — threshold mata ngantuk parah (eyeBlink > nilai ini = Engagement berkurang)

Jika siswa mata sipit natural sering dapat Engagement rendah: naikkan `Heavy Blink Th` (misal 0.50 → 0.70).

---

### Skor Confusion terlalu rendah (miss deteksi)

Confusion paling sering miss karena threshold terlalu ketat.

**Yang diubah — coba satu per satu:**
1. **Turunkan `BrowDown Th`** (misal 0.23 → 0.15) — alis tidak harus turun terlalu banyak
2. **Turunkan `BrowInnerUp Th`** (misal 0.30 → 0.20) — alis dalam tidak harus naik terlalu tinggi
3. **Turunkan `BrowInner CoGate`** (misal 0.25 → 0.15) — co_signal tidak harus sekuat itu untuk mengaktifkan browInnerUp
4. **Turunkan `LookUp Threshold`** (misal 0.35 → 0.20) — lirik atas kecil sudah dihitung
5. **Turunkan `Iris Up Dead Zone`** (misal 0.15 → 0.08) — iris sedikit ke atas sudah dihitung
6. Di **Hybrid Weights**: naikkan `SigLIP` weight untuk Confusion (misal 0.50 → 0.65) — SigLIP lebih baik menangkap ekspresi berpikir halus
7. Turunkan threshold slider Confusion di panel kanan (misal 0.50 → 0.38)

---

### Skor Confusion terlalu tinggi (banyak false positive)

Kemungkinan penyebab: siswa dengan bentuk alis natural yang berkerut/naik.

**Yang diubah:**
1. **Naikkan `BrowInner CoGate`** (misal 0.25 → 0.40) — butuh sinyal iris/kepala yang lebih kuat untuk mengaktifkan browInnerUp
2. **Naikkan `BrowDown Th`** (misal 0.23 → 0.35) — alis harus lebih turun secara signifikan
3. **Naikkan `Smile Penalty Th`** (misal 0.15 → 0.25) — senyum sudah cukup untuk mengurangi skor confusion
4. **Naikkan `Jaw End`** (misal 0.40 → 0.55) — mulut terbuka lebih lebar tidak dianggap confusion (lebih mungkin menguap)

---

### Skor Frustration terlalu rendah

Frustration membutuhkan **kombinasi sinyal** (SUM logic). Jika miss, biasanya satu sinyal mendominasi tapi tidak cukup.

**Yang diubah:**
1. **Turunkan `NoseSneer Th`** (misal 0.20 → 0.10) — mengernyit hidung kecil sudah kuat
2. **Turunkan `BrowDown Th`** Frustration (misal 0.40 → 0.25)
3. **Turunkan `MouthPress Th`** (misal 0.40 → 0.25) — bibir sedikit ditekan sudah dihitung
4. **Turunkan `EyeSquint Th`** (misal 0.40 → 0.25)
5. Turunkan threshold slider Frustration di panel kanan

---

### Skor Frustration terlalu tinggi (banyak false positive dengan Confusion)

**Yang diubah:**
1. **Naikkan `BrowDown Th`** Frustration (misal 0.40 → 0.55)
2. **Naikkan `NoseSneer Th`** (misal 0.20 → 0.30) — harus benar-benar mengernyit
3. **Naikkan `CheekSquint Th`** dan **`EyeSquint Th`**
4. Naikkan threshold slider Frustration di panel kanan

---

### SigLIP memberi skor aneh (semua tinggi atau semua rendah)

**Yang diubah:**
1. **SigLIP Empirical Bias** — nilai default 3.5. Naikkan jika semua skor terlalu rendah (misal ke 4.5). Turunkan jika terlalu tinggi (misal ke 2.5). Bias menggeser kurva sigmoid dari logit negatif ke area yang bermakna.
2. Edit **prompt** langsung di panel kanan UI — prompt yang lebih spesifik biasanya lebih baik dari yang generik

---

## 3. Referensi Parameter Lengkap

### Gaze (shared)

Parameter gaze digunakan bersama oleh Boredom dan Engagement.

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **H Scale** (`scale_h`) | 35.0 | 10–60 | Konversi offset iris horizontal ke derajat. `iris_x=0.2` → `0.2×35=7°`. Naikkan jika siswa dengan mata sipit under-detected. |
| **V Scale** (`scale_v`) | 25.0 | 10–50 | Konversi offset iris vertikal ke derajat. Dipakai untuk gaze_v dan iris_up (Confusion). |
| **Iris Side Mult** (`iris_side_mult`) | 2.0 | 0.5–4.0 | Pengali floor lateral iris. Mencegah kompensasi penuh antara yaw kepala dan iris. Naikkan untuk memperkuat "kepala miring + mata lihat balik ke kamera tetap dianggap miring". |
| **V Dead Zone** (`v_dead_zone`) | 15.0 | 0–30 | Gaze vertikal < nilai ini diabaikan (kompensasi untuk siswa yang layarnya di bawah level mata). Turunkan jika banyak siswa yang lihat ke bawah dan tidak terdeteksi. |

---

### Boredom

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **Gaze Dead Zone** (`gaze_dead_zone`) | 5.0° | 0–20 | gaze_dev < nilai ini = skor gaze Boredom = 0. Gaze kecil tidak dianggap bosan. |
| **Gaze Range** (`gaze_range`) | 20.0° | 4–40 | Range gaze di atas dead zone untuk mencapai skor 1.0. `gaze_dev = dead_zone + range` → score = 1. |
| **Blink Dead Zone** (`blink_dead_zone`) | 0.20 | 0–0.5 | eyeBlink < nilai ini diabaikan (mengedip normal tidak dihitung). |
| **Blink Range** (`blink_range`) | 0.50 | 0.1–1.0 | Range blink di atas dead zone untuk skor 1.0. |
| **Yawn Threshold** (`yawn_threshold`) | 0.35 | 0.1–0.8 | `jawOpen / threshold = yawn_v`. Semakin kecil = mulut sedikit terbuka sudah dianggap menguap. |
| **Expr Weight** (`sig_expr_weight`) | 0.70 | 0–1.0 | Bobot max(blink, yawn, pitch_up) dalam skor akhir. Turunkan agar Boredom lebih bergantung pada gaze saja. |

---

### Engagement

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **Tegak Dead Zone** (`tegak_dead_zone`) | 5.0° | 0–10 | gaze_dev < nilai ini = gate = 1.0 (Engagement penuh). |
| **Tegak Range** (`tegak_range`) | 12.0° | 5–25 | Gate turun dari 1 ke 0 dalam range ini. Naikkan untuk lebih toleran terhadap gaze sedikit menyamping. |
| **Heavy Blink Th** (`blink_heavy_th`) | 0.50 | 0.2–0.9 | eyeBlink > nilai ini = dianggap droopy parah, Engagement berkurang. Naikkan jika siswa mata sipit natural sering kena penalti. |
| **Min Engagement** (`blink_heavy_min`) | 0.30 | 0–0.5 | Nilai minimum Engagement meski mata sangat ngantuk. 0.30 artinya Engagement tidak pernah nol hanya karena mata setengah tertutup. |

---

### Confusion

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **Iris Up Dead Zone** (`iris_up_dead_zone`) | 0.15 | 0–0.4 | iris_y < −nilai_ini sebelum iris_up_v mulai dihitung. Memfilter pergerakan iris ke atas yang kecil. |
| **Iris Up Range** (`iris_up_range`) | 0.30 | 0.1–0.6 | Range iris_up_v setelah dead zone untuk skor 1.0. |
| **LookUp Threshold** (`look_up_threshold`) | 0.35 | 0.1–0.7 | `eyeLookUp / threshold = look_up_v`. Turunkan untuk mendeteksi lirik atas yang subtle. |
| **Pitch Start** (`pitch_start`) | 5.0° | 0–15 | Kepala mendongak melebihi nilai ini mulai berkontribusi ke Confusion. Berbeda dari Boredom: mendongak sedikit = berpikir. |
| **Pitch Range** (`pitch_range`) | 15.0° | 5–30 | Range pitch mendongak untuk skor penuh. |
| **BrowDown Th** (`brow_dn_th`) | 0.23 | 0.1–0.5 | Threshold browDown untuk sinyal brow_dn_v. Alis tidak harus turun banyak — 0.23 cukup sensitif. |
| **BrowInnerUp Th** (`brow_in_th`) | 0.30 | 0.1–0.5 | Threshold browInnerUp (alis dalam naik). **Harus ada co_signal** (lihat di bawah). |
| **BrowInner CoGate** (`brow_in_co_gate`) | 0.25 | 0.1–0.5 | co_signal (max iris_up, look_up, pitch) harus melebihi nilai ini agar browInnerUp dihitung. **Parameter kunci** untuk mencegah false positive pada siswa dengan alis natural melengkung. Naikkan untuk lebih ketat, turunkan untuk lebih sensitif. |
| **Smile Penalty Th** (`smile_penalty_th`) | 0.15 | 0–0.3 | mouthSmile > nilai ini mulai mengurangi skor Confusion (senyum = tidak bingung). |
| **Jaw Start** (`jaw_start`) | 0.05 | 0–0.2 | jawOpen < nilai ini → jaw_val_conf = 0 (mulut tertutup tidak berkontribusi). |
| **Jaw Peak** (`jaw_peak`) | 0.25 | 0.1–0.5 | jawOpen = nilai ini → jaw_val_conf = 1.0 (titik puncak mangap Confusion). |
| **Jaw End** (`jaw_end`) | 0.40 | 0.3–0.8 | jawOpen > nilai ini → jaw_val_conf = 0 (sudah terlalu lebar = bukan confusion, lebih ke menguap). |
| **Pucker Th** (`pucker_th`) | 0.30 | 0.1–0.6 | mouthPucker / threshold = pucker_co. Bibir mengerucut saat berpikir (langsung, tanpa gate). |

---

### Frustration

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **BrowDown Th** (`brow_dn_th`) | 0.40 | 0.1–0.6 | Frustration butuh alis turun lebih dalam dari Confusion (0.40 vs 0.23). |
| **NoseSneer Th** (`nose_sneer_th`) | 0.20 | 0.05–0.5 | **Sinyal terkuat.** noseSneer (mengernyit hidung) jarang terjadi alami — nilai rendah sudah cukup untuk efek besar. |
| **CheekSquint Th** (`cheek_squint_th`) | 0.40 | 0.1–0.6 | Pipi menegang ke atas (senyum otot tegang). |
| **MouthPress Th** (`mouth_press_th`) | 0.40 | 0.1–0.6 | Bibir ditekan keras bersama. |
| **EyeSquint Th** (`eye_squint_th`) | 0.40 | 0.1–0.6 | Mata menyipit dengan otot — berbeda dari mengantuk (blink). |
| **Jaw Start** (`jaw_start`) | 0.10 | 0–0.3 | jawOpen < nilai ini = kontribusi rahang ke Frustration = 0. |
| **Jaw Range** (`jaw_range`) | 0.20 | 0.05–0.5 | Range jawOpen di atas jaw_start untuk kontribusi penuh. |

---

### Hybrid Weights & SigLIP

#### SigLIP Empirical Bias

| Parameter | Default | Range | Penjelasan |
|---|---|---|---|
| **SigLIP Empirical Bias** (`empirical_bias`) | 3.5 | 1.0–6.0 | Ditambahkan ke logit SigLIP sebelum sigmoid. Logit zero-shot biasanya negatif (-3 sampai -6). Bias +3.5 menggeser ke area sigmoid yang sensitif (0.2–0.9). **Naikkan** jika semua skor SigLIP terlalu rendah. **Turunkan** jika terlalu tinggi. |

#### ~~Restless Bonus (Boredom Temporal)~~ — DIHAPUS

Parameter `restless_bonus_max` / `restless_std_min` / `restless_std_range` **sudah dihapus** dari kode: menambah skor Boredom dari gerakan kepala antar-frame **tidak punya dasar paper** (Craig 2008 hanya memvalidasi AU43 untuk Boredom). Tidak ada lagi slider ini di Rules Editor.

#### Bobot Per Label

Setiap label memiliki dua bobot yang dijumlahkan dan dinormalisasi:

| Label | SigLIP default | Landmark default | Rekomendasi |
|---|---|---|---|
| **Boredom** | 0.25 | 0.75 | Landmark dominan — gaze (geometrik) + AU43; SigLIP kecil (tired/vacant) |
| **Engagement** | 0.50 | 0.50 | **HOLISTIK** — tak ada AU tunggal dominan (Whitehill 2014) → SigLIP tertinggi |
| **Confusion** | 0.35 | 0.65 | AU4+AU7 (MediaPipe, stretch agresif) primer + SigLIP jaring pengaman |
| **Frustration** | 0.30 | 0.70 | AU1+2 (brow_raise 0.70) + AU4/AU14 (face_weight 0.65) + SigLIP gestalt stres |

Bobot **tidak harus berjumlah 1.0** — sistem otomatis menormalisasi. Contoh: SigLIP=3, Landmark=1 → `α = 0.75, β = 0.25`.

---

## 4. Alur Kerja Recalculate

Recalculate **tidak menjalankan ulang model**. Prosesnya:

```
raw_cache/{video}.json   +   siglip_cache/{video}.json
         ↓                           ↓
reconstruct LandmarkResult    ambil siglip_scores per frame
         ↓
compute_emotion_scores(lr, rules_baru)   ← PAKAI RULES YANG BARU
         ↓
hybrid_score = α × siglip + β × landmark  ← PAKAI BOBOT YANG BARU
         ↓
threshold → frame_preds → vote → prediction  ← PAKAI THRESHOLD SAAT INI
         ↓
update batch_history + frame_annotations + viz thumbnail
```

**Syarat recalculate berhasil:** Video harus sudah pernah diproses sehingga kedua cache ada. Video yang belum pernah diproses tetap di-skip.

**Yang berubah setelah recalculate:**
- `batch_history.json` — semua avg_score, prediction, frame_scores diperbarui
- `frame_annotations.json` — label per frame diperbarui
- Viz thumbnail video aktif — di-render ulang dengan skor emosi baru
- AI score bars di panel kanan — menampilkan nilai terbaru

**Yang tidak berubah:**
- Crop wajah clean (`cropped_faces/clean/`) — tidak perlu re-crop
- Label manual yang diset oleh labeler — frame_annotations yang diubah manual tetap aman
- File raw_cache dan siglip_cache — tidak dimodifikasi

---

## 5. Tips & Strategi Tuning

### Workflow yang Disarankan

1. **Proses beberapa video sample** dengan Batch Semua (atau Proses Video Ini per video)
2. **Aktifkan Viz** (toggle di topbar) untuk melihat sinyal per frame
3. **Buka Rules Editor** — perhatikan parameter mana yang kemungkinan bermasalah
4. **Ubah satu parameter** → klik **Recalculate Batch** → lihat perubahan di viz dan AI score bars
5. **Iterasi** sampai hasil memuaskan
6. **Simpan** rules

### Gunakan Viz untuk Diagnosis

Saat Viz aktif, thumbnail menampilkan:
- **Panah gaze** (kuning) — panjang dan arah menunjukkan gaze_dev. Panah panjang ke samping = Boredom tinggi.
- **Bars blendshape** — nilai masing-masing otot wajah. Lihat apakah browDown, noseSneer, dll. tinggi.
- **Bars emosi** (bawah) — skor akhir Boredom/Engagement/Confusion/Frustration per frame setelah rules diterapkan.

### Jangan Ubah Semua Sekaligus

Ubah satu parameter, recalculate, dan amati efeknya sebelum mengubah parameter lain. Parameter saling berinteraksi — perubahan ganda bisa menyebabkan efek yang sulit ditelusuri.

### Threshold vs Rules

- **Threshold** (slider di panel kanan per label) — mempengaruhi keputusan akhir (0 atau 1) tanpa mengubah skor. Paling mudah disesuaikan.
- **Rules** — mengubah cara skor dihitung. Lebih fundamental, butuh recalculate.

Mulai dari threshold dulu. Jika threshold tidak cukup (misal skor selalu terlalu rendah bahkan di 0.1), baru ubah rules.

### Simpan Versi Rules

Sebelum eksperimen besar, gunakan **"Buat batch baru"** dengan nama yang mencerminkan konfigurasi (misal `batch_rules_v2_confusion_adjusted`). Ini mempertahankan hasil lama sebelum perubahan rules diterapkan.

---

## Dokumen Terkait

| Dokumen | Relevansi |
|---|---|
| [docs/COMPUTATION.md](COMPUTATION.md) | Rumus lengkap setiap parameter — apa yang terjadi secara matematis saat nilai diubah |
| [core/README_SIGLIP.md](../core/README_SIGLIP.md) | Arsitektur pipeline, cara SigLIP bekerja, sistem cache |
| [README.md](../README.md) | Cara install, panduan UI, FAQ umum |
