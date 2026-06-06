# Alur Metode Sistem Pelabelan Emosi — Penjelasan Berurutan

Dokumen ini menjelaskan **alur metode dari awal** secara runtut dan mudah dipahami. Setiap langkah:
kutipan **verbatim (Bahasa Inggris)** dari paper, lalu **penjelasan (Bahasa Indonesia)**, lalu **wujudnya di kode**.
Detail lengkap (semua kutipan + link halaman PDF) ada di [`ACADEMIC_BASIS.md`](ACADEMIC_BASIS.md); keputusan teknis di [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md). Dokumen ini **tidak menggantikan** keduanya — hanya merangkai alurnya.

---

## Langkah 0 — Tujuan aplikasi

Aplikasi ini **melabeli emosi belajar siswa dari video, per-frame**, menghasilkan **4 label** (Boredom, Engagement, Confusion, Frustration). Label per-frame ini dipakai untuk: (a) melatih **model realtime frame-level** (mis. MobileNet) sebagai *student*, dan/atau (b) dikirim ke **LLM** agar tahu kondisi emosi siswa saat itu. Sistem ini berperan sebagai *teacher/annotator* otomatis.

---

## Langkah 1 — Kenapa 4 label ini (dari dataset DAiSEE + literatur)?

> "One finding is that confusion, frustration, boredom, and engagement/flow are the major affective states that students experience across diverse learning contexts, student populations, and methods to track emotions."
> — D'Mello & Graesser (2012)

> "our dataset consists of labels for four affective states related to user engagement, viz., engagement, frustration, confusion, and boredom. Recent work has shown that the six basic expressions ... are not reliable in prolonged learning situations."
> — Gupta et al. (2016), DAiSEE

**Penjelasan:** Empat emosi inilah yang **paling sering muncul saat belajar** (D'Mello) dan menjadi label dataset **DAiSEE** — acuan utama kita karena konteksnya persis sama (e-learning in-the-wild). Enam emosi dasar (marah/jijik/takut/dll.) sengaja **tidak** dipakai karena tidak reliable di situasi belajar berdurasi panjang.

**Di kode:** `LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]` (`ui/constants.py`). *(verbatim + halaman: ACADEMIC_BASIS §1, §9)*

---

## Langkah 2 — Kenapa MULTI-LABEL (4 label biner), bukan pilih-1-dari-4?

> "facial expressions in human daily life are in multiple or co-occurring mental states ... they appear as combinations, blends, or compounds of different basic emotions."
> — Li & Deng (2019), RAF-ML

> "the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization."
> — Zhai et al. (2023), SigLIP

**Penjelasan:** Emosi bisa **muncul bersamaan** (mis. *productive confusion*: bingung sambil tetap engaged — D'Mello). Maka tiap emosi dinilai **independen**: 4 label, masing-masing **biner (2 kelas: ada/tidak)**. Inilah alasan memakai **SigLIP (sigmoid)**, bukan CLIP (softmax): sigmoid memberi skor tiap emosi yang berdiri sendiri (cocok multi-label); softmax memaksa satu pemenang (single-label).

**Di kode:** `torch.sigmoid(...)` per-label di `core/inference.py`; prediksi `1 if avg_score >= threshold else 0` per label. *(ACADEMIC_BASIS §10, §11)*

---

## Langkah 3 — Apa yang diukur? Action Unit (AU) FACS

> "The Facial Action Coding System (Ekman & Friesen, 1978) is an objective method for quantifying facial movement in terms of component actions."
> — Bartlett et al. (1999)

**Penjelasan:** Emosi diukur lewat **gerakan otot wajah objektif (Action Unit FACS)**, bukan tebakan kategori. Karena blendshape MediaPipe **bukan** AU FACS sungguhan (terbukti lemah untuk AU alis: korelasi AU4 r=0.09 vs detektor FACS), ekstraksi AU untuk **Confusion & Frustration** memakai **MediaPipe FaceLandmarker blendshape** (dikonversi ke AU via normalisasi baseline-relative, kalibrasi stretch agresif AU4). MediaPipe juga dipakai untuk gaze, AU43-boredom, dan visualisasi — semua dalam satu proses (TANPA py-feat).

**Di kode:** `core/action_units.py`. Semua AU sinkron dalam satu proses — tidak ada subprocess. *(ACADEMIC_BASIS §2, §3; DESIGN_RATIONALE §16)*

---

## Langkah 4 — Pemetaan AU → tiap emosi (inti, dari Craig 2008)

> "AUs 1, 2, and 14 were primarily associated with frustration ... Confusion displayed associations with AUs 4, 7, and 12 ... boredom displayed a significant association with action unit 43 (eye closure)."
> — Craig et al. (2008), Table 2 / p. 784

**Penjelasan & wujud di kode:**

- **Frustration** = **AU1 (alis dalam naik) + AU2 (alis luar naik)** bersamaan (Craig, 100%) + AU4/AU14 sekunder (Grafsgaard 2013). → `au["AU1"]·au["AU2"]` (MediaPipe baseline-normalized).
- **Confusion** = **AU4 (alis turun) + AU7 (kelopak menegang)** (Craig 95%/78%) + AU12 sebagai *gate*. → `au["AU4"]`, `au["AU7"]` (MediaPipe baseline-normalized). *Tidak* pakai gaze/AU1 (Craig tidak mengaitkannya ke confusion).
- **Boredom** = **AU43 (mata menutup)** (Craig) **+ gaze menjauh dari layar**. → `eyeBlink` (MediaPipe, tervalidasi r=0.51) + gaze.
- **Engagement** = **penampilan holistik** (lihat Langkah 5) — Craig **tidak** menemukan AU primer untuk engagement.

*(verbatim per emosi + halaman: ACADEMIC_BASIS §4 Frustration, §5 Confusion, §6 Boredom)*

---

## Langkah 5 — Gaze, Head-Pose, & Engagement holistik

> "most of the information about the appearance of engagement is contained in the static pixels, not the motion per se."
> — Whitehill et al. (2014)

> "[Gaze Tutor] monitor a student's gaze patterns and identify when the student is bored, disengaged, or is zoning out."
> — D'Mello, Olney, Williams & Hays (2012)

**Penjelasan:**
- **Engagement** tidak punya AU tunggal → dinilai dari **penampilan wajah menyeluruh** (Whitehill) via SigLIP + tatapan ke layar + mata terbuka.
- **Gaze/head-pose** (menatap vs menjauh) jadi penanda atensi: *konstruk*-nya dari GazeTutor (gaze menjauh = bored/disengaged) & Whitehill ("looking away"); *metode webcam* (head pose) dari Sümer et al. (2021). Bukan AU.

**Di kode:** `gaze_dev_*`, `yaw/pitch/roll gate` (MediaPipe). *(ACADEMIC_BASIS §7, §8.5; DESIGN_RATIONALE §11)*

---

## Langkah 6 — Hybrid: SigLIP (holistik/zero-shot) + Landmark (AU/gaze)

> "we explored a strategy that integrates multiple frames within 1-2 second video clips to enhance labeling performance and reduce costs."
> — Zhang & Fu (2025)

**Penjelasan:** Skor akhir tiap emosi = **gabungan** SigLIP (menangkap penampilan holistik, ala Whitehill) + skor landmark (AU/gaze). Bobotnya beda per emosi (mis. Confusion 35/65 (land=primer, SigLIP=jaring pengaman oklusi)). Pelabelan zero-shot + multi-frame ini didukung paradigma Zhang & Fu (2025).

**Di kode:** `siglip_w`/`land_w` di `rules.py`; digabung di `core/inference.py`. *(DESIGN_RATIONALE §1, §2, §16; ACADEMIC_BASIS §11)*

---

## Langkah 7 — Frame-level & dual-label (2 pasangan eksklusif)

> "engagement labels of 10-second video clips can be reliably predicted from the average labels of their constituent frames (Pearson r = 0.85)."
> — Whitehill et al. (2014)

> "When engagement is low, boredom is generally high and vice-versa."
> — Gupta et al. (2016), DAiSEE

**Penjelasan:** Label dibuat **per-frame (statis)** — sesuai temuan Whitehill bahwa frame statis sudah informatif (cocok untuk target model realtime frame-level). Praktis maksimal **2 label aktif**, karena 2 pasangan **saling eksklusif**: Boredom/Engagement (kuat: DAiSEE/D'Mello) dan Confusion/Frustration (penyederhanaan desain). *(ACADEMIC_BASIS §11)*

---

## Langkah 8 — Yang JUJUR bukan dari paper (limitasi)

Agar tidak over-claim, ini bagian yang **engineering / interpretasi**, bukan ketentuan paper:
- **Angka** (threshold deteksi, anchor AU, bobot hybrid, dead-zone) = kalibrasi empiris — paper tak memberi angka untuk MediaPipe/SigLIP.
- **Cross-suppression** — hanya **Bore↔Eng** yang dipertahankan (DAiSEE "complementary" + D'Mello near-exclusive). Conf→Eng / Conf→Bore / Frus→Bore + dominance-gap + strict-rules **DIHAPUS** (tak berdasar / lawan multi-label). Bore↔Eng = operasionalisasi per-frame dari temuan eksklusivitas.
- **Hand-over-face** — dipetakan ke dua emosi (lihat DESIGN_RATIONALE §4):
  - **Semua tangan (`max(hand_one,hand_two)`) → cue KUAT Confusion**: Behera 2020 (*HoF naik saat difficulty ↑*) + D'Mello 2012 (*Confusion = cognitive disequilibrium saat impasses*) + Mahmoud 2016. Dua paper saling menguatkan; Behera & Mahmoud tidak bedakan jumlah tangan.
  - **2 tangan → cue pendukung Frustration JUGA**: Grafsgaard 2013b (verbatim: *two-hands ↔ self-efficacy rendah*). 2-tangan boleh contribute ke Confusion DAN Frustration (multi-label).
  - Bobot kecil (0.20) → hanya menambah, tidak memicu sendiri.
- **Gaze webcam** — konstruk dari eye-tracker (GazeTutor); metode webcam (Sümer) lebih kasar.

*(rincian: DESIGN_RATIONALE §15, §17)*

---

## Ringkasan verifikasi: kode ↔ standar paper

| Komponen | Standar paper | Sesuai di kode? |
|---|---|---|
| 4 label emosi | DAiSEE 2016, D'Mello 2012 | Ya: `LABELS` |
| Multi-label biner (sigmoid) | RAF-ML/DAiSEE + Zhai 2023 | Ya: `torch.sigmoid` per-label |
| Frustration = AU1+AU2 (+AU4/14) | Craig 2008, Grafsgaard 2013 | Ya: `au["AU1"]·au["AU2"]` (MediaPipe blendshape→AU) |
| Confusion = AU4+AU7 (gate AU12) | Craig 2008 | Ya: `au["AU4"]`,`au["AU7"]` (MediaPipe blendshape→AU) |
| Boredom = AU43 + gaze | Craig 2008 + GazeTutor/Sümer | Ya: `au["AU43"]` MediaPipe eyeBlink + gaze |
| Engagement = holistik + gaze gate | Whitehill 2014 + GazeTutor | Ya: SigLIP + gaze gate inference + AU43 anti-engagement |
| AU FACS asli (semua emosi) | Bartlett/Craig (FACS) | Ya: MediaPipe blendshape (baseline-normalized)`au_source`; scoring TIDAK akses blendshape langsung |
| Hand-over-face (any) → Confusion (kuat) | Behera 2020 (HoF ↑ saat difficulty ↑) + D'Mello 2012 (Confusion=cognitive disequilibrium) + Mahmoud 2016 | Ya: `max(hand_one,hand_two) * hand_conf_w(0.78)` |
| Hand-over-face: 2-tangan → Frustration (pendukung) | Grafsgaard 2013b self-efficacy | Ya: `hand_two * hand_frus_w(0.40)` |
| Gaze gate Engagement di hybrid final | Whitehill 2014 + GazeTutor | Ya: `eng_gaze_gate_*` di inference.py |

**Kesimpulan:** *Apa* yang diukur tiap emosi **sudah mengikuti standar paper**. Mekanisme tak-berdasar (suppress, dominance-gap, strict-rules, restless bonus, akses blendshape langsung, eye_wide) **sudah DIHAPUS**. Yang tersisa sebagai engineering jujur: **angka kalibrasi** (threshold/anchor/bobot) & **gaze webcam** (konstruk dari eye-tracker, metode webcam lebih kasar).
