# Design Rationale — Keputusan Teknis Berbasis Paper

Dokumen ini menjelaskan **mengapa** setiap keputusan teknis utama dibuat, dikaitkan langsung ke temuan paper. Kutipan dalam tanda `"…"` adalah verbatim dari sumber asli.

---

## 1. Mengapa Hybrid SigLIP + MediaPipe?

**Masalah:** Tidak ada satu sensor/model pun yang cukup untuk mendeteksi semua 4 emosi belajar secara akurat.

**Dari paper:**

Craig et al. (2008) memvalidasi AU wajah spesifik per emosi menggunakan FACS:
> "We were able to detect facial AUs and patterns of AUs that occur during the affective states of confusion, frustration, and boredom." *(p. 785)*

Namun Whitehill et al. (2014) menemukan bahwa untuk **Engagement**, penampilan wajah secara holistik lebih informatif dari AU individual:
> "We hypothesize that a good deal of the information used by humans to make engagement judgements is based on the student's face." *(p. 2)*
> "This accuracy is quite high and suggests that most of the information about the appearance of engagement is contained in the static pixels, not the motion per se." *(§2.4)*

**Konsekuensi desain:** SigLIP (vision-language model) menangkap representasi holistik wajah seperti yang dimaksud Whitehill — tanpa perlu mendefinisikan AU satu per satu. MediaPipe/landmark menangkap AU spesifik yang divalidasi Craig. Kedua sinyal digabung per-label dengan bobot yang berbeda.

---

## 2. Mengapa Bobot SigLIP dan Landmark Berbeda Per Emosi?

**Kode:** `rules.py → hybrid → siglip_w / land_w`

```python
"siglip_w": [0.25, 0.50, 0.35, 0.30],  # [Bore, Eng, Conf, Frus]
"land_w":   [0.75, 0.50, 0.65, 0.70],
# SigLIP tertinggi di Engagement (holistik, Whitehill 2014). Emosi AU-diskrit (Craig 2008)
# bertumpu MediaPipe AU (baseline-normalized) TAPI tetap diberi SigLIP sbg cross-check.
```

**Prinsip:** SigLIP **tertinggi di Engagement** (satu-satunya emosi holistik tanpa AU dominan — Whitehill 2014). Tiga emosi lain punya **AU diskrit** (Craig 2008) → bertumpu **AU MediaPipe (blendshape→AU FACS, baseline-normalized) + cue tangan/mulut**, tapi **tetap diberi SigLIP** sebagai *cross-check independen*. Ini **ada dasarnya**: SigLIP = expression reader tervalidasi (Zhai 2023), dan D'Mello 2009 memakai *"facial features"* untuk **keempat** emosi (§8). Nilai bobotnya = kalibrasi empiris yang sah (status sama dengan threshold), **bukan** mekanisme tak-berdasar seperti restless bonus.

### Engagement — SigLIP 50%, Landmark 50% (SigLIP tertinggi)

**Alasan:** Whitehill et al. (2014) menunjukkan engagement dideteksi dari penampilan **holistik** wajah (static pixels), bukan AU tertentu. Inilah satu-satunya emosi tanpa AU diskrit → SigLIP berperan paling besar.

> "Furthermore, we found that engagement labels of 10-second video clips can be reliably predicted from the average labels of their constituent frames (Pearson r = 0.85), suggesting that static expressions contain the bulk of the information about engagement used by observers." *(abstract)*

Craig et al. (2008) tidak menemukan AU primer untuk Engagement → justru ini alasan SigLIP-nya tertinggi (0.50). Landmark tetap 0.50 karena gaze (Whitehill level 1: "looking away") di-gate setelah scoring.

### Frustration — SigLIP 30%, Landmark 70%

**Alasan:** Craig et al. (2008) menemukan AU spesifik untuk Frustration dengan coverage 100%:
> "It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together." *(p. 784)*

AU1+AU2/AU4/AU14 (MediaPipe blendshape, baseline-normalized) = **primer** (land_w 0.70). SigLIP 0.30 pelengkap (gestalt stres + jaring pengaman saat landmark gagal).

### Confusion — SigLIP 35%, Landmark 65% (AU primer + SigLIP jaring pengaman)

**Riwayat:** Dulu **SigLIP 65%** karena AU4 (Craig 95%) **hampir mati** di MediaPipe `browDown` (median=0.001). Setelah kalibrasi stretch agresif (`AU4_active=0.05`) + noseSneer co-occur booster + AU7 standalone (78% Craig), AU MediaPipe cukup reliable → SigLIP diturunkan 0.65→0.35.

**Final:** AU4+AU7 (MediaPipe, stretch agresif) = **primer** (land_w 0.65). **SigLIP 0.35 dipertahankan** sebagai jaring pengaman oklusi tangan (SigLIP tetap membaca crop walau AU gagal).

### Boredom — SigLIP 25%, Landmark 75%

**Alasan:** AU43 (eye closure, Craig 2008) + gaze deviation reliable & geometrik oleh MediaPipe → landmark **primer** (0.75). SigLIP 0.25 pelengkap (tampang lelah/kosong yang holistik).

---

## 3. Mengapa AU1+AU2 Co-occurrence untuk Frustration?

**Kode:** `landmark_analyzer.py` — `brow_raise_co = (bou_fr * biu_fr) ** 0.5`

**Alasan dari Craig et al. (2008) Table 2:**

| Pattern | Deskripsi | Coverage |
|---|---|---|
| AU outer brow raise | Outer brow raise | 100% |
| AU inner brow raise | Inner brow raise | 100% |
| Keduanya bersama | Inner and outer brow raised together | 100% |
| Aturan asosiatif | Inner brow raise → outer brow raise dan sebaliknya | 100% confidence |

> "a strong association was found for a link between AUs 1 and 2 occurring together. Additionally, these AUs mutually trigger each other. That is, a raised inner brow tends to trigger a raised outer brow, and vice versa." *(p. 784)*

**Kenapa geometric mean `(A × B)^0.5`?** Ini mengimplementasikan syarat co-occurrence secara matematika: sinyal hanya tinggi jika **keduanya** aktif. Jika salah satu lemah (misal outer=0.8, inner=0.1), hasilnya 0.28 — jauh lebih rendah dari rata-rata (0.45). Ini sesuai temuan paper bahwa kedua AU *selalu muncul bersama* dalam frustration, bukan satu saja.

**Kenapa `brow_raise_direct_w = 0.85` jauh lebih tinggi dari `face_weight = 0.65`?**

- `brow_raise_direct_w = 0.85` → AU1+AU2 (inner+outer brow raise), coverage **100%** di Craig 2008 Table 2 → sinyal **PRIMER** Frustration, bobot tertinggi. Dinaikkan dari 0.70 (kompensasi MediaPipe-only + mengangkat frustrasi yang sebelumnya kurang terdeteksi).
- `face_weight = 0.65` → AU4 (brow lowerer) + AU14 (dimpler), sinyal **pendukung** mengikuti Grafsgaard et al. (2013) yang menemukan AU4/AU14 sebagai korelat positif frustrasi. Di bawah AU1+AU2 karena AU4 lemah-terukur di MediaPipe (browDown median 0.001).

Sinyal legacy lain (nose sneer, cheek squint) yang tidak muncul di Craig Table 2 sebagai sinyal primer frustration tetap berbobot lebih rendah.

---

## 4. Hand Signals — cue KUAT untuk Confusion (HoF) DAN Frustration (2-tangan)

> **KOREKSI:** Versi lama bagian ini menyatakan "tidak ada paper yang memvalidasi posisi tangan". **Itu keliru.** Proceedings ACII 2011 (LNCS 6974 — PDF yang sama dengan paper Grafsgaard confusion) memuat:
> - **Mahmoud, Baltrušaitis, Robinson & Riek (2011), "3D Corpus of Spontaneous Complex Mental States"** — analisis **kuantitatif** hand-over-face: *"hand-over-face gestures appeared in 20.8% of the segments (94 segments)"*, dikodekan per **hand shape, action, dan facial region occluded**, dikaitkan dengan **complex mental states**.
> - **Mahmoud & Robinson (2011), "Interpreting Hand-Over-Face Gestures"** — *"different positions and actions of the hand occluding the face can imply different affective states"*.

**Jadi hand-over-face PUNYA dasar paper** sebagai cue afektif/kognitif (sebelumnya Craig FACS-wajah + D'Mello "gross body language = seat-pressure pad" memang tidak mencakup tangan — Mahmoud-lah yang menutup celah ini).

**⚠️ Catatan konteks (jangan force-fit):** state yang dianalisis Mahmoud = **complex mental states** taksonomi Baron-Cohen (thinking, unsure, interested, dll.), pada sesi computer-based + dyadic — **bukan persis** 4 affect belajar (boredom/confusion/frustration/engagement). Hand-over-face paling kuat terkait **state kognitif/"thinking"**. Konsekuensi desain:
- *Keberadaan* sinyal tangan = berdasar paper (Mahmoud: 14/15 segmen index-finger = thinking/unsure ≈ **93% coverage**).
- *Pemetaan spesifik* gesture→emosi-belajar = interpretasi; deteksi kita **count-based** (tak bisa bedakan jari-aktif kognitif vs bersandar-pasif). **Ambiguitas posisi ini diselesaikan oleh anotator manusia** saat review (lihat PANDUAN: dagu→Confusion, dahi/2-tangan→Frustration).

**Keputusan: cue KUAT, bukan lemah.** Karena coverage Mahmoud (93%) bahkan **lebih tinggi dari AU7 (78%)**, dan Confusion adalah emosi yang paling sering **under-detected**, bobot tangan dinaikkan **0.40→0.78 = setara `au7_alone_w`** (sinyal AU diskrit terkuat). Tangan jadi cue first-class, bukan sekadar penambah kecil.

**Dukungan konteks-belajar (jembatan ke emosi-belajar):** Behera et al. (2020, IJAIED) menemukan **HoF naik saat difficulty ↑** (beban kognitif) → menjembatani HoF ke **Confusion** (D'Mello 2012: confusion = cognitive disequilibrium saat materi sulit). Grafsgaard et al. (2013) "Embodied Affect in Tutorial Dialogue" menemukan **two-hands-to-face lebih sering pada siswa dengan self-efficacy rendah** (≈ Frustration) → menjembatani 2-tangan ke **Frustration**.

**Pemetaan tangan ke emosi (keputusan akhir, argumentasi dua paper):**

**HoF — `max(hand_one, hand_two)` → cue KUAT Confusion** (beban kognitif):
- Dua paper saling menguatkan: **Behera 2020** (*"HoF naik saat difficulty ↑"*) + **D'Mello 2012** (*"Confusion = cognitive disequilibrium saat menghadapi impasses/materi sulit"*). Behera & Mahmoud 2016 tidak membedakan jumlah tangan → pakai `max(hand_one, hand_two)`.
- Cue KUAT (`hand_conf_w=0.78`, setara AU7) → kontribusi besar. Landmark-alone ≈ 0.66 (di bawah threshold 0.5 setelah hybrid), jadi masih **belum memicu sendiri** tapi mendekati — cukup kuat agar Confusion tidak under-detect. Tetap memungkinkan co-occur Engagement+Confusion (*productive confusion*).

**`hand_two` (≥2 tangan) → cue pendukung Frustration** (tambahan dari Confusion):
- **Grafsgaard 2013b** (verbatim langsung): *"two-hands-to-face gestures occurred significantly more frequently among students with low self-efficacy."* Low self-efficacy ≈ Frustration D'Mello.
- `hand_frus_w=0.40` (pendukung). 2-tangan boleh co-occur ke Confusion DAN Frustration (multi-label).

Set `hand_conf_w=0` / `hand_frus_w=0` untuk menonaktifkan. Tangan tetap dideteksi + tampil di viz.

---

## 5. Mengapa AU4+AU7 Co-occurrence untuk Confusion?

**Kode:** `au4_au7_co = (brow_dn_v * au7_v) ** 0.5` dengan `au4_au7_co_w = 0.50`

**Alasan dari Craig et al. (2008) Table 2:**

| AU | Deskripsi | Coverage |
|---|---|---|
| AU4 | Brow lowerer | 95% |
| AU7 | Lid tightener | 78% |
| AU4+AU7 | Brow lowered with tightened lids | 73% |
| AU7→AU4 | Tightened lids will lead to a lowered brow | 52% confidence |

> "Confusion displayed associations with AUs 4, 7, and 12. Action units 4 and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow)." *(p. 784)*

**Kenapa `au4_au7_co_w = 0.50`?** Co-occurrence coverage 73% (tidak 100% seperti frustration). Maka bobotnya lebih rendah (0.50) dibanding brow_raise_direct_w Frustration (0.85). AU4 tetap memiliki bobot sendiri (95% coverage), sehingga co-occurrence adalah sinyal tambahan di atas sinyal AU4 individual.

**Kenapa AU7 juga dipakai STANDALONE (`au7_alone_w = 0.78`)?**

Craig Table 2 mencantumkan AU7 dengan coverage **78% sebagai pola tersendiri** — bukan hanya dalam co-occurrence dengan AU4. Versi lama kode hanya memakai AU7 saat co-occurrence dengan AU4, sehingga ketika AU4 (browDown) **nyaris mati di MediaPipe** (median 0.001, p99 0.18), seluruh sinyal landmark confusion ikut mati — inilah penyebab utama Confusion jauh under-detected dibanding Frustration.

`base_conf = max(AU4, AU7×0.78, (AU4·AU7)^0.5×0.50)` membuat AU7 (eyeSquint, yang **terukur baik**: median 0.31, p90 0.50) bisa menyalakan confusion sendiri sesuai temuan 78% Craig. Bobot 0.78 = rasio coverage AU7(78%)/AU4(95%) ≈ 0.82, dibulatkan ke 0.78. Validasi di 21.204 frame: confusion land-score p90 naik dari **0.149 → 0.732** setelah AU7 standalone diaktifkan.

---

## 6. Mengapa AU12 (Smile) adalah Gate Floor untuk Confusion, Bukan Suppressor Penuh?

**Kode:** `smile_gate = clamp(1.0 - au["AU12"]/gate_th, floor=0.30, max=1.0)`

**⚠️ KOREKSI:** AU12 **tidak** muncul di Table 2 Craig 2008 dengan coverage 95% (klaim itu keliru di versi lama doc ini — 95% adalah coverage **AU4**). AU12 hanya disebut di **prosa** sebagai asosiasi **sekunder yang lebih lemah** dan lintas-emosi:

> "AU 12, AU 14, and AU 43, which received less support from the associations…" *(p. 784)*
> "Notable exceptions are AUs 12 and 14 that occur during expressions of both confusion and frustration." *(p. 785)*

**Konsekuensi desain:** karena AU12 lemah dan lintas-emosi (muncul di confusion DAN frustration), AU12 **tidak** dipakai sebagai sinyal positif confusion. Ia hanya jadi *gate* lembut: senyum lebar sedikit meredam confusion, tapi `floor = 0.30` mencegah senyum men-zero-kan confusion sepenuhnya (karena "questioning smile" memang bisa co-occur dengan kebingungan).

`gate_th = 0.45` (skala intensitas AU12 ter-normalisasi) artinya senyum lemah/sedang tidak menekan confusion.

---

## 7. Mengapa BrowInnerUp (AU1) Tidak Dipakai untuk Confusion?

**Alasan:** Craig et al. (2008) Table 2 **tidak** mencantumkan AU1 (inner brow raise) atau AU2 (outer brow raise) sebagai sinyal confusion. AU1 dan AU2 adalah sinyal **frustration** (100% coverage):

> "It appears that AUs 1, 2, and 14 were primarily associated with frustration" *(p. 784)*

Confusion hanya divalidasi dengan AU4 (brow lowerer 95%), AU7 (lid tightener 78%), AU4+AU7 co-occurrence (73%), dan AU12 secondary (95%). BrowInnerUp **tidak dipakai sama sekali** untuk confusion scoring.

---

## 8. Mengapa AU43 (eyeBlink) sebagai Sinyal Independen untuk Boredom?

**Kode:** `blink_direct = clamp((blink_corrected - 0.45) / range, 0, 1) * 0.45`

**Alasan dari Craig et al. (2008) Table 2:**

| AU | Deskripsi | Coverage |
|---|---|---|
| AU43* | Eye closure | 40% |

*Note: Of secondary importance*

> "While boredom displayed a significant association with action unit 43 (eye closure), no association rules between AUs were observed." *(p. 784)*

**Kenapa independen dari gaze gate?** Coverage 40% — AU43 muncul pada hampir separuh episode boredom bahkan ketika murid mungkin masih menghadap layar. Jika eye closure digatekeeping oleh gaze deviation, kita akan melewatkan kasus boredom yang valid di mana murid masih menatap layar tapi matanya mulai menutup (mengantuk/bosan tapi belum benar-benar noleh).

**Kenapa `blink_direct_w = 0.45`?** Coverage hanya 40% (vs. 100% untuk AU1+AU2 frustration), sehingga bobotnya jauh lebih rendah. AU43 adalah secondary importance dalam Table 2 Craig.

---

## 9. Mengapa Confusion dan Engagement Boleh Co-exist? (TANPA suppression)

**Kode:** TIDAK ada suppression Conf→Eng. Confusion & Engagement dibiarkan **co-occur bebas** (multi-label).

**Alasan dari D'Mello & Graesser (2012):**

> "The second hypothesis is the productive confusion hypothesis (Hypothesis 2). According to this hypothesis, cognitive disequilibrium, impasses, and confusion provide learners with an opportunity to think, deliberate, and problem solve." *(p. 149)*

> "The Confusion → Engagement/Flow and Confusion → Frustration transitions were also significant, while the Confusion → Boredom transition occurred at chance levels, thereby confirming the productive confusion and hopeless confusion hypotheses." *(p. 153)*

**Konsekuensi (revisi):** Karena transisi Confusion→Engagement/Flow **signifikan** (productive confusion), confusion & engagement memang **CO-OCCUR** — bukan saling meniadakan. Maka `conf_eng_suppress` yang dulu menekan engagement saat confusion tinggi **DIHAPUS** (itu kebalikan dari yang dikatakan paper). Sekarang keduanya bisa tinggi bersamaan — sesuai productive confusion + sifat multi-label DAiSEE. (Lihat §15.)

---

## 10. Mengapa Tidak Ada Chin-Resting untuk Boredom?

**Alasan:** Untuk **Boredom** spesifik, tidak ada paper yang memvalidasi chin-resting (Craig 2008 hanya AU43 eye closure untuk Boredom). Hand-over-face secara umum **memang** punya dasar (Mahmoud 2011, §4), tapi Mahmoud mengaitkannya ke state **kognitif/thinking**, bukan boredom — jadi chin-resting→Boredom tetap tidak didukung. Chin-resting tidak dipakai untuk boredom.

---

## 11. Mengapa Gaze Deviation untuk Boredom dan Engagement?

**Kode:** `gaze_dev_bore`, `gaze_dev_eng` — kombinasi yaw + iris_x + iris_y

Landasan dipisah **dua lapis** karena paper konstruk memakai eye-tracker, sedang sistem ini webcam:

**Lapis 1 — Konstruk (apa arti sinyalnya):** gaze/atensi menjauh dari konten = boredom/disengagement.
> "The tutor uses a commercial eye tracker to monitor a student's gaze patterns and identify when the student is bored, disengaged, or is zoning out." — *D'Mello, Olney, Williams & Hays (2012), Gaze Tutor.*
> "1: Not engaged at all – e.g., looking away from computer ... eyes completely closed." — *Whitehill et al. (2014), §2.2.*

**Lapis 2 — Metode (cara mengukur dari webcam):** head pose (yaw/pitch/roll) + offset iris MediaPipe, **tanpa** eye-tracker.
> "...training Attention-Net for head pose estimation and Affect-Net for facial expression recognition." — *Sümer et al. (2021), Multimodal Engagement Analysis from Facial Videos in the Classroom.*

**Limitasi (jujur):** Berbeda dari GazeTutor yang memakai eye-tracker terkalibrasi (Tobii), gaze webcam di sini lebih kasar. Karena itu **head pose dijadikan cue atensi utama** (sesuai Sümer), iris hanya penghalus arah. Threshold/dead-zone gaze (`scale_h`, `gaze_dead_zone`, dll) = kalibrasi empiris, bukan dari paper. Konstruk "gross body language" D'Mello et al. (2009) merujuk seat-pressure pad, **bukan** gaze — jadi gaze di sini disandarkan ke GazeTutor + Whitehill + Sümer, bukan ke kalimat itu.

**Kenapa gaze_dev_bore BERBEDA dari gaze_dev_eng?**

- `gaze_dev_bore` menyertakan roll dan hanya komponen gaze ke atas (bukan ke bawah). Murid yang nunduk baca masih engaged, bukan bosan.
- `gaze_dev_eng` tidak menyertakan roll (natural head tilt ≠ disengagement), tidak menyertakan gaze ke bawah (nunduk = baca/ngetik = engaged).

---

## 12. Mengapa SigLIP Menggunakan Prompt Bahasa Inggris Deskriptif?

**Kode:** SigLIP diberi prompt teks per emosi, bukan label satu kata.

**Basis — kenapa SigLIP (sigmoid), BUKAN CLIP (softmax):**

Zhai et al. (2023) — paper SigLIP — *"the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization."* Artinya tiap pasangan gambar-teks dinilai **independen** → skor tiap emosi berdiri sendiri 0–1 → **multi-label**. Sebaliknya CLIP memakai **softmax** yang menormalisasi antar-kandidat (jumlah = 1) → memaksa **satu label dominan** (single-label).

Tugas ini memang multi-label: Li & Deng (2019, RAF-ML) — *"facial expressions in human daily life are in multiple or co-occurring mental states"*; Liu et al. (2022, MAFW) menganotasi tiap klip ke *"one or more of the 11 widely-used emotions"*; DAiSEE (2016) memberi 4 state berlabel independen; D'Mello (2012) menunjukkan Confusion+Engagement co-occur. Jadi SigLIP dipilih **karena** multi-label, bukan kebetulan. (Lihat ACADEMIC_BASIS §10.)

Catatan: Zhang & Fu (2025) menjustifikasi *paradigma zero-shot + multi-frame* untuk anotasi emosi, tetapi tugasnya single-label 7-kelas — jadi bukan sumber untuk klaim multi-label/SigLIP spesifik.

Prompt deskriptif (bukan satu kata "boredom") memungkinkan model menggunakan pengetahuan visual-linguistik yang lebih kaya. DAiSEE (Gupta et al., 2016) mendefinisikan emosi dalam 4 level (very low/low/high/very high):
> "Each of the affective states is defined at four levels: (1) very low (2) low (3) high and (4) very high." *(§3.2)*

Prompt SigLIP dirancang untuk menangkap level "high/very high" dari setiap emosi, sesuai target pelabelan.

---

## 13. Mengapa `empirical_bias = 3.5` untuk SigLIP?

**Kode:** `inference.py` — logit SigLIP di-offset sebelum sigmoid

**Alasan teknis:** SigLIP zero-shot logit untuk deskripsi spesifik sering bernilai sangat negatif (-3 sampai -6). Tanpa bias, `sigmoid(-5) ≈ 0.007` — terlalu kecil untuk hybrid scoring yang bermakna. Bias +3.5 menggeser distribusi ke rentang 0.2–0.9 tanpa mengubah **urutan relatif** antar prompt. Ini bukan manipulasi hasil, melainkan kalibrasi skala output agar kompatibel dengan landmark scores (yang sudah berada di 0–1).

---

## 14. Bisakah Blendshape MediaPipe Dijadikan Action Unit FACS? (parameter ukur = paper)

> **ARSITEKTUR SAAT INI: MediaPipe-only.** Semua AU (AU1/AU2/AU4/AU7/AU12/AU14/AU25/AU26/AU43) dihitung dari blendshape MediaPipe via `core/action_units.py` dengan normalisasi baseline-relative dan kalibrasi per-orang (`person_neutral`). Tidak ada py-feat, tidak ada subprocess.

**Jawaban singkat: Bisa, sebagai APROKSIMASI yang dikalibrasi — dengan dua langkah wajib.** Inilah inti agar "parameter ukur sama dengan paper". Lihat `core/action_units.py`.

Paper (Craig 2008, Grafsgaard) mengukur emosi lewat **intensitas AU FACS** yang dikode manusia bersertifikat. MediaPipe meng-output **52 blendshape gaya ARKit** — itu **bukan** AU FACS. Tapi blendshape MediaPipe **dapat dipetakan** ke AU dengan 2 langkah:

### Langkah 1 — Pemetaan nama (ARKit ↔ FACS)

| AU FACS | Deskripsi | Blendshape MediaPipe |
|---|---|---|
| AU1 | Inner Brow Raiser | `browInnerUp` |
| AU2 | Outer Brow Raiser | `mean(browOuterUpLeft, browOuterUpRight)` |
| AU4 | Brow Lowerer | `mean(browDownLeft, browDownRight)` |
| AU7 | Lid Tightener | `mean(eyeSquintLeft, eyeSquintRight)` |
| AU12 | Lip Corner Puller | `max(mouthSmileLeft, mouthSmileRight)` |
| AU14 | Dimpler | `mean(mouthDimpleLeft, mouthDimpleRight)` |
| AU25 | Lips Part | `mouthOpen` (Namba 2024: thinking face Component 2) |
| AU26 | Jaw Drop | `jawOpen` (Namba 2024: thinking face Component 2) |
| AU43 | Eyes Closed | `mean(eyeBlinkLeft, eyeBlinkRight)` |

### Langkah 2 — Normalisasi baseline (KRITIS, sebelumnya tidak ada)

Intensitas AU FACS = seberapa jauh otot bergerak **DARI NETRAL**. Masalahnya, pada 21.204 frame nyata dataset ini, blendshape "diam" MediaPipe **tidak** bernilai 0:

| AU | Blendshape | Median (netral) | p90 | p99 |
|---|---|---|---|---|
| AU1 | browInnerUp | **0.46** | 0.89 | 0.98 |
| AU2 | browOuterUp | **0.47** | 0.87 | 0.96 |
| AU4 | browDown | **0.001** | 0.029 | 0.18 |
| AU7 | eyeSquint | **0.31** | 0.50 | 0.60 |
| AU43 | eyeBlink | **0.11** | 0.53 | 0.69 |

Pakai nilai mentah = **salah dua arah**: AU1/AU2 (frustration) seolah selalu aktif (~0.46 di wajah netral → over-fire), sedangkan AU4 (confusion) terkubur di rentang sempit (≤0.18). Inilah akar bias **Frustration ≫ Confusion**.

**Solusi:** anchor tiap AU ke (neutral, active) lalu normalisasi:

```
intensity = clamp( (raw - neutral) / (active - neutral), 0, 1 )
```

- `neutral` ≈ median populasi (otot diam → intensitas 0)
- `active`  ≈ p90–p99 (AU aktif penuh → intensitas 1)
- `browDown` yang rentangnya sempit (0–0.18) **di-stretch** ke 0–1 sehingga AU4 akhirnya bisa terdeteksi.

Anchor disimpan di `rules.py → "action_units"` dan bisa diatur lewat Rules panel.

### Hasil validasi (21.204 frame, sebelum vs sesudah)

| Sinyal | Sebelum (raw) | Sesudah (AU baseline-normalized) |
|---|---|---|
| Frustration land-score (median) | 0.343 (over-fire di wajah netral) | **0.150** |
| Confusion land-score (p90) | 0.149 (nyaris mati) | **0.732** (AU4 stretch + AU7 standalone) |

### Batas kejujuran

Ini tetap **aproksimasi**, bukan FACS coding manusia bersertifikat:
- Korespondensi blendshape↔AU tidak 1:1 sempurna.
- Skala dikalibrasi empiris ke dataset ini, bukan dari anchor FACS A–E.
- AU4 (browDown) **lemah di MediaPipe** (median 0.001). Dikompensasi: stretch agresif (`AU4_active=0.05`) + noseSneer co-occur booster (×0.3) + per-person neutral (`person_neutral` Bosch 2023).

Tapi setelah dua langkah ini, **"parameter ukur" sistem adalah intensitas AU FACS bernama (AU1/AU2/AU4/AU7/AU12/AU43) — sama seperti paper** — bukan lagi blendshape mentah.

---

## Ringkasan: Kode vs. Paper

| Sinyal | Paper Basis | Coverage | Implementasi |
|---|---|---|---|
| **MediaPipe blendshape → AU FACS (semua emosi)** | Craig 2008 + ARKit↔FACS mapping | stretch agresif + per-person calib | `core/action_units.py`, anchor `rules.py["action_units"]` |
| AU1 (inner) + AU2 (outer) Frustration | Craig 2008 Table 2 | 100% | `au["AU1"]·au["AU2"]` geometric mean + `brow_raise_direct_w=0.85` |
| AU4 (brow lowerer) + AU14 (dimpler) Frustration **PRIMER** | Grafsgaard 2013 | positif korelasi | `au["AU4"]/au["AU14"] * face_weight=0.65` (dinaikkan) |
| AU4 (brow lowerer) Confusion | Craig 2008 Table 2 + Grafsgaard 2011 | 95% | `au["AU4"]` (browDown lemah di MediaPipe → andalkan AU7+SigLIP) |
| AU7 (lid tightener) Confusion — **standalone** | Craig 2008 Table 2 | 78% | `au["AU7"] * au7_alone_w=0.78` (bukan co-occurrence saja) |
| AU4+AU7 co-occurrence Confusion | Craig 2008 Table 2 | 73% | `au4_au7_co` geometric mean + `au4_au7_co_w=0.50` |
| AU12 (questioning smile) Confusion gate | Craig 2008 prosa (sekunder, **bukan 95%**) | lemah/lintas-emosi | `smile_conf_gate_floor=0.30` (gate, bukan sinyal positif) |
| AU43 (eye closure) Boredom | Craig 2008 Table 2 | 40% secondary | `blink_direct_w=0.45`, independen dari gaze gate |
| Gaze deviation Boredom/Engagement | Whitehill 2014 §2.2 | level 1: "looking away" | `gaze_dev_bore`, `gaze_dev_eng` |
| Eye openness Engagement | Whitehill 2014 §2.2 | level 1-2 descriptions | `blink_heavy_th=0.50`, `eye_wide_boost=0.20` |
| Boredom ↔ Engagement complementary | D'Mello 2012 + **DAiSEE 2016 §5** ("when engagement is low, boredom is generally high and vice-versa") | empiris | `bore_eng_suppress=0.40` |
| Frus→Bore / Conf→Bore suppress | — (tak berdasar / "at chance") | DIHAPUS | — |
| Conf+Eng co-exist (productive) | D'Mello 2012 Hyp. 2 | signifikan | dibiarkan co-occur (suppress DIHAPUS) |
| Single-label dominance / strict-rules | "spec" (lawan multi-label) | DIHAPUS | — |
| Confusion 35/65, Frustration 30/70, Boredom 25/75 — landmark/AU primer + SigLIP cross-check | AU diskrit (Craig) via MediaPipe baseline-normalized; SigLIP jaring pengaman | - | `siglip_w=[.25,.50,.35,.30]` |
| Engagement hybrid 50/50 — SigLIP **tertinggi** (holistik) | Whitehill 2014: tak ada AU dominan, "static pixels" | r=0.85 | `siglip_w[1]=0.50, land_w[1]=0.50` |
| 4 emosi (bukan 6 basic) | DAiSEE 2016, D'Mello 2009 | - | Label 0=Bore, 1=Eng, 2=Conf, 3=Frus |
| HoF (any hand) → cue KUAT Confusion | Behera 2020 + D'Mello 2012 + Mahmoud 2016 | kuantitatif | `sig_hand_conf = max(hand_one,hand_two) * hand_conf_w(0.78)` (dinaikkan) |
| 2-tangan → cue pendukung Frustration | Grafsgaard 2013b (2-tangan ↔ self-efficacy rendah) | signifikan | `sig_hand_frus = hand_two * hand_frus_w(0.40)` |
| Gaze gate Engagement final | Whitehill 2014 ("looking away from computer" = not engaged) + GazeTutor 2012 | - | `eng_gaze_gate_*` di inference.py hybrid; SigLIP tidak tahu arah pandang |
| Chin-resting boredom: tidak dipakai | Tidak divalidasi utk boredom (Craig hanya AU43) | - | tidak diimplementasikan |
| Gaze konstruk (away→bored/disengaged) | GazeTutor 2012 + Whitehill 2014 | - | `gaze_dev_bore`, `gaze_dev_eng` |
| Gaze metode (head pose dari webcam) | Sümer 2021 | - | `yaw/pitch/roll`, MediaPipe iris |
| AU4 + AU14 Frustration sekunder | Grafsgaard 2013 | korelasi positif | `face_secondary = max(au["AU4"], au["AU14"])` |

---

## 15. Limitasi: Apa yang Berbasis Paper vs Engineering (jujur untuk skripsi)

Setiap **sinyal/parameter ukur** punya landasan paper (lihat tabel di atas). Yang **TIDAK** dari paper, dan kenapa itu memang harus begitu:

**A. Kalibrasi angka — wajib engineering (paper tak memberi angka untuk MediaPipe/SigLIP):**
- Threshold deteksi per-label, anchor `neutral/active` tiap AU (`action_units`), bobot hybrid `siglip_w/land_w`, `empirical_bias`, semua dead-zone/range/`scale_h`, `blend_a/blend_b`. → dikalibrasi empiris ke distribusi dataset (mis. 21.204 frame), bukan tebakan, tapi tetap bukan dari paper.

**B. Operasionalisasi/interpretasi — arah dari paper, penerapan = engineering:**
- **Cross-suppression: SUDAH DIHAPUS sebagian** (tak berdasar). Yang dihapus: **Conf→Eng** (D'Mello bilang conf→eng *significant/productive* = co-occur, jadi menekan = terbalik), **Conf→Bore** (D'Mello "at chance" = tak ada hubungan), **Frus→Bore** (transisi temporal, bukan per-frame). Juga **dual_label_gap (dominance)** & **strict_rules_bias** (keduanya "spec", lawan multi-label DAiSEE) dihapus. Yang **DIPERTAHANKAN**: hanya **Bore↔Eng** (DAiSEE "complementary" + near-mutually-exclusive — berdasar) + mutual-exclusion Conf⊕Frus (penyederhanaan, lihat §11).
- **Frustration single-brow** (`sig_bou_alone`, bobot 0.70): Craig Table 2 memberi AU1 sendiri 100% & AU2 sendiri 100% (jadi *valid* per paper), tapi bobot parsial 0.70 = kalibrasi.
- **Gaze dari webcam**: konstruk (GazeTutor/Whitehill) pakai eye-tracker; metode webcam (Sümer) lebih kasar → head pose dominan, iris refinement (lihat §11).

**C. Komponen di luar paper rujukan (teknologi modern, dibenarkan secara konsep):**
- **Hybrid SigLIP2** (VLM zero-shot): paper rujukan pakai FACS/AU, bukan VLM. Dibenarkan konsep "holistic facial appearance" Whitehill (2014).
- **AU = MediaPipe FaceLandmarker blendshape** dengan normalisasi baseline-relative + kalibrasi agresif AU4 + per-person neutral (Bosch 2023). Ini aproksimasi FACS, bukan py-feat, tapi parameter ukur (AU1/2/4/7/14/25/26/43) tetap sama dengan yang divalidasi Craig/Grafsgaard.

> **Kesimpulan jujur:** *Apa* yang diukur = berbasis paper. *Di angka berapa* & *cara menggabung* = engineering. MediaPipe-only adalah trade-off: sedikit presisi AU4 berkurang vs. kecepatan dan kesederhanaan proses yang jauh lebih baik.

---

## 16. Mengapa MediaPipe-Only (Tanpa py-feat)?

**Arsitektur branch ini:** `core/action_units.py` `compute_action_units(blendshapes, cfg, person_neutral)` — satu fungsi sinkron, satu proses, nol subprocess.

**Masalah py-feat (yang diatasi):**
- py-feat (numpy<2) & SigLIP (numpy 2) konflik dependency → subprocess terpisah, latency, fragile.
- py-feat berat (~2GB), instalasi sulit, sering gagal di environment baru.
- Akurasi AU4 py-feat memang lebih tinggi (median 0.31) vs MediaPipe (median 0.001), tapi overhead-nya tidak sebanding untuk use-case labeling offline.

**Kompensasi MediaPipe-only untuk AU4 lemah:**
- Stretch agresif: `AU4_neutral=0.001, AU4_active=0.05` → deviasi kecil terdeteksi.
- noseSneer co-occur booster ×0.3 → sinyal AU4 implisit diperkuat.
- Per-person neutral (`person_neutral`, Bosch 2023): browDown baseline tiap orang berbeda → kalibrasi per-individu menghilangkan bias struktural.
- `brow_raise_direct_w=0.85` (dinaikkan): kompensasi tidak adanya py-feat, menguatkan AU1+AU2 primer Craig 2008.
- `face_weight=0.65` (dinaikkan): kompensasi AU4 MediaPipe, menguatkan Grafsgaard 2013 AU4/AU14.

**Tambahan AU25/AU26 (Namba 2024):** mouthOpen (≈AU25) dan jawOpen (≈AU26) kini tersedia langsung dari MediaPipe — sebelumnya hanya via py-feat. Keduanya dipakai sebagai cue Confusion kuat.

**Basis paper tetap tidak berubah:** AU yang diukur = AU FACS dari Craig 2008 + Grafsgaard 2011/2013 (tidak berubah). MediaPipe mengukurnya via blendshape→AU mapping yang dinormalisasi — aproksimasi yang jauh lebih cepat dan stabil.

---

## 17. Audit Kesesuaian Konteks Paper (jangan force-fit)

Tiap paper diperiksa: apakah konteks aslinya **sama** dengan tugas ini (e-learning, 4 affect belajar, per-frame, multi-label) atau **beda** (→ hanya dipakai untuk prinsip yang generalisasi, jangan diklaim spesifik).

| Paper | Konteks asli | Status untuk tugas ini |
|---|---|---|
| Craig 2008, Grafsgaard 11/13, D'Mello 2012, **DAiSEE 2016** | Learning affect, 4 emosi (DAiSEE persis e-learning) | ✅ **konteks sama** — boleh dipakai spesifik |
| Whitehill 2014, Sümer 2021 | Engagement (edukasi), webcam | ✅ sama untuk engagement/gaze; "static pixels" khusus engagement (ekstrapolasi ke 4 emosi disebut) |
| GazeTutor 2012 | Boredom via **eye-tracker** | ⚠️ konstruk sama, **alat beda** (webcam) → dipakai untuk konsep, bukan metode |
| RAF-ML 2019, MAFW 2022 | **Emosi basic/compound** (bukan learning) | ⚠️ hanya bukti **PRINSIP** multi-label; bukti spesifik learning = DAiSEE |
| SigLIP/Zhai 2023 | Pretraining umum (ImageNet) | ⚠️ properti **arsitektur** (sigmoid independen), bukan validasi emosi |
| Zhang 2025 | Zero-shot anotasi 7 emosi basic, single-label, LMM | ⚠️ hanya **paradigma** zero-shot+multi-frame; bukan multi-label/SigLIP |

**Keputusan berbasis audit ini:**

- **#5 (DAiSEE "both bore+eng low → conf/frus high") TIDAK di-force.** Konteks beda: DAiSEE **mengecualikan 'neutral'** saat anotasi (*"...to avoid the 'neutral' state..."*), data ini bisa punya frame netral → boost akan salah-tembak. Mekanismenya ada di kode tapi **DEFAULT NONAKTIF** (`bore_eng_low_boost=0.0`) + caveat. Aktifkan hanya bila datasetmu juga tanpa netral.
- **#4 (AU12/AU14 lintas-emosi, Craig p.785).** Craig mencatat AU12 & AU14 muncul di **both** confusion & frustration → **non-diskriminatif**. Karena itu kode SENGAJA memakainya **konservatif** (AU12 = gate ber-floor; AU14 = sekunder lemah), **bukan** sinyal pembeda kuat. Menambahkannya ke kedua emosi malah mengaburkan diskriminasi → tidak dilakukan. Ini operasionalisasi yang setia pada temuan Craig.
- **Frame-level & struktur label:** lihat ACADEMIC_BASIS §11. Frame-level dibenarkan Whitehill ("static pixels") + RAF-ML (image-level) + Zhang (anotasi key-frame). Struktur label = **4 label biner independen** (multi-label binary) → berdasar DAiSEE (4 state independen) + SigLIP sigmoid (skor per emosi mandiri). "Maks 2 aktif" = konsekuensi 2 pasangan eksklusif: Bore⊕Eng (kuat: DAiSEE/D'Mello) & Conf⊕Frus (lemah: paper hanya tunjukkan transisi → penyederhanaan desain). Kirim ke LLM hilir = alasan praktis, bukan dari paper.
