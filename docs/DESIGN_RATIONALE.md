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
"siglip_w": [0.30, 0.40, 0.45, 0.30],  # [Bore, Eng, Conf, Frus]
"land_w":   [0.70, 0.60, 0.55, 0.70],
```

### Engagement — SigLIP 40%, Landmark 60%

**Alasan:** Whitehill et al. (2014) menunjukkan engagement dapat dideteksi dari penampilan holistik wajah (static pixels), bukan hanya AU tertentu. SigLIP mendekati cara manusia menilai engagement dari foto.

> "Furthermore, we found that engagement labels of 10-second video clips can be reliably predicted from the average labels of their constituent frames (Pearson r = 0.85), suggesting that static expressions contain the bulk of the information about engagement used by observers." *(abstract)*

Craig et al. (2008) tidak menemukan AU primer untuk Engagement — berbeda dengan Confusion dan Frustration yang punya AU spesifik dengan coverage tinggi. Oleh karena itu, bobot SigLIP lebih tinggi (0.40 vs. 0.30 emosi lain) karena tidak ada sinyal AU tunggal yang dominan.

### Frustration — SigLIP 30%, Landmark 70%

**Alasan:** Craig et al. (2008) menemukan AU yang sangat spesifik untuk Frustration dengan coverage 100%:
> "It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together." *(p. 784)*

Karena ada sinyal AU yang sangat kuat dan empirically validated (coverage 100%), landmark scoring lebih diandalkan → bobot landmark lebih tinggi (0.70).

### Confusion — SigLIP 45%, Landmark 55%

**Alasan:** Confusion memiliki AU yang divalidasi (AU4+AU7 co-occur 73%, AU4 coverage 95%), tetapi ada ambiguitas lebih tinggi — AU yang sama (browDown, squint) juga muncul di Frustration. SigLIP membantu memisahkan konteks visual yang sulit dipisahkan oleh AU saja. Bobot SigLIP tertinggi (0.45) di antara semua emosi.

### Boredom — SigLIP 30%, Landmark 70%

**Alasan:** Craig et al. (2008) memvalidasi AU43 (eye closure) sebagai sinyal Boredom. Selain itu, gaze deviation (head pose + iris) adalah sinyal kuat untuk ketidakfokusan. Kedua sinyal ini dapat diukur langsung oleh landmark detector → bobot landmark tinggi (0.70).

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

**Kenapa `brow_raise_direct_w = 0.65` lebih tinggi dari `face_weight = 0.45`?**

Karena AU1+AU2 adalah sinyal primer dengan coverage 100% (Craig 2008), sedangkan sinyal legacy (nose sneer, cheek squint) tidak muncul di Craig Table 2 sebagai sinyal primer frustration. Tier berbeda → bobot berbeda.

---

## 4. Mengapa `hand_weight = 0.35` (Bukan 0.65) untuk Frustration?

**Kode:** `rules.py → frustration → hand_weight: 0.35`

**Alasan:** Tidak ada paper dalam referensi yang memvalidasi posisi tangan (hand-to-face gesture) sebagai sinyal frustration.

Craig et al. (2008) hanya menggunakan FACS — gerakan otot wajah, bukan tubuh atau tangan. Tidak ada deteksi tangan dalam studi mereka.

D'Mello, Craig, Fike & Graesser (2009) menyebut:
> "The new versions of AutoTutor detect learners' boredom, confusion, and frustration by monitoring conversational cues, gross body language, and facial features." *(abstract)*

"Gross body language" di sini merujuk pada data **seat pressure pad** (lean/fidget dari sensor tekanan kursi), bukan posisi tangan yang ditangkap kamera.

**Kesimpulan:** Hand signals adalah heuristik praktis tanpa dasar paper langsung. Dipertahankan sebagai supplementary (0.35), sinyal AU wajah tetap primer (0.65+ dari face contribution).

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

**Kenapa `au4_au7_co_w = 0.50`?** Co-occurrence coverage 73% (tidak 100% seperti frustration). Maka bobotnya lebih rendah (0.50) dibanding brow_raise_direct_w Frustration (0.65). AU4 tetap memiliki bobot sendiri (95% coverage), sehingga co-occurrence adalah sinyal tambahan di atas sinyal AU4 individual.

---

## 6. Mengapa AU12 (Smile) adalah Gate Floor untuk Confusion, Bukan Suppressor Penuh?

**Kode:** `smile_gate = clamp(1.0 - smile/gate_th, floor=0.30, max=1.0)`

**Alasan dari Craig et al. (2008) Table 2:**

| AU | Deskripsi | Coverage |
|---|---|---|
| AU12* | Lip corner puller (questioning smile) | 95% |

*Note: Of secondary importance*

> "Action units [...] 12 [...]. Notable exceptions are AUs 12 and 14 that occur during expressions of both confusion and frustration." *(p. 785)*

**Kenapa `smile_conf_gate_th = 0.35` (bukan 0.20 sebelumnya) dan `floor = 0.30`?**

AU12 muncul pada 95% episode Confusion — ini adalah "questioning smile" (senyum ketidakpastian), bukan senyum kebahagiaan. Jika smile men-zero-kan confusion sepenuhnya, 95% data confusion Craig (2008) akan di-suppress secara salah.

`gate_th = 0.35` artinya senyum lemah/sedang tidak menekan confusion. `floor = 0.30` artinya bahkan senyum penuh pun tidak bisa membawa confusion di bawah 30% — sesuai kenyataan bahwa senyum co-occurs dengan confusion di hampir semua kasus.

---

## 7. Mengapa BrowInnerUp Suppresses Confusion?

**Kode:** `brow_in_v = brow_in_v * clamp(1.0 - bou_check * biu_au1_suppress, 0, 1)`

**Alasan:** Craig et al. (2008) menunjukkan bahwa inner brow raise bersama outer brow raise adalah sinyal **Frustration**, bukan Confusion:

> "It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together." *(p. 784)*

Jika browInnerUp aktif **bersama** browOuterUp yang tinggi, pola itu adalah pola frustration (AU1+AU2 co-occurrence). Dalam konteks itu, browInnerUp bukan sinyal confusion — ia harus di-suppress dari confusion scoring. `biu_au1_suppress = 0.80` memastikan 80% dari browInnerUp contribution ke confusion dihilangkan ketika pola frustration terdeteksi.

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

## 9. Mengapa Confusion dan Engagement Boleh Co-exist?

**Kode:** `conf_eng_suppress_th = 0.50`, `conf_eng_suppress = 0.35`

**Alasan dari D'Mello & Graesser (2012):**

> "The second hypothesis is the productive confusion hypothesis (Hypothesis 2). According to this hypothesis, cognitive disequilibrium, impasses, and confusion provide learners with an opportunity to think, deliberate, and problem solve." *(p. 149)*

> "The Confusion → Engagement/Flow and Confusion → Frustration transitions were also significant, while the Confusion → Boredom transition occurred at chance levels, thereby confirming the productive confusion and hopeless confusion hypotheses." *(p. 153)*

Model transisi D'Mello & Graesser secara eksplisit menunjukkan bahwa Confusion **bertransisi ke** Engagement (productive confusion), bukan memblokir Engagement. Seorang murid bisa bingung tapi masih aktif mencoba memahami materi — ini adalah state yang valid dan produktif.

Sebelumnya `conf_eng_suppress = 0.55` terlalu agresif — confusion hampir selalu membunuh engagement. Sekarang `conf_eng_suppress = 0.35` dan threshold dinaikkan ke 0.50, sehingga hanya ketika confusion sangat tinggi barulah engagement mulai ditekan.

---

## 10. Mengapa `chin_bore_max = 0.35` (Bukan 0.70) untuk Boredom?

**Kode:** `rules.py → boredom → chin_bore_max: 0.35`

**Alasan:** Tidak ada paper yang memvalidasi chin-resting sebagai sinyal boredom melalui kamera.

Craig et al. (2008) hanya memvalidasi AU43 (eye closure) untuk Boredom. D'Mello & Graesser (2010) memvalidasi "gross body language" tapi melalui **seat pressure pad** (Tekscan BPMS), bukan posisi tangan.

Menopang dagu (hand_chin) adalah heuristik praktis — dipertahankan karena secara intuisi sinyal postur pasif, tetapi dengan bobot sangat rendah (max 0.35 vs. sebelumnya 0.70) karena tidak ada validasi empiris langsung.

---

## 11. Mengapa Gaze Deviation untuk Boredom dan Engagement?

**Kode:** `gaze_dev_bore`, `gaze_dev_eng` — kombinasi yaw + iris_x + iris_y

**Basis:** D'Mello, Craig, Fike & Graesser (2009):
> "The new versions of AutoTutor detect learners' boredom, confusion, and frustration by monitoring conversational cues, gross body language, and facial features." *(abstract)*

Gaze deviation (tatapan meninggalkan layar) merupakan sinyal non-facial yang diimplementasikan melalui head pose + iris tracking dari MediaPipe — proxy dari "attentiveness" yang disebutkan dalam literatur engagement.

Whitehill et al. (2014) engagement scale secara eksplisit menyebut gaze:
> "1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task, eyes completely closed." *(§2.2)*

**Kenapa gaze_dev_bore BERBEDA dari gaze_dev_eng?**

- `gaze_dev_bore` menyertakan roll dan hanya komponen gaze ke atas (bukan ke bawah). Murid yang nunduk baca masih engaged, bukan bosan.
- `gaze_dev_eng` tidak menyertakan roll (natural head tilt ≠ disengagement), tidak menyertakan gaze ke bawah (nunduk = baca/ngetik = engaged).

---

## 12. Mengapa SigLIP Menggunakan Prompt Bahasa Inggris Deskriptif?

**Kode:** SigLIP diberi prompt teks per emosi, bukan label satu kata.

**Basis:** SigLIP2 (Tschannen et al., 2025) adalah vision-language model yang menggunakan **sigmoid loss** — scoring tiap pasangan gambar-teks secara independen, bukan softmax. Artinya skor setiap emosi bersifat independen satu sama lain, sesuai kebutuhan multi-label (satu video bisa Confusion + Engagement sekaligus).

Prompt deskriptif (bukan satu kata "boredom") memungkinkan model menggunakan pengetahuan visual-linguistik yang lebih kaya. DAiSEE (Gupta et al., 2016) mendefinisikan emosi dalam 4 level (very low/low/high/very high):
> "Each of the affective states is defined at four levels: (1) very low (2) low (3) high and (4) very high." *(§3.2)*

Prompt SigLIP dirancang untuk menangkap level "high/very high" dari setiap emosi, sesuai target pelabelan.

---

## 13. Mengapa `empirical_bias = 3.5` untuk SigLIP?

**Kode:** `inference.py` — logit SigLIP di-offset sebelum sigmoid

**Alasan teknis:** SigLIP zero-shot logit untuk deskripsi spesifik sering bernilai sangat negatif (-3 sampai -6). Tanpa bias, `sigmoid(-5) ≈ 0.007` — terlalu kecil untuk hybrid scoring yang bermakna. Bias +3.5 menggeser distribusi ke rentang 0.2–0.9 tanpa mengubah **urutan relatif** antar prompt. Ini bukan manipulasi hasil, melainkan kalibrasi skala output agar kompatibel dengan landmark scores (yang sudah berada di 0–1).

---

## Ringkasan: Kode vs. Paper

| Fitur | Paper Basis | Implementasi |
|---|---|---|
| AU1+AU2 Frustration (100%) | Craig et al. (2008) Table 2 | `brow_raise_co` geometric mean + `brow_raise_direct_w=0.65` |
| AU4+AU7 Confusion (73%) | Craig et al. (2008) Table 2 | `au4_au7_co` + `au4_au7_co_w=0.50` |
| AU12 Confusion (95%) gate floor | Craig et al. (2008) Table 2 | `smile_conf_gate_floor=0.30` |
| AU43 Boredom (40%) independent | Craig et al. (2008) Table 2 | `blink_direct_w=0.45`, no gaze gate |
| BrowInnerUp suppress confusion | Craig et al. (2008): AU2=frustration | `biu_au1_suppress=0.80` |
| Engagement holistic appearance | Whitehill et al. (2014) | `siglip_w=0.40` untuk Engagement |
| Conf+Eng co-exist (productive) | D'Mello & Graesser (2012) | `conf_eng_suppress=0.35`, `th=0.50` |
| Hand signals bukan primer | Tidak ada paper | `hand_weight=0.35`, `chin_bore_max=0.35` |
| 4 emosi (bukan 6 basic) | DAiSEE (2016), D'Mello et al. (2009) | Label 0=Bore, 1=Eng, 2=Conf, 3=Frus |
