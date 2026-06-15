# Dasar Akademis Sistem Pelabelan Emosi

Dokumen ini menjabarkan **landasan akademis setiap keputusan metode** dalam aplikasi ini. Setiap bagian mengikuti alur:
1. **Konteks** — mengapa konsep ini dibutuhkan (Bahasa Indonesia)
2. **Verbatim** — kutipan kata per kata dari paper asli (Bahasa Inggris, tidak diubah)
3. **Penjelasan** — arti verbatim, istilah teknis (termasuk nama AU), dan hubungannya ke kode/sistem (Bahasa Indonesia)

> **Catatan kutipan:** Nomor referensi dalam tanda kurung siku seperti `[17]` dan penanda internal paper seperti `(Link 3)` dihilangkan mengikuti praktik standar. Semua kata dalam blok kutipan adalah verbatim dari sumber asli.

> **✅ Status verifikasi (terakhir dicek):** Seluruh blok kutipan di dokumen ini **dicek kata-per-kata** terhadap 37 PDF paper sumber (`2-anotasi-data/paper/`, ekstraksi `pdftotext` + pencocokan substring ternormalisasi). **Semua kutipan akurat.** Perbedaan yang muncul saat pengecekan otomatis seluruhnya artefak ekstraksi PDF (soft-hyphen pemenggalan baris, penanda sitasi inline yang sengaja dihilangkan, header halaman nyelip) — bukan salah kutip. **Bartlett 1999** adalah PDF hasil pindai (tanpa lapisan teks); ia di-**OCR** (EasyOCR, 300 dpi, 11 hal → ~9.700 kata) dan kutipan FACS-nya **TERKONFIRMASI cocok**. Jadi **tidak ada kutipan yang tersisa tanpa verifikasi.** Pemeriksaan ulang menyeluruh (semua 37 teks diekstrak ke satu folder + cek tiap span) memperbaiki beberapa slip kecil: *"looking away from **the** computer"* → tanpa "the"; *"uncertainty"* → *"uncertain about what to do next"*; ejaan *"nonsignificant"* sesuai sumber; serta merapikan glosa/parafrase yang tadinya diberi tanda kutip (mis. *forced-effort theories*).

---

## Ringkasan Kekuatan Bukti per Sinyal

Tiap sinyal punya kekuatan bukti berbeda. **KUAT** = banyak studi / coverage tinggi / langsung; **SEDANG** = satu studi atau rantai tak-langsung tapi didukung; **LEMAH** = sampel kecil / rantai panjang / sulit dilihat / keterbatasan detektor.

| Emosi | Sinyal | Kekuatan | Alasan singkat |
|---|---|---|---|
| Confusion | Alis turun **AU4** | **KUAT** | Craig 95% + Grafsgaard 2011 + ConfusionBench 2026 (3 studi independen) |
| Confusion | Kelopak menyipit **AU7** / **AU4+AU7** | **KUAT** | Craig 78%/73%; ConfusionBench: AU4+AU7 *"most reliable"* |
| Confusion | Mulut terbuka **AU25+AU26** | SEDANG | Namba 2024 (1 studi, konteks *thinking face*, ambigu di praktik) |
| Confusion | Tangan dagu/telunjuk (1 tangan) | LEMAH | Rantai Behera+D'Mello; Mahmoud N kecil; detektor buta posisi |
| Engagement | Menatap layar + mata terbuka (holistik) | **KUAT** | Whitehill κ=0.96; info di "static pixels" |
| Engagement | Nunduk = on-task | SEDANG | Sümer 2021 (satu studi kelas) |
| Boredom | Mata berat/menutup **AU43** | SEDANG | Craig: satu-satunya AU signifikan, **tapi coverage 40%, 1 studi** |
| Boredom | Gaze ke samping/atas | SEDANG | Konstruk 3 paper; **gaze webcam kasar** |
| Frustration | Alis naik **AU1+AU2** | **KUAT\*** | Craig 100% — **\*tapi 1 studi**; Grafsgaard 2013 justru soroti AU4 turun |
| Frustration | Alis turun sekunder **AU4** | SEDANG | Grafsgaard 2013 (auto/CERT, korelasi positif) |
| Frustration | 2 tangan menekan dahi / tutup wajah | SEDANG | Grafsgaard 2013b (self-efficacy rendah, "significant") |
| Frustration | Lesung pipit **AU14** | LEMAH | Halus, sulit dilihat; range MediaPipe sempit |

Detail tiap penilaian ada di section masing-masing di bawah. Versi praktis (untuk anotasi) ada di `PANDUAN_ANOTASI_MANUAL.md`.

---

## Peta Rantai Argumen — baca ini dulu

Dokumen ini **bukan kumpulan kutipan lepas**, melainkan **satu rantai argumen**. Tiap mata rantai memakai verbatim sebagai bukti, lalu menyambung ke pertanyaan yang dijawab section berikutnya (tanda **↓** = "pertanyaan lanjutan"):

0. **§0 — Blendshape langsung sebagai sinyal emosi (Chain Rule).** Turrisi 2026 (κ=0.92): blendshape MediaPipe → AU FACS (expert-validated). Aldenhoven 2026: native blendshapes → 68.3% accuracy. Chain: Craig (AU→emosi) + Turrisi (BF→AU) = blendshape langsung valid. eyeLookDown = AU64 (gaze), bukan AU7/AU43.
   **↓** *Emosi apa yang perlu dideteksi?*
1. **§1 — Kenapa 4 emosi ini?** D'Mello & DAiSEE: di belajar berdurasi panjang, hanya *boredom, engagement, confusion, frustration* yang dominan & reliable (6 emosi dasar gugur).
   **↓** *Kalau cuma 4 ini, bagaimana mengukurnya secara objektif?*
2. **§2 — Kenapa FACS/Action Unit?** Bartlett & Whitehill: mengukur otot wajah (AU) lebih reliable daripada menebak kategori emosi.
   **↓** *AU mana untuk emosi mana?*
3. **§3 — Peta AU→emosi (fondasi semua scoring).** Craig 2008 (*association rule mining*): AU1+AU2→Frustration, AU4+AU7→Confusion, AU43→Boredom.
   **↓** *Bedah tiap emosi satu per satu:*
4. **§4 Frustration** — AU1+AU2 primer; **AU4/AU14 dinaikkan ke primer** (Grafsgaard 2013); ekspresi bervariasi antar-orang → **kalibrasi frame netral per-orang** (Bosch 2023). **↓**
5. **§5 Confusion** — AU4+AU7 (AU7 bisa *standalone*); **boleh co-exist dengan Engagement** (*productive confusion*, D'Mello). **↓**
6. **§6 Boredom** — AU43 (mata menutup) + *disengagement*. **↓**
7. **§7 Engagement** — penampilan holistik + atensi (gaze).
   **↓** *Wajah saja tidak cukup — tambah konteks:*
8. **§8 Multimodal** → **§8.5 Gaze & head-pose** (arah atensi: nunduk = lihat keyboard/menulis = engaged; noleh atas/samping = bored) → **§8.6 Tangan** (*hand-over-face*: **dagu/telunjuk → Confusion**, **dahi ditekan / 2-tangan → Frustration**).
   **↓** *Bagaimana bentuk output & datanya:*
9. **§9 DAiSEE** (dataset acuan) → **§10 Multi-label biner** (tiap emosi dinilai independen; karena itu pakai **SigLIP sigmoid**, bukan softmax single-label) → **§11 Frame-level labeling** (label statis ini dipakai melatih model realtime hilir).

> Jadi tiap verbatim adalah **bukti untuk satu mata rantai keputusan desain** — bukan kutipan yang berdiri sendiri. Saat membaca tiap section, perhatikan baris **Konteks** (menyambung dari section sebelumnya) dan **Penjelasan** (menerjemahkan kutipan Inggris + menautkannya ke kode).

---

## §0. Blendshape Features sebagai Sinyal Emosi Langsung (Chain Rule)

**Konteks:** Kode sumber menyebut `core/blendshape_features.py` — bukan `action_units.py`. Ini mencerminkan bahwa sistem menggunakan **blendshape MediaPipe secara langsung** sebagai sinyal, bukan sebagai perantara AU semu. Bagian ini menjelaskan chain rule yang menghubungkan blendshape → emosi belajar melalui dua paper.

**Chain Rule dua mata:**

```
Craig et al. (2008)          Turrisi et al. (2026)
AU1+AU2 → Frustration   ←   browInnerUp / browOuterUp    (κ=1.00)
AU4+AU7 → Confusion     ←   browDown + eyeSquint          (κ=0.92)
AU43    → Boredom        ←   eyeBlink                     (κ≥0.85)
AU25+AU26 → Confusion   ←   jawOpen                      (Namba 2024)
```

**Verbatim Turrisi 2026 (expert-validated BF→AU mapping):**

> "We recruited ten clinical experts in video-based behavioral coding and implemented a protocol including an independent blinded annotation phase. Inter-rater reliability was then quantified to evaluate agreement across experts. This process yielded a lookup table linking 52 MediaPipe BFs to corresponding FACS AUs."

— Turrisi et al. (2026). *Blendshape Features Meet Action Units.* Computers in Human Behavior Reports, 22. (§5)  [PDF](../../paper/07-%20Turrisi%202026%20-%20Blendshape%20Features%20Meet%20Action%20Units.pdf)

**Penjelasan:** Turrisi dkk. memvalidasi secara klinis bahwa setiap dari 52 blendshape MediaPipe berkorespondensi ke AU FACS tertentu, dengan agreement κ=0.92 dari 10 psikolog bersertifikat. Ini berarti menggunakan blendshape langsung (misal `eyeSquintLeft`) adalah ekuivalen dengan menggunakan AU7 — dengan justifikasi expert. Sistem ini mengadopsi chain: **Craig 2008 (AU→emosi)** + **Turrisi 2026 (BF→AU)** = blendshape langsung merepresentasikan sinyal emosi belajar.

---

**Verbatim Aldenhoven 2026 (native blendshapes untuk emosi, tanpa model tambahan):**

> "A cosine-similarity classifier on standardized ARKit blend shapes offers an attractive accuracy–complexity trade-off for embedded real-time emotion sensing. [...] the cosine-similarity approach achieved an overall accuracy of 68.3% and AUCs ≥ 0.84 for all classes, outperforming the mean human accuracy (58.9%)."

— Aldenhoven et al. (2026). *Real-Time Emotion Recognition Performance of Mobile Devices.* Sensors, 26, 1060. (§5 Conclusions)  [PDF](../../paper/14-%20Aldenhoven%202026%20-%20Real-Time%20Emotion%20Recognition%20Performance%20of%20Mobile%20Devices%20ARKit.pdf)

**Penjelasan:** Aldenhoven dkk. membuktikan bahwa native ARKit/MediaPipe blendshapes — tanpa model aproksimasi tambahan — sudah cukup akurat untuk deteksi emosi (68.3% vs human 58.9%). Mereka secara eksplisit menyebut "similar coefficient streams are also exposed by alternative stacks (e.g., MediaPipe Face Landmarker)", mengkonfirmasi portabilitas ke pipeline kita. Ini adalah dasar penggunaan `BLENDSHAPE_SOURCE=mediapipe` (native) bukan model aproksimasi mp_blendshapes.

---

**eyeLookDown Gating — Turrisi 2026:**

**Verbatim:**

> "eye look down left and eye look down right, which jointly codify AU64."

— Turrisi et al. (2026). *Blendshape Features Meet Action Units.* (§6 Results, mapping detail)  [PDF](../../paper/07-%20Turrisi%202026%20-%20Blendshape%20Features%20Meet%20Action%20Units.pdf)

**Penjelasan:** AU64 (*Eyes Turn Down*) adalah AU gaze direction — bukan AU7 (lid tightener dari usaha kognitif) dan bukan AU43 (eye closure dari mengantuk). Saat siswa melihat ke bawah (membaca buku/keyboard), kelopak mata sedikit menutup secara mekanik mengikuti arah pandang, menyebabkan `eyeSquint` dan `eyeBlink` naik. Tanpa gating, ini falsely trigger Confusion (AU7) dan Boredom (AU43). Sistem ini menekan AU7 dan AU43 secara proporsional ketika `eyeLookDown` tinggi — diimplementasi di `_raw_blendshape_signals()`.

---

## 1. Mengapa Empat Emosi Ini?

**Konteks:** Sistem ini perlu memilih emosi mana yang relevan untuk situasi belajar. Bukan semua emosi bisa dideteksi dengan andal saat belajar berdurasi panjang — ada penelitian yang menjawab ini secara empiris.

> "One finding is that confusion, frustration, boredom, and engagement/flow are the major affective states that students experience across diverse learning contexts, student populations, and methods to track emotions."

— D'Mello, S.K. & Graesser, A. (2012). *Dynamics of Affective States during Complex Learning.* Learning and Instruction, 22, 145–157. (p. 148)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=4)

**Penjelasan:** D'Mello & Graesser meneliti emosi yang dominan muncul saat siswa belajar materi sulit. Hasil di berbagai konteks belajar, berbagai populasi siswa, dan berbagai metode pengukuran selalu menunjukkan keempat emosi yang sama: *confusion* (bingung), *frustration* (frustrasi), *boredom* (bosan), dan *engagement/flow* (terlibat/fokus). Inilah alasan utama sistem ini hanya melabeli 4 emosi ini — bukan karena arbitrary, tapi karena terbukti paling dominan.

---

> "The affect-detection phase focused on the development of computational systems that monitor conversational cues, gross body language, and facial features to detect the presence of boredom, engagement, confusion, and frustration (delight and surprise were excluded because they are extremely rare)."

— D'Mello, S., Craig, S., Fike, K., & Graesser, A. (2009). *Responding to Learners' Cognitive-Affective States with Supportive and Shakeup Dialogues.* (p. 2)  [PDF](../../paper/23-%20DMello%202009%20-%20Responding%20to%20Learners%20Cognitive-Affective%20States.pdf#page=2)

**Penjelasan:** D'Mello 2009 secara eksplisit menyebut *delight* (senang) dan *surprise* (terkejut) dikecualikan karena sangat jarang muncul. Ini memvalidasi pilihan 4 emosi di sistem ini — emosi positif sesaat seperti *delight* tidak cukup sering untuk dijadikan target pelabelan. Frasa "facial features" juga mengkonfirmasi bahwa wajah merupakan kanal deteksi yang diakui peneliti.

---

> "our dataset consists of labels for four affective states related to user engagement, viz., engagement, frustration, confusion, and boredom. Recent work has shown that the six basic expressions: anger, disgust, fear, joy, sadness, and surprise are not reliable in prolonged learning situations, as they are prone to rapid changes."

— Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016). *DAiSEE: Towards User Engagement Recognition in the Wild.* arXiv:1609.01885. (§3.2)  [PDF](../../paper/02-%20Gupta%202016%20-%20DAiSEE%20Towards%20User%20Engagement%20Recognition%20in%20the%20Wild.pdf#page=5)

**Penjelasan:** DAiSEE adalah dataset e-learning paling relevan untuk sistem ini — konteksnya persis sama (video webcam mahasiswa belajar mandiri). Gupta dkk. membangun dataset ini dengan 4 emosi yang sama dan secara eksplisit menolak 6 emosi dasar (marah, jijik, takut, senang, sedih, terkejut) karena *tidak reliable* di situasi belajar berdurasi panjang. Ini memperkuat D'Mello — bukan hanya emosi lain *jarang*, tapi juga *tidak bisa diandalkan* untuk anotasi.

---

**Definisi operasional tiap emosi (dipakai saat menurunkan pemetaan AU di paper Craig 2008):**

> "Boredom—the state of being weary and restless through lack of interest."

> "Confusion—the failure to differentiate from an often similar or related other."

> "Frustration—making vain or ineffectual efforts, however vigorous; a deep chronic sense or state of insecurity and dissatisfaction arising from unresolved problems or unfulfilled needs."

— Craig, S.D., D'Mello, S., Witherspoon, A., & Graesser, A. (2008). *Emote aloud during learning with AutoTutor: Applying the Facial Action Coding System to cognitive–affective states during learning.* Cognition & Emotion, 22(5), 777–788. (p. 781)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=5)

**Penjelasan:** Craig 2008 memberi definisi operasional tiap emosi yang dipakai peserta saat mereka melabeli sendiri kondisi mereka (*emote aloud*). Definisi ini penting karena pemetaan AU (gerakan otot wajah) yang ditemukan Craig didasarkan pada *saat siswa melaporkan* emosi ini. Dengan kata lain: AU4 (alis turun) berkaitan dengan confusion karena saat siswa bilang "saya bingung", otot AU4-nya aktif.

**Di kode:** `LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]` di `ui/constants.py`.

> **Jembatan → 2. Mengapa Mengukur Gerakan Otot Wajah (FACS/Action Unit)?:** Empat emosi ini sudah pasti relevan. Pertanyaan berikutnya: bagaimana mengukurnya tanpa menebak-nebak "ini ekspresi apa"? Jawabannya — ukur gerakan otot wajah secara objektif (FACS).

---

## 2. Mengapa Mengukur Gerakan Otot Wajah (FACS/Action Unit)?

**Konteks:** Ada dua cara menilai ekspresi wajah — secara subjektif (orang menebak "ini marah") atau secara objektif (mengukur otot mana yang bergerak). Sistem ini memilih cara kedua karena lebih reliable dan lebih ilmiah.

> "The Facial Action Coding System (Ekman & Friesen, 1978) is an objective method for quantifying facial movement in terms of component actions."

— Bartlett, M.S., Hager, J.C., Ekman, P., & Sejnowski, T.J. (1999). *Measuring facial expressions by computer image analysis.* Psychophysiology, 36, 253–263. (abstract)  [PDF](../../paper/21-%20Bartlett%201999%20-%20Measuring%20Facial%20Expressions%20by%20Computer%20Image%20Analysis.pdf#page=1)

**Penjelasan:** FACS (Facial Action Coding System) adalah sistem standar internasional untuk mengukur ekspresi wajah. Alih-alih menebak "ekspresi ini marah", FACS menguraikannya ke **Action Unit (AU)** — unit gerakan otot spesifik yang terukur. Misalnya: AU4 = otot *corrugator supercilii* yang menarik alis ke bawah (alis mengkerut/menurun). AU tidak bergantung pada interpretasi pengamat, sehingga lebih objektif.

---

> "FACS provides an objective and comprehensive language for describing facial expressions and relating them back to what is known about their meaning from the behavioral science literature. Because it is comprehensive, FACS also allows for the discovery of new patterns related to emotional or situational states."

— Bartlett, M.S., Littlewort, G.C., Frank, M.G., Lainscsek, C., Fasel, I.R., & Movellan, J.R. (2006). *Automatic Recognition of Facial Actions in Spontaneous Expressions.* Journal of Multimedia, 1(6), 22–35. (p. 22)  [PDF](../../paper/22-%20Bartlett%202006%20-%20Automatic%20Recognition%20of%20Facial%20Actions%20in%20Spontaneous%20Expressions.pdf#page=1)

**Penjelasan:** Bartlett 2006 menegaskan FACS bukan hanya alat ukur — ia juga *bahasa*: ekspresi wajah bisa dideskripsikan secara universal lewat AU, lalu dihubungkan ke makna afektif yang sudah diketahui ilmu perilaku. Ini yang memungkinkan Craig 2008 menghubungkan AU ke emosi belajar secara sistematis.

---

> "subjective labeling of expressions has also been shown to be less reliable than objective coding for finding relationships between facial expression and other state variables."

— Bartlett et al. (2006). (p. 22)  [PDF](../../paper/22-%20Bartlett%202006%20-%20Automatic%20Recognition%20of%20Facial%20Actions%20in%20Spontaneous%20Expressions.pdf#page=1)

> "Spontaneous facial expressions differ from posed expressions in both which muscles are moved, and in the dynamics of the movement."

— Bartlett et al. (2006). (abstract)  [PDF](../../paper/22-%20Bartlett%202006%20-%20Automatic%20Recognition%20of%20Facial%20Actions%20in%20Spontaneous%20Expressions.pdf#page=1)

**Penjelasan:** Dua poin penting: (1) penilaian subjektif ("ini ekspresi marah") kurang reliable daripada pengkodean objektif AU — ini yang membenarkan penggunaan MediaPipe FaceLandmarker (dengan konversi blendshape→AU berbasis FACS) alih-alih tebakan kategori visual. (2) Ekspresi *spontan* (saat siswa belajar sungguhan) berbeda dari ekspresi *dipaksakan* (posed) — ini alasan kita tidak bisa sekadar latih model dari dataset emosi dasar yang biasanya acted.

> "The Facial Action Coding System is a comprehensive framework for objectively describing facial expression in terms of Action Units, which measure the intensity of over 40 distinct facial muscles. Manual FACS coding has previously been used to study student engagement and other emotions relevant to automated teaching."

— Whitehill, J., Serpell, Z., Lin, Y-C., Foster, A., & Movellan, J.R. (2014). *The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions.* IEEE Transactions on Affective Computing. (§3.1.3)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=7)

**Penjelasan:** Whitehill mengkonfirmasi bahwa FACS — dan khususnya AU — sudah dipakai secara manual untuk mempelajari *engagement* siswa. Ini menjembatani §1 (kenapa 4 emosi belajar) dengan §3 (AU spesifik apa untuk tiap emosi): karena Craig 2008 pakai FACS pada konteks belajar, hasilnya bisa langsung dipakai sebagai pemetaan AU→emosi. **Di kode:** alasan menggunakan MediaPipe FaceLandmarker blendshape (dipetakan ke intensitas AU baseline-normalized via `core/blendshape_features.py`, chain Turrisi 2026 BF→AU — lihat §0) alih-alih hanya tebakan kategori visual.

> **Jembatan → 3. AU Mana untuk Emosi Mana? (Temuan Empiris Craig 2008):** FACS/AU terbukti cara ukur yang objektif & reliable. Lalu AU spesifik mana yang muncul untuk tiap emosi belajar? Craig 2008 memetakannya secara empiris lewat data mining.

---

## 3. AU Mana untuk Emosi Mana? (Temuan Empiris Craig 2008)

**Konteks:** Kita sudah tahu bahwa FACS/AU adalah cara yang tepat. Sekarang pertanyaannya: AU mana yang secara empiris terbukti muncul saat siswa mengalami tiap emosi belajar? Craig 2008 menjawab ini lewat *data mining* dari video ekspresi wajah siswa yang dikode FACS oleh dua rater terlatih.

> "A standard data mining procedure called the *a priori* algorithm was used to identify frequent sets of action units and to extract association rules that could conditionally detect the presence of AUs on the face. Association rules are probabilistic in nature and take the form *Antecedent → Consequent [support, confidence]*."

> "The confidence measures its certainty and is the conditional probability that a data instance containing the antecedent will contain the consequent. For example, we observed an association rule with confusion of AU4 → AU7 (see Table 2) for two action units AU4 (antecedent) and AU7 (consequent). This can be interpreted as 'the presence of action unit 4 triggers action unit 7'."

— Craig et al. (2008). (p. 783)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=7)

**Penjelasan:** Craig memakai *association rule mining* pada data FACS yang sudah dikode — bukan opini atau teori, melainkan pola statistik dari ekspresi wajah nyata siswa. "Coverage 100%" berarti pola itu muncul di seluruh 100 random sample untuk emosi tersebut. "AU4 → AU7" berarti: saat siswa bingung dan AU4 (alis turun) aktif, AU7 (kelopak mata menegang) ikut muncul 73% dari waktu.

---

> "Our analyses were able to determine significant relationships with AUs for frustration, confusion, and boredom. It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together. Additionally, these AUs mutually trigger each other. That is, a raised inner brow tends to trigger a raised outer brow, and vice versa. Confusion displayed associations with AUs 4, 7, and 12. Action units 4 and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow). While boredom displayed a significant association with action unit 43 (eye closure), no association rules between AUs were observed, but some weaker nonsignificant trends between eye movement, such as blinks and eye closure, and AUs related to mouth movement."

— Craig et al. (2008). (p. 784)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=8)

**Penjelasan — hasil inti dan terjemahan AU:**

| AU | Nama FACS | Artinya (gerakan otot) | Emosi |
|---|---|---|---|
| **AU1** | Inner Brow Raiser | Bagian **dalam alis naik** (alis bagian tengah terangkat) | Frustration (primer) |
| **AU2** | Outer Brow Raise | Bagian **luar alis naik** (seluruh alis terangkat) | Frustration (primer) |
| **AU14** | Dimpler | **Lesung pipit** terbentuk / sudut bibir tertarik | Frustration (sekunder) |
| **AU4** | Brow Lowerer | **Alis turun/mengerut** (alis ditarik ke bawah & tengah) | Confusion (primer, 95%) |
| **AU7** | Lid Tightener | **Kelopak mata menegang** / sedikit menyipit | Confusion (78%) |
| **AU12** | Lip Corner Puller | **Sudut bibir tertarik ke samping** (senyum) | Confusion (sekunder, *gate*) |
| **AU43** | Eyes Closed | **Mata menutup** / mengantuk | Boredom (satu-satunya signifikan) |

AU1+AU2 "saling memicu" berarti: begitu satu terangkat, yang lain ikut — ini kenapa kode memakai perkalian geometris `(AU1·AU2)^0.5` (baru kuat kalau keduanya aktif bersamaan). Boredom hanya punya AU43 yang signifikan — sinyal lain (gerakan mulut, dll.) disebut Craig "nonsignificant" sehingga tidak dipakai di kode.

**Koreksi penting:** Versi lama dokumen ini mencantumkan "Confusion | AU12 | 95%". **Itu keliru** — 95% di Table 2 milik **AU4**, bukan AU12. AU12 hanya disebut di *prosa* sebagai asosiasi sekunder yang lebih lemah:

> "AU 12, AU 14, and AU 43, which received less support from the associations…" *(Craig 2008, p. 784)*
> "Notable exceptions are AUs 12 and 14 that occur during expressions of both confusion and frustration." *(Craig 2008, p. 785)*

AU12 muncul di **kedua** confusion dan frustration (tidak diskriminatif) → dipakai di kode sebagai *gate* (peredam) dengan floor, bukan sinyal positif.

**Tabel ringkas Craig 2008 Table 2:**

| Emosi | AU | Deskripsi (FACS) | Coverage |
|---|---|---|---|
| Frustration | 2 | Inner brow raise (alis dalam naik) | 100% |
| Frustration | 1 | Outer brow raise (alis luar naik) | 100% |
| Frustration | 1,2 | Keduanya bersamaan | 100% |
| Confusion | 4 | Brow lowerer (alis turun) | 95% |
| Confusion | 7 | Lid tightener (kelopak menegang) | 78% |
| Confusion | 4,7 | Alis turun + kelopak menegang bersama | 73% |
| Boredom | 43 | Eye closure (mata menutup/mengantuk) | 40% |

> **⚠️ Catatan penomoran (PENTING — jangan dikira salah ketik):** Tabel di atas **verbatim** dari Craig 2008 Table 2, yang memakai penomoran **NON-STANDAR**: di Craig, pola **"2" = Inner brow raise** dan **"1" = Outer brow raise** (terbalik dari FACS standar). FACS standar (Ekman & Friesen 1978) dan MediaPipe: **AU1 = Inner Brow Raiser** (`browInnerUp`), **AU2 = Outer Brow Raiser** (`browOuterUpLeft/Right`). **Di kode**, sistem memakai penomoran **FACS standar** (`au["AU1"]` = browInnerUp = inner; `au["AU2"]` = browOuterUp = outer) — dikonfirmasi Grafsgaard 2013. Jadi maknanya tetap sama (inner+outer brow raise = Frustration); hanya label angka di tabel Craig yang terbalik. Tabel ini sengaja dipertahankan apa adanya agar setia ke sumber.

> "The two expert raters received an overall kappa that ranged between .76 and .84."

— Craig et al. (2008). (abstract)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=1)

**Penjelasan:** Kappa 0.76–0.84 = kesepakatan antar-rater yang *substansial hingga hampir sempurna* (skala Cohen's kappa). Ini membuktikan pengkodean AU-nya reliable — dua orang terlatih yang mengkode wajah yang sama mendapat hasil serupa, artinya AU yang terdeteksi bukan subjektif.

### 3.1 PRESENCE vs INTENSITAS — apa arti "emosi dipengaruhi AU"? (penting, sering disalahpahami)

Pertanyaan wajar: kalau AU4 = sinyal Confusion, apakah **makin besar nilai AU4 makin bingung**? Jawabannya **bertingkat** — dan penting dibedakan supaya tidak salah klaim:

**(a) Dasar inti Craig 2008 = PRESENCE (kehadiran), BUKAN intensitas.** Craig memakai *association rule mining* atas AU yang dikode **ada/tidak**, dan metriknya **coverage** = seberapa **sering AU itu HADIR** saat suatu emosi:

> "coverage of 100% for a pattern indicates that it was observed in all 100 data sets for an affective state"

> "the presence of action unit 4 triggers action unit 7"

— Craig et al. (2008). (p. 783–784)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=7)

Jadi "AU4 = Confusion (95%)" artinya **"saat siswa bingung, AU4 HADIR di 95% episode"** — sebuah **asosiasi kehadiran**, bukan kurva "makin tinggi AU4 makin bingung". **Craig tidak menyebut kata *intensity* sama sekali.**

**(b) Dukungan untuk graded ("makin tinggi → makin cenderung") datang dari Grafsgaard 2013, lewat KORELASI:**

> "Action Unit 4 (brow lowering) was positively correlated with frustration"

— Grafsgaard et al. (2013). (abstract)  [PDF](../../paper/16-%20Grafsgaard%202013%20-%20Automatically%20Recognizing%20Facial%20Indicators%20of%20Frustration.pdf#page=1)

CERT mengeluarkan nilai AU **kontinu**, jadi "berkorelasi positif" memang berarti **kecenderungan graded** (AU4 lebih tinggi → lebih cenderung frustrasi). Tapi ini **korelasi**, bukan ambang pasti.

**(c) FACS sendiri punya intensitas** (Whitehill: *"measure the intensity of over 40 distinct facial muscles"*), dan **sistem ini memang memakai intensitas kontinu**: blendshape MediaPipe → AU baseline-normalized (deviasi dari netral) → skor kontinu → **threshold empiris** menentukan label aktif/tidak.

**Kesimpulan jujur:**
- Yang **divalidasi paper** untuk emosi belajar = **PRESENCE** (Craig) + **arah korelasi** (Grafsgaard). Keduanya mendukung "ciri itu HADIR → emosinya ada" dan "lebih kuat → lebih cenderung".
- "Nilai AU sekian = pasti emosi X" **TIDAK** dari paper — **angka ambang (threshold) adalah kalibrasi empiris** sistem ini (lihat DESIGN_RATIONALE), bukan ketentuan jurnal.
- Maka untuk **anotasi manual**: nilai **KEHADIRAN/kejelasan** ciri (alis berkerut **terlihat** atau tidak), bukan mengukur angka. Makin jelas → makin yakin, tapi keputusan akhir tetap **biner** (label ada/tidak). Lihat `PANDUAN_ANOTASI_MANUAL.md` bagian "Cara membaca AU".

> **Jembatan → 4. FRUSTRATION:** Peta AU→emosi sudah solid (kappa antar-rater tinggi, berbasis presence + korelasi). Sekarang tiap emosi dibedah satu per satu — dimulai dari Frustration dan sinyal alis-naiknya.

---

## 4. FRUSTRATION — AU1+AU2 sebagai Sinyal Primer

**Konteks:** Dari §3 kita tahu AU1+AU2 (alis dalam dan luar naik) adalah penanda Frustration. Tapi *mengapa* frustration muncul saat belajar? Dan apa yang membedakannya dari confusion?

> "It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together."

— Craig et al. (2008). (p. 784)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=8)

**Penjelasan:** AU1 (alis dalam naik) + AU2 (alis luar naik) bersama-sama = sinyal primer Frustration dengan coverage 100% — artinya setiap kali ada episode frustration, kedua AU ini muncul. "Mutually trigger each other" berarti ekspresi alis naik ini self-reinforcing. **Di kode:** `brow_raise_co = (au["AU1"] * au["AU2"]) ** 0.5` — perkalian geometris memastikan keduanya harus aktif bersamaan agar skornya tinggi.

---

> "Hopeless confusion occurs when the impasse cannot be resolved, the student gets stuck, there is no available plan, and important goals are blocked. The model hypothesizes that learners will experience frustration in these situations."

— D'Mello & Graesser (2012). (p. 147)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=3)

**Penjelasan:** D'Mello menjelaskan *kapan* frustration muncul: ketika siswa tidak bisa menyelesaikan hambatan (*stuck*, tidak ada rencana, tujuan terblokir). Ini berbeda dari confusion (§5) yang bisa produktif. Frustration = confusion yang tidak terselesaikan. Ini menghubungkan §4 dan §5 secara logis.

---

> "Negative affective states, such as the frustration, disappointment, or anger can occur when a learner is stuck at an impasse or in reaction to feedback from the learning environment."

— D'Mello, S.K., Blanchard, N., Baker, R., Ocumpaugh, J., & Brawner, K. (2014). *I Feel Your Pain: A Selective Review of Affect-Sensitive Instructional Strategies.* (p. 1)  [PDF](../../paper/24-%20DMello%202014%20-%20I%20Feel%20Your%20Pain%20Affect-Sensitive%20Instructional%20Strategies.pdf#page=1)

**Penjelasan:** D'Mello 2014 mengkonfirmasi frustration = reaksi terhadap hambatan (*stuck at impasse*) atau umpan balik negatif. Ini menguatkan bahwa frustration bukan emosi yang berdiri sendiri — ia muncul *sebagai respons* terhadap situasi spesifik dalam proses belajar.

**AU4 & AU14 sebagai korelat POSITIF frustration — Grafsgaard 2013 (verbatim):**

> "1) Action Unit 2 (outer brow raise) was negatively correlated with learning gain, 2) Action Unit 4 (brow lowering) was positively correlated with frustration, and 3) Action Unit 14 (mouth dimpling) was positively correlated with both frustration and learning gain."

— Grafsgaard, J.F., Wiggins, J.B., Boyer, K.E., Wiebe, E.N., & Lester, J.C. (2013). *Automatically Recognizing Facial Indicators of Frustration: A Learning-Centric Analysis.* ACII 2013. (abstract)  [PDF](../../paper/16-%20Grafsgaard%202013%20-%20Automatically%20Recognizing%20Facial%20Indicators%20of%20Frustration.pdf#page=1)

**Penjelasan & keputusan penting:** Grafsgaard 2013 = paper frustrasi konteks **belajar dengan deteksi OTOMATIS (CERT)** — paling cocok dengan use-case sistem ini (otomatis, e-learning). Temuannya: **AU4 (alis turun) berkorelasi POSITIF dengan frustration**, dan **AU14 (dimpler) positif dengan frustration + learning**. Di MediaPipe, browDown (AU4) memiliki range sangat sempit (median 0.001, p99 ~0.18), sehingga kalibrasi stretch agresif diterapkan (`AU4_active=0.05`) dan sinyal co-occur noseSneer (×0.3) ditambahkan sebagai booster implisit — sehingga deviasi kecil tetap terdeteksi. **Di kode:** `face_secondary = max(au["AU4"], au["AU14"])` dengan `face_weight=0.65` (dinaikkan dari 0.60 untuk kompensasi MediaPipe-only).

---

**Konfirmasi lintas-domain ekspresi frustrasi — Bosch et al. (2023):**

> "An analysis of participants' facial expressions during frustrating driving situations confirms previously reported expressions of frustration (Brow Lowerer, Dimpler, Brow Raiser, Smile and Lip Press)."

— Bosch, E., Käthner, D., Jipp, M., Drewitz, U., & Ihme, K. (2023). *Fifty Shades of Frustration: Intra- and Interindividual Variances in Expressing Frustration.* Transportation Research Part F, 94, 436–452. (abstract)  [PDF](../../paper/10-%20Bosch%202023%20-%20Fifty%20Shades%20of%20Frustration%20Intra%20and%20Interindividual%20Variances%20in%20Expressing%20Frustration.pdf#page=1)

**Penjelasan:** Bosch 2023 (konteks mengemudi, bukan belajar) mengkonfirmasi AU frustrasi yang sama yang kita pakai: **Brow Lowerer (AU4)**, **Dimpler (AU14)**, **Brow Raiser (AU1/AU2)**, Smile, Lip Press. Karena temuan lintas-domain (mengemudi DAN belajar) konvergen ke AU yang sama, ini menguatkan validitas pemetaan AU→Frustrasi kita. (Lip Press/AU24 tidak dipakai karena dynamic range MediaPipe mouthPress-nya sempit — diprioritaskan AU4 dan AU14 yang lebih ekspresif.)

---

**Variasi antar-individu → kalibrasi baseline per-orang (Bosch 2023):**

> "the results also hint towards high variance between and low variance within participants for all other expressions, suggesting the existence of individual-typical expressions of frustration."

> "future frustration-aware systems could benefit from considering these individual differences by using a universally trained algorithm that is then customized towards each individual."

— Bosch et al. (2023). (abstract / discussion)  [PDF](../../paper/10-%20Bosch%202023%20-%20Fifty%20Shades%20of%20Frustration%20Intra%20and%20Interindividual%20Variances%20in%20Expressing%20Frustration.pdf#page=1)

**Penjelasan & wujud di kode:** Bosch menemukan ekspresi frustrasi sangat **bervariasi antar-orang** tapi konsisten dalam-orang → merekomendasikan **kalibrasi per-individu**. Inilah dasar fitur **baseline netral per-orang** di sistem: tiap orang ditandai 1 frame netral, lalu intensitas AU dihitung sebagai **deviasi dari netral pribadinya** (`person_neutral`), bukan dari baseline populasi. Ini menghilangkan bias struktural (mis. orang yang alisnya natural rendah tidak lagi salah-baca "confused"). Sesuai juga dengan prinsip FACS: intensitas AU = deviasi dari netral (Bartlett 1999, §2). **Di kode:** `core/action_units.py` `compute_action_units(..., person_neutral=...)` + `utils/person_neutral.py`. Nilai netral disimpan dalam format MediaPipe AU (AU1, AU2, AU4, ...) per orang.

> **Jembatan → 5. CONFUSION:** Frustration = impasse yang TAK teratasi (alis naik, AU4/AU14). Tapi sebelum menyerah, siswa lebih dulu mengalami Confusion yang justru bisa produktif — dan sinyal alisnya berlawanan (turun).

---

## 5. CONFUSION — AU4+AU7 dan Hubungannya dengan Engagement

**Konteks:** Confusion adalah emosi kognitif yang unik — bisa *produktif* (bingung tapi masih mencari solusi) atau *deadlock* (bingung dan menyerah). Penting untuk memahami ini agar tidak salah menekan confusion di sistem.

> "Confusion displayed associations with AUs 4, 7, and 12. Action units 4 and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow)."

— Craig et al. (2008). (p. 784)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=8)

**Penjelasan — AU Confusion:**
- **AU4** (alis turun/mengkerut): otot *corrugator supercilii* menarik alis ke bawah — ekspresi alis berkerut
- **AU7** (kelopak menegang): otot *orbicularis oculi* (bagian orbital) menegang — mata terlihat sedikit menyipit tapi bukan tertutup
- **AU7→AU4** (asosiasi rule): saat kelopak menegang (AU7 aktif), alis cenderung ikut turun (AU4). Coverage 73% untuk keduanya bersama.
- **AU12** (senyum kecil): disebut Craig secara prosa sebagai sekunder, muncul di confusion DAN frustration → dipakai sebagai *gate* (peredam ber-floor), bukan sinyal positif

**Di kode:** `brow_dn_v = au["AU4"]`, `au7_v = au["AU7"]`, `base_conf = max(brow_dn_v, au7_sig, au4_au7_co_sig)`.

---

> "Learners experience cognitive disequilibrium when they are confronted with a contradiction, anomaly, system breakdown, or error, and when they are uncertain about what to do next. Confusion is a key signature of the cognitive disequilibrium that occurs when an impasse is detected."

— D'Mello & Graesser (2012). (p. 146)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=2)

**Penjelasan:** *Cognitive disequilibrium* (ketidakseimbangan kognitif) = kondisi saat informasi baru bertentangan dengan yang sudah diketahui. Confusion bukan sekadar "tidak tahu" — ia adalah *sinyal* bahwa otak sedang memproses ketidaksesuaian aktif. Ini penting untuk sistem: AU4+AU7 bukan hanya "wajah tidak senang" — ia spesifik untuk proses kognitif aktif ini.

---

> "The second hypothesis is the productive confusion hypothesis (Hypothesis 2). According to this hypothesis, cognitive disequilibrium, impasses, and confusion provide learners with an opportunity to think, deliberate, and problem solve."

— D'Mello & Graesser (2012). (p. 149)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=5)

**Penjelasan:** *Productive confusion* = confusion yang justru mendorong belajar, karena memaksa siswa berpikir lebih dalam. Ini adalah alasan utama confusion **boleh co-exist dengan engagement** di sistem (tidak saling suppress): seorang siswa yang engaged dan sedang berpikir keras *bisa sekaligus* menampilkan tanda confusion.

---

> "Confusion is considered to be the affective signature of these states. Therefore, one hypothesis is that events that confuse learners might provide valuable learning opportunities because learners need to engage in deep cognitive activities in order to resolve their confusion."

— D'Mello et al. (2014). (p. 6)  [PDF](../../paper/24-%20DMello%202014%20-%20I%20Feel%20Your%20Pain%20Affect-Sensitive%20Instructional%20Strategies.pdf#page=6)

**Penjelasan:** D'Mello 2014 mengkonfirmasi bahwa confusion bisa menjadi kesempatan belajar — bukan hanya hambatan. Ini menguatkan desain sistem yang membolehkan Confusion+Engagement co-occur.

---

**Confusion = emosi PALING SERING saat belajar (kenapa sistem harus mendeteksinya banyak):**

> "found that confusion was the most frequent emotion. The prevalence of confusion during complex learning activities motivated the present focus on this emotion."

> "confusion, which accompanies a state of cognitive disequilibrium that is triggered by contradictions, conflicts, anomalies, erroneous information, and other discrepant events, can be beneficial to learning if appropriately induced, regulated, and resolved."

— D'Mello, S., Lehman, B., Pekrun, R., & Graesser, A. (2014). *Confusion can be beneficial for learning.* Learning and Instruction, 29, 153–170. (abstract / §1)  [PDF](../../paper/25-%20DMello%202014b%20-%20Confusion%20Can%20Be%20Beneficial%20for%20Learning.pdf#page=1)

**Penjelasan & implikasi praktis:** Paper landmark ini menemukan **confusion = emosi PALING SERING** dalam pembelajaran kompleks (lebih sering dari boredom/frustration/engagement). Ini **validasi penting**: kalau sistem jarang mendeteksi confusion, itu menandakan **kalibrasi kurang sensitif** (threshold/per-orang), BUKAN karena confusion memang jarang — secara empiris ia justru dominan. Maka threshold Confusion sengaja diturunkan (0.35) dan kalibrasi per-orang dipakai agar confusion yang prevalen ini tertangkap. Paper juga menegaskan confusion **menguntungkan belajar** bila terselesaikan → memperkuat desain Confusion+Engagement co-occur (*productive confusion*).

---

**Confusion & Frustration sering MUNCUL BERSAMA ("confrustion") dan menguntungkan:**

> "it may be that students experience greater confusion and frustration while studying erroneous examples, but that their confusion and frustration lead to greater learning. We created and applied affect detectors for a combination of confusion and frustration ('confrustion')."

— Richey, J.E., Andres-Bray, J.M.L., Mogessie, M., Scruggs, R., Andres, J.M.A.L., Star, J.R., Baker, R.S., & McLaren, B.M. (2019). *More confusion and frustration, better learning: The impact of erroneous examples.* Computers & Education, 139, 173–190. (abstract)  [PDF](../../paper/26-%20Richey%202019%20-%20More%20Confusion%20and%20Frustration%20Better%20Learning%20Impact%20of%20Erroneous%20Examples.pdf#page=1)

**Penjelasan:** Richey 2019 (Carnegie Mellon/UPenn/Harvard, grup Baker) menemukan confusion & frustration **sering co-occur** sampai mereka memperlakukannya sebagai satu konstruk gabungan **"confrustion"**, dan kombinasi ini **berkorelasi dengan learning gain lebih tinggi**. Ini mendukung desain **multi-label** sistem (Confusion & Frustration boleh aktif bersamaan, tidak saling suppress). Catatan jujur: paper ini pakai *log-based affect detector* (sensor-free, dari interaksi tutor), **bukan** deteksi wajah/AU — jadi ia mendukung *konsep co-occurrence*, bukan menambah sinyal AU baru.

---

**Hasil empiris transisi Confusion ↔ Engagement (D'Mello 2012):**

> "The Confusion → Engagement/Flow and Confusion → Frustration transitions were also significant, while the Confusion → Boredom transition occurred at chance levels, thereby confirming the productive confusion and hopeless confusion hypotheses."

— D'Mello & Graesser (2012). (p. 153)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=7)

**Penjelasan:** Ini adalah **hasil uji statistik** dari D'Mello 2012 (dua studi terpisah). Transisi yang signifikan:
- **Confusion → Engagement** = signifikan (*productive confusion* terbukti) → confusion bisa berlanjut ke engagement
- **Confusion → Frustration** = signifikan (*hopeless confusion* terbukti) → confusion bisa berlanjut ke frustration
- **Confusion → Boredom** = **at chance** (tidak ada hubungan) → confusion tidak menyebabkan boredom

Catatan penting: paper D'Mello mengukur **transisi temporal** (urutan emosi selama sesi 30+ menit, bukan satu frame). Namun transisi ini menginformasikan *co-occurrence per frame*: jika Confusion→Engagement adalah pola temporal yang valid, maka dalam satu momen belajar, keduanya bisa aktif bersamaan (productive confusion). **Di kode:** suppression Conf→Eng **DIHAPUS** karena transisi ini justru signifikan positif.

---

> "Confusion has been correlated with facial action unit 4 (AU4, "Brow Lowerer") in multiple studies"

— Grafsgaard, J.F., Boyer, K.E., & Lester, J.C. (2011). *Predicting Facial Indicators of Confusion with Hidden Markov Models.* ACII 2011. (p. 1)  [PDF](../../paper/17-%20Grafsgaard%202011%20-%20Predicting%20Facial%20Indicators%20of%20Confusion%20with%20Hidden%20Markov%20Models.pdf#page=1)

**Penjelasan:** Grafsgaard 2011 secara independen mengaitkan AU4 (alis turun) dengan confusion dalam konteks belajar — menggunakan Hidden Markov Model (HMM) pada data video tutoring nyata, paper ini "utilizes these findings to predict student confusion as evidenced by student AU4". Ini menguatkan Craig 2008 dari sudut pendekatan machine learning yang berbeda. (Catatan: versi lama dokumen ini mengutip kalimat parafrase "AU4 was the most predictive…" yang BUKAN verbatim paper — sudah diganti dengan kalimat asli di atas.)

---

**Konfirmasi terbaru (2026): AU4+AU7 = korelat confusion paling reliable**

> "using the Facial Action Coding System (FACS), D'Mello and colleagues identified several facial actions associated with confusion, including brow lowering or frowning (AU4), eyelid tightening or squinting (AU7), upper lip raising (AU10), and lip pressing or tightening (AU24). Among these, AU4, AU7, and especially their combination (AU4+AU7), have been reported as the most reliable facial correlates of confusion."

— Dong, L., Wang, X., Frank, M., Setlur, S., Govindaraju, V., & Nwogu, I. (2026). *ConfusionBench: An Expert-Validated Benchmark for Confusion Recognition and Localization in Educational Videos.* arXiv:2603.17267. (§2)  [PDF](../../paper/13-%20Dong%202026%20-%20ConfusionBench%20Expert-Validated%20Benchmark%20for%20Confusion%20Recognition%20in%20Educational%20Videos.pdf#page=2)

**Penjelasan:** ConfusionBench (paper benchmark **terbaru, 2026**, untuk confusion di video edukasi) mengkonfirmasi secara langsung bahwa **AU4+AU7** (yang jadi inti Confusion di sistem ini) adalah **korelat wajah paling reliable** untuk confusion. Ini validasi terkini & terkuat untuk pemetaan AU4+AU7→Confusion. Paper juga menyebut AU10 (upper lip raise) & AU24 (lip press) sebagai asosiasi tambahan — tapi di MediaPipe keduanya tidak memiliki blendshape yang cukup akurat (AU10 tidak tersedia langsung; mouthPress dynamic range sempit), sehingga tidak dipakai agar tidak menambah noise. ConfusionBench juga mencatat **gaze direction + head pose** sebagai "supportive evidence" dan **hand-to-face** sebagai indikator "thinking, frustration, or hesitation".

---

**Cue tambahan: Mulut Terbuka (AU25+AU26) → Confusion** — Namba et al. (2024):

Setelah menemukan bahwa Craig 2008 tidak menemukan sinyal mulut yang signifikan untuk emosi belajar, paper baru ini mengisi celah tersebut secara langsung:

> "The results of Component 2 indicated opening the mouth (AU25, AU26)... Opening the mouth (Component 2) is a novel form that has not been previously reported in the relevant literature. From a data-driven perspective based on effect sizes, this movement can be considered the most significant component"

> "thinking situations elicited opening of the mouth (Component 2), blinking (Component 3), furrowing (Component 4), and smiling (Component 5)"

— Namba, S., Sato, W., Namba, S., Diel, A., Ishi, C., & Minato, T. (2024). *How an Android Expresses "Now Loading…": Examining the Properties of Thinking Faces.* International Journal of Social Robotics, 16, 1861–1877.  [PDF](../../paper/05-%20Namba%202024%20-%20How%20an%20Android%20Expresses%20Thinking%20Face%20Examining%20Properties%20of%20Thinking%20Faces.pdf#page=1)

**Penjelasan:**
- **AU25** (Lips Part) = bibir terbuka sedikit; **AU26** (Jaw Drop) = rahang turun/mulut terbuka lebih lebar
- Namba meneliti 40 partisipan yang menjawab pertanyaan sulit vs tidak berpikir (counting) — ekspresi diukur objektif dengan OpenFace
- Hasil: mulut terbuka (Component 2) adalah komponen **paling signifikan** saat berpikir menjawab pertanyaan sulit
- Chain ke sistem: *thinking face* (Namba) + *thinking/impasse = Confusion* (D'Mello 2012) → AU25+AU26 = cue Confusion
- **Di kode:** `sig_mouth_conf = max(au25_au26_co, au25_alone*0.5) * mouth_open_conf_w(0.78)` → Confusion cue kuat (tidak bisa memicu sendiri)

> **Jembatan → 6. BOREDOM:** Confusion masih aktif memproses (bahkan boleh co-exist dengan Engagement). Kebalikan dari keterlibatan aktif itu adalah penarikan diri pasif — Boredom.

---

## 6. BOREDOM — AU43 dan Hubungannya dengan Disengagement

**Konteks:** Dari semua emosi belajar, boredom memiliki sinyal AU paling sederhana: hanya AU43. Namun boredom juga berkaitan erat dengan arah pandang dan disengagement dari materi.

> "While boredom displayed a significant association with action unit 43 (eye closure), no association rules between AUs were observed, but some weaker nonsignificant trends between eye movement, such as blinks and eye closure, and AUs related to mouth movement."

— Craig et al. (2008). (p. 784)  [PDF](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf#page=8)

**Penjelasan:** **AU43** = mata menutup / kelopak turun berat — ekspresi mengantuk atau mata berat. Craig menemukan ini satu-satunya AU yang signifikan untuk boredom. AU lain (gerakan mulut seperti menguap, dll.) disebut "nonsignificant trends" — artinya tidak cukup kuat untuk dijadikan sinyal. **Di kode:** `blink_corrected = au["AU43"]` dari blendshape `eyeBlinkLeft/Right` MediaPipe (baseline-normalized via `compute_action_units()`). Yawn (`jawOpen`) sengaja *tidak* dipakai sebagai sinyal Boredom primer karena Craig tidak memvalidasinya — meskipun `jawOpen` (AU26) tersedia dan dipakai sebagai cue Confusion kuat (Namba 2024).

---

> "The fourth hypothesis, or the disengagement hypothesis (Hypothesis 4), states that persistent failure, which is related to frustration, eventually transitions into disengagement and boredom. According to this fourth hypothesis there should be a transition from frustration to boredom (Link 4), but frustration should not transition into engagement/flow."

> "Furthermore, consistent with forced-effort theories of boredom (Larson & Richards, 1991; Robinson, 1975), persistent frustration may transition into boredom, a crucial point at which the learner disengages from the learning process."

— D'Mello & Graesser (2012). (p. 147–149)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=3)

**Penjelasan:** D'Mello mendeskripsikan boredom sebagai titik di mana siswa **melepaskan diri dari proses belajar** (*disengages*) — sering sebagai ujung dari frustration yang berlarut. Sesuai *forced-effort theories* (Larson & Richards 1991; Robinson 1975): boredom terjadi ketika siswa terpaksa tetap mengerjakan sesuatu tapi tidak berhasil, sehingga akhirnya menyerah. Ini adalah transisi temporal (menit ke menit), bukan per-frame.

---

**Boredom dan Engagement bersifat komplementer — bukti empiris dari dataset e-learning:**

> "While working with the dataset, we noticed how boredom and engagement complimented each other. When engagement is low, boredom is generally high and vice-versa. In the instances that both boredom and engagement were low for a video snippet, the subject displayed high levels of confusion or frustration."

— Gupta et al. (2016). *DAiSEE.* (§5, "Complementary Labels")  [PDF](../../paper/02-%20Gupta%202016%20-%20DAiSEE%20Towards%20User%20Engagement%20Recognition%20in%20the%20Wild.pdf#page=10)

**Penjelasan:** DAiSEE menemukan secara empiris — dari data anotasi 9068 video klip e-learning nyata — bahwa boredom dan engagement cenderung berlawanan. *"When engagement is low, boredom is generally high and vice-versa"* = observasi distribusi label, bukan aturan deterministik. Penting: DAiSEE sendiri menunjukkan **pengecualian** (gambar Fig. 14 di paper): keduanya bisa rendah bersamaan saat frustration tinggi. Ini berarti Boredom↔Engagement bukan selalu mutually exclusive — hanya *cenderung* berlawanan.

**Di kode:** `bore_eng_suppress` di engagement mengurangi skor engagement saat boredom tinggi. Landasan empirisnya dari DAiSEE ("generally high... vice-versa") — ini observasi distribusi dataset, bukan aturan keras. Nilai suppression kecil (0.40) agar tidak memaksa mutually exclusive di setiap frame.

> **Jembatan → 7. ENGAGEMENT:** Boredom = disengagement (skornya bahkan saling meredam dengan engagement). Maka langsung kita lihat lawannya: Engagement dan bagaimana ia terbaca.

---

## 7. ENGAGEMENT — Penampilan Holistik sebagai Sinyal Utama

**Konteks:** Berbeda dari 3 emosi lain yang punya AU spesifik (AU1/2 untuk frustration, AU4/7 untuk confusion, AU43 untuk boredom), engagement tidak punya AU tunggal yang dominan. Whitehill 2014 meneliti ini secara khusus.

> "We found that human observers reliably agree when discriminating low versus high degrees of engagement (Cohen's κ = 0.96). When fine discrimination is required (4 distinct levels) the reliability decreases, but is still quite high (κ = 0.56). Furthermore, we found that engagement labels of 10-second video clips can be reliably predicted from the average labels of their constituent frames (Pearson r = 0.85), suggesting that static expressions contain the bulk of the information about engagement used by observers."

— Whitehill et al. (2014). (abstract)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=1)

**Penjelasan:** Dua temuan kunci:
1. **κ=0.96** untuk diskriminasi kasar (engaged vs tidak engaged) → manusia sangat konsisten menilai engagement dari wajah — artinya sinyal wajahnya cukup kuat untuk dibaca
2. **r=0.85** antara label klip video dan rata-rata label frame-nya → sebagian besar informasi engagement ada di penampilan statis per-frame, bukan di gerakan. Ini membenarkan sistem berlabel per-frame (bukan per-video): rata-rata prediksi per-frame sudah cukup mewakili engagement keseluruhan klip.

---

> "This accuracy is quite high and suggests that most of the information about the appearance of engagement is contained in the static pixels, not the motion per se."

— Whitehill et al. (2014). (§2.4)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=5)

**Penjelasan:** "Static pixels" = penampilan wajah diam, bukan gerakan. Ini menjustifikasi pendekatan **SigLIP** (visual language model yang membaca penampilan holistik frame statis) sebagai komponen utama untuk engagement, bukan hanya AU. Karena engagement tersebar di banyak fitur wajah sekaligus, SigLIP yang membaca penampilan secara menyeluruh lebih tepat daripada AU tunggal.

---

> "We hypothesize that a good deal of the information used by humans to make engagement judgements is based on the student's face."

— Whitehill et al. (2014). (p. 2)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=2)

**Skala engagement yang dipakai dalam penelitian (dan mendasari gate di kode):**

> "1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task, eyes completely closed."
> "2: Nominally engaged – e.g., eyes barely open, clearly not 'into' the task."
> "3: Engaged in task – student requires no admonition to 'stay on task'."
> "4: Very engaged – student could be 'commended' for his/her level of engagement in task."

— Whitehill et al. (2014). (§2.2)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=4)

**Penjelasan:** Empat level ini adalah deskripsi yang dipakai anotator manusia. Level 1 secara eksplisit menyebut "looking away from computer" dan "eyes completely closed" sebagai indikator tidak engaged. Ini menjadi landasan:
- **Gate gaze** di kode: saat wajah menoleh jauh dari layar (gaze_dev besar), engagement diturunkan
- **AU43 anti-engagement**: saat mata sangat tertutup (AU43 tinggi), engagement berkurang
- **Gaze gate di inference.py**: SigLIP tidak tahu arah pandang → gate gaze diterapkan setelah scoring untuk cegah engagement tinggi saat jelas tidak menatap layar

---

> "Engagement/flow is a cognitive-affective state that sometimes has a short time span, but at other times forms part of Csikszentmihalyi's (1990) conception of flow. It is important to point out that a learner can be engaged without necessarily experiencing flow."

— D'Mello & Graesser (2012). (p. 146)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=2)

**Penjelasan:** D'Mello mendefinisikan Engagement/Flow sebagai spektrum: dari sekadar "terlibat" (engaged) sampai kondisi *flow* yang mendalam. Sistem ini melabel "Engagement" dalam arti luas — mencakup keduanya. Engagement dianggap kondisi *baseline* saat siswa aktif mengejar tujuan belajar.

> **Jembatan → 8. Pendekatan Multimodal:** Engagement (dan ketiga emosi lain) terbaca dari wajah. Tapi wajah saja tidak cukup — peneliti memakai banyak kanal sinyal. Inilah dasar pendekatan multimodal.

---

## 8. Pendekatan Multimodal — Wajah + Konteks

**Konteks:** Mengapa sistem ini menggunakan lebih dari satu sumber sinyal (wajah, gaze, tangan)?

> "The new versions of AutoTutor detect learners' boredom, confusion, and frustration by monitoring conversational cues, gross body language, and facial features."

— D'Mello et al. (2009). *Responding to Learners' Cognitive-Affective States.* (abstract)  [PDF](../../paper/23-%20DMello%202009%20-%20Responding%20to%20Learners%20Cognitive-Affective%20States.pdf#page=1)

> "The affective states are sensed by monitoring conversational cues and other discourse features, gross body movements, and facial features."

— D'Mello et al. (2014). (p. 4)  [PDF](../../paper/24-%20DMello%202014%20-%20I%20Feel%20Your%20Pain%20Affect-Sensitive%20Instructional%20Strategies.pdf#page=4)

**Penjelasan:** Sistem AutoTutor yang D'Mello kembangkan memakai tiga kanal: (1) cue percakapan, (2) bahasa tubuh kasar, (3) fitur wajah. Sistem ini memakai dua dari tiga: wajah (AU + SigLIP) dan bahasa tubuh kasar (gaze head-pose + tangan). "Gross body language" di sini merujuk **sensor tekanan kursi** dalam penelitian D'Mello, bukan posisi tangan — sehingga tangan disandarkan ke Grafsgaard (§8.6), bukan ke kalimat ini.

> **Jembatan → 8.5 GAZE & HEAD POSE:** Dari tiga kanal D'Mello (percakapan, bahasa tubuh, wajah), sistem ini memakai wajah + bahasa tubuh. Kanal bahasa-tubuh pertama: arah pandang & kepala (gaze/head-pose).

---

## 8.5 GAZE & HEAD POSE — Sinyal Atensi (Dua Lapis: Konstruk + Metode)

**Konteks:** Gaze (arah pandang) dan posisi kepala dipakai sebagai sinyal boredom (pandang menjauh) dan engagement (pandang ke layar). Basis dibagi dua: *konstruk* (apa artinya gaze menjauh) dan *metode* (cara mengukurnya dari webcam).

**Konstruk — gaze/atensi menjauh dari konten = boredom/disengagement:**

> "We developed an intelligent tutoring system (ITS) that aims to promote engagement and learning by dynamically detecting and responding to students' boredom and disengagement. The tutor uses a commercial eye tracker to monitor a student's gaze patterns and identify when the student is bored, disengaged, or is zoning out."

— D'Mello, Olney, Williams & Hays (2012). *Gaze Tutor: A gaze-reactive intelligent tutoring system.* Int. J. Human-Computer Studies, 70, 377–398. (abstract)  [PDF](../../paper/12-%20DMello%202012%20-%20Gaze%20Tutor%20A%20Gaze-Reactive%20Intelligent%20Tutoring%20System.pdf#page=1)

**Penjelasan:** GazeTutor secara eksplisit menggunakan pola gaze (tatapan) untuk mendeteksi boredom dan disengagement — siswa yang *zoning out* (perhatian melayang) terdeteksi dari polanya tidak menatap area konten. Ini adalah justifikasi konstruk: gaze menjauh = tidak atensi = boredom/disengagement. **Di kode:** `gaze_dev_bore` mengukur seberapa jauh pandangan dari arah kamera/layar.

---

> "1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task, eyes completely closed."

— Whitehill et al. (2014). (§2.2) — "looking away" sebagai indikator tidak-engaged.  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=4)

**Penjelasan:** Whitehill secara eksplisit menyebut "looking away from computer" sebagai contoh level tidak engaged. Ini memperkuat GazeTutor dari sisi engagement — bukan hanya boredom.

---

**Metode — estimasi gaze dari WEBCAM (head pose), bukan eye-tracker:**

> "We trained deep embeddings for attentional and emotional features, training Attention-Net for head pose estimation and Affect-Net for facial expression recognition."

— Sümer, Goldberg, D'Mello, Gerjets, Trautwein & Kasneci (2021). *Multimodal Engagement Analysis from Facial Videos in the Classroom.* (abstract)  [PDF](../../paper/03-%20Sumer%202021%20-%20Multimodal%20Engagement%20Analysis%20from%20Facial%20Videos%20in%20the%20Classroom.pdf#page=1)

**Penjelasan:** Sümer 2021 memvalidasi pendekatan *head pose estimation dari video webcam* sebagai proksi atensi di kelas nyata. GazeTutor memakai eye-tracker (akurat tapi butuh kalibrasi), Sümer membuktikan estimasi head pose dari video cukup untuk analisis engagement. **Di kode:** `yaw/pitch/roll` dari MediaPipe FaceLandmarker + offset iris sebagai estimasi gaze — mengikuti pendekatan Sümer. Catatan limitasi: akurasi webcam lebih rendah dari eye-tracker; threshold dikalibrasi empiris.

---

**Logika ARAH gaze — kenapa bawah ≠ atas/samping (penting):**

Tidak semua "tidak menatap kamera" sama. Sümer 2021 membedakan **nunduk** (on-task) dari menjauh lainnya:

> "head-down (i.e., taking notes or reading learning material)" — Sümer et al. (2021)

> "Students can still focus on content when looking around or taking notes." — Sümer et al. (2021)

**Penjelasan & wujud di kode:**
- **Ke BAWAH (nunduk)** = baca soal / catat / keyboard = **ON-TASK → Engagement** (Sümer 2021). TIDAK memicu Boredom, TIDAK mengurangi Engagement.
- **Ke SAMPING / ATAS** = *"looking away from computer"* / zoning out = **disengaged → Boredom** (Whitehill 2014 level 1, GazeTutor).

Maka di kode, perhitungan gaze Boredom (`gaze_dev_bore`) & gate Engagement HANYA memakai komponen **horizontal (samping) + vertikal-ATAS**, dan **mengecualikan komponen ke bawah**. Hasilnya: nunduk-baca tetap Engagement; menoleh-samping/mendongak → Boredom. Angka threshold (yaw 14°, dll.) = kalibrasi empiris (arah-nya paper-justified, angkanya bukan dari paper).

> **Jembatan → 8.6 HAND-OVER-FACE GESTURES:** Gaze memberi tahu KE MANA atensi tertuju. Kanal bahasa-tubuh kedua melengkapinya: posisi TANGAN di wajah (hand-over-face).

---

## 8.6 HAND-OVER-FACE GESTURES — Tangan sebagai Cue Kognitif

**Konteks:** Tangan yang menyentuh/menutupi wajah adalah perilaku yang sering terlihat saat siswa belajar. Apakah ini bisa dijadikan sinyal emosi?

> "hand-over-face gestures can serve as an additional valuable channel for multi-modal affect inference for cognitive mental states."

— Mahmoud, M., Baltrušaitis, T., & Robinson, P. (2016). *Automatic Analysis of Naturalistic Hand-Over-Face Gestures.* ACM Trans. Interact. Intell. Syst. 6(2). (p. 2)  [PDF](../../paper/18-%20Mahmoud%202016%20-%20Automatic%20Analysis%20of%20Naturalistic%20Hand-Over-Face%20Gestures.pdf#page=2)

**Penjelasan:** Mahmoud 2016 membuktikan secara eksperimental (akurasi deteksi 83%) bahwa hand-over-face dapat diklasifikasikan otomatis dan berkaitan dengan *cognitive mental states*. "Cognitive mental states" = kondisi berpikir aktif, termasuk saat memproses informasi kompleks. Ini menjadi basis bahwa mendeteksi ada/tidaknya tangan di wajah relevan untuk sistem ini.

---

**Bukti KUANTITATIF tangan → "unsure" (≈ Confusion):**

> "index finger touching face appeared in 12 thinking segments and 2 unsure segments out of a total of 15 segments in this category... actions like stroking, tapping and touching facial regions - especially with index finger - are all associated with cognitive mental states, namely thinking and unsure."

— Mahmoud, M. & Robinson, P. (2011). *Interpreting Hand-Over-Face Gestures.* ACII 2011 (Doctoral Consortium). (§3.1)  [PDF](../../paper/11-%20Mahmoud%202011%20-%20Interpreting%20Hand-Over-Face%20Gestures.pdf)

**Penjelasan:** Ini link **paling langsung** tangan → Confusion. Mahmoud & Robinson 2011 menemukan secara kuantitatif: gestur tangan aktif ke wajah (jari telunjuk menyentuh/stroking/tapping) muncul di **"thinking" dan "unsure"** (cognitive states). *"Unsure"* ≈ Confusion (definisi D'Mello: *"uncertain about what to do next"* / *"noticeable lack of understanding"*). Jadi tangan→Confusion = **1 langkah** (bukan chain difficulty). Catatan jujur: Mahmoud juga menemukan gestur **pasif** (bersandar di telapak/kepalan) = *relaxed mood*, BUKAN cognitive — sedangkan deteksi sistem kita count-based (tak bedakan aktif vs pasif), jadi cue ini sedikit noisy.

---

> "There is a prominent increase in hand-over-face gestures when the difficulty level of the given exercise increases. The hand-over-face gestures occur more frequently during problem-solving (easy 23.79%, medium 19.84% and difficult 30.46%) exercises in comparison to reading (easy 16.20%, medium 20.06% and difficult 20.18%)."

— Behera, A., Matthew, P., Keidel, A., Vangorp, P., Fang, H., & Canning, S. (2020). *Associating Facial Expressions and Upper-Body Gestures with Learning Tasks.* Int. J. Artificial Intelligence in Education, 30, 236–270. (p. 251)  [PDF](../../paper/15-%20Behera%202020%20-%20Associating%20Facial%20Expressions%20and%20Upper-Body%20Gestures%20with%20Learning%20Tasks.pdf#page=16)

**Penjelasan:** Behera 2020 mengukur frekuensi hand-over-face pada 40 menit sesi belajar nyata dengan tiga tingkat kesulitan. Hasilnya: tangan-ke-wajah *meningkat signifikan* saat soal sulit (30.46%) vs mudah (23.79%), dan lebih sering saat *problem-solving* daripada membaca. Model kausal Behera: **difficulty → affect → gesture** (kesulitan → emosi → gestur). Ini menjadi chain argumentasi: HoF ↑ saat sulit → sulit = Confusion (D'Mello) → HoF = cue Confusion.

---

Dua verbatim berikut saling menguatkan argumen:

> "Learners experience cognitive disequilibrium when they are confronted with a contradiction, anomaly, system breakdown, or error, and when they are uncertain about what to do next. Confusion is a key signature of the cognitive disequilibrium that occurs when an impasse is detected."

— D'Mello & Graesser (2012). (p. 146) — *impasse/materi sulit = Confusion*.  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=2)

**Penjelasan chain:** Behera: *HoF naik saat difficulty naik*. D'Mello: *difficulty/impasse = Confusion*. Gabungan: HoF naik saat Confusion meningkat → HoF = cue kuat Confusion. Argumentasi dua langkah ini valid secara akademis karena tiap paper mendukung bagiannya masing-masing. **Di kode:** `sig_hand_conf = max(r.hand_one, r.hand_two) * ccfg["hand_conf_w"]` → Confusion. `max` karena Behera & Mahmoud tidak membedakan jumlah tangan.

---

**Perbedaan 1-tangan vs 2-tangan (Grafsgaard 2013b):**

> "Initial analyses of these hand-to-face gestures indicated that one-hand-to-face gestures may be associated with less negative affect, while two-hands-to-face gestures may be indicative of reduced focus."

— Grafsgaard, J.F., Wiggins, J.B., Boyer, K.E., Wiebe, E.N., & Lester, J.C. (2013). *Embodied Affect in Tutorial Dialogue: Student Gesture and Posture.* AIED 2013. (p. 2)  [PDF](../../paper/06-%20Grafsgaard%202013b%20-%20Embodied%20Affect%20in%20Tutorial%20Dialogue%20Student%20Gesture%20and%20Posture.pdf#page=2)

> "One-hand-to-face gestures are often thought of as embodiments of a thoughtful state."

— Grafsgaard et al. (2013b). (p. 6)  [PDF](../../paper/06-%20Grafsgaard%202013b%20-%20Embodied%20Affect%20in%20Tutorial%20Dialogue%20Student%20Gesture%20and%20Posture.pdf#page=6)

> "two-hands-to-face gestures occurred significantly more frequently among students with low self-efficacy."

— Grafsgaard et al. (2013b). (abstract)  [PDF](../../paper/06-%20Grafsgaard%202013b%20-%20Embodied%20Affect%20in%20Tutorial%20Dialogue%20Student%20Gesture%20and%20Posture.pdf#page=1)

**Penjelasan:**
- **1 tangan** = *"thoughtful state"* / *"less negative affect"* → sedang berpikir. Berpikir/unsure saat materi sulit = Confusion (Mahmoud 2011 "unsure" + Behera/D'Mello chain).
- **2 tangan** = temuan **signifikan**: muncul lebih sering pada siswa *self-efficacy* rendah → Frustration D'Mello (*"insecurity and dissatisfaction"*).

**Tangan → Frustration juga didukung paper lain:**

> "hand over face occlusions can provide additional information for recognition of some affective states such as curiosity, frustration and boredom."

— Nojavanasghari, B., Hughes, C.E., Baltrušaitis, T., & Morency, L-P. (2017). *Hand2Face: Automatic Synthesis and Recognition of Hand Over Face Occlusions.* ACII 2017. (abstract)  [PDF](../../paper/31-%20Nojavanasghari%202017%20-%20Hand2Face%20Synthesis%20and%20Recognition%20of%20Hand%20Over%20Face%20Occlusions.pdf)

> "Hand-to-face actions such as touching the chin, pressing the forehead, and covering the mouth may indicate thinking, frustration, or hesitation."

— Dong et al. (2026). *ConfusionBench.* (§2)  [PDF](../../paper/13-%20Dong%202026%20-%20ConfusionBench%20Expert-Validated%20Benchmark%20for%20Confusion%20Recognition%20in%20Educational%20Videos.pdf#page=2)

**Penjelasan:** Hand2Face (2017) dan ConfusionBench (2026) mengkonfirmasi hand-over-face berkaitan dengan **frustration** (juga boredom). Ini menguatkan pemetaan tangan→Frustration di luar Grafsgaard 2013b. Tetapi sistem ini **membatasi** kontribusi Frustration ke **2-tangan saja** (yang punya temuan paling spesifik: self-efficacy rendah, Grafsgaard 2013b) — 1-tangan hanya ke Confusion (thoughtful/unsure).

**Di kode:**
- **Confusion**: `sig_hand_conf = max(hand_one, hand_two) * hand_conf_w(0.78)` — semua tangan, dinaikkan karena basis "unsure" Mahmoud 2011 langsung.
- **Frustration**: `sig_hand_frus = hand_two * hand_frus_w(0.40)` — **HANYA 2-tangan** (Grafsgaard 2013b self-efficacy, dikuatkan Hand2Face/ConfusionBench).
- Semua cue → hanya menambah, tidak memicu sendiri.

> **Jembatan → 9. Dataset Referensi:** Sinyal lengkap sekarang: AU wajah + gaze + tangan. Pada dataset nyata seperti apa pendekatan ini berpijak? DAiSEE — e-learning in-the-wild.

---

## 9. Dataset Referensi — DAiSEE

**Konteks:** Sistem ini perlu dataset pembanding yang konteksnya sama persis: e-learning, webcam, 4 emosi yang sama, multi-label.

> "we introduce DAiSEE, the first multi-label video classification dataset comprising of 9068 video snippets captured from 112 users for recognizing the user affective states of boredom, confusion, engagement, and frustration 'in the wild'. The dataset has four levels of labels namely - very low, low, high, and very high for each of the affective states, which are crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists."

— Gupta et al. (2016). *DAiSEE.* (abstract)  [PDF](../../paper/02-%20Gupta%202016%20-%20DAiSEE%20Towards%20User%20Engagement%20Recognition%20in%20the%20Wild.pdf#page=1)

**Penjelasan:** DAiSEE adalah *dataset rujukan utama* karena: (1) e-learning in-the-wild — kondisi nyata, bukan lab; (2) webcam biasa, resolusi tinggi; (3) 4 emosi yang persis sama; (4) multi-label (tiap emosi berlabel independen); (5) anotasi *crowd-sourced* divalidasi psikolog ahli. Sistem ini menggunakan DAiSEE sebagai referensi desain: multi-label biner (§10), komplementaritas Bore/Eng (§6), dan 4 label target.

> **Jembatan → 10. Multi-label Biner:** DAiSEE memberi konteks dataset belajar nyata. Lalu bagaimana STRUKTUR label-nya — satu emosi atau banyak sekaligus? Inilah alasan multi-label biner.

---

## 10. Multi-label Biner — Mengapa Tiap Emosi Dinilai Independen

**Konteks:** Apakah satu orang hanya bisa merasa satu emosi sekaligus, atau bisa lebih? Ini menentukan arsitektur output sistem.

> "Authoritative studies have revealed that facial expressions in human daily life are in multiple or co-occurring mental states."
> "humans' facial representations are often not pure examples of a single expression category, but always admixtures of different emotions; that is, they appear as combinations, blends, or compounds of different basic emotions."

— Li, S. & Deng, W. (2019). *Blended Emotion in-the-Wild: Multi-label Facial Expression Recognition (RAF-ML).* IJCV. (p. 1)  [PDF](../../paper/19-%20Li%20Deng%202019%20-%20Blended%20Emotion%20in-the-Wild%20Multi-label%20Facial%20Expression%20Recognition%20%28RAF-ML%29.pdf#page=1)

**Penjelasan:** RAF-ML meneliti ekspresi wajah dari ratusan ribu gambar nyata dan menemukan ekspresi manusia hampir selalu *campuran* (*blended*) dari beberapa emosi, bukan satu emosi murni. Ini prinsip umum yang mendukung pendekatan multi-label. Catatan: RAF-ML meneliti emosi dasar (senang/sedih/marah/dll.), bukan emosi belajar — ia memberi *prinsip* multi-label, bukan pemetaan emosi belajar spesifik.

---

> "most FER databases are annotated with several basic mutually exclusive emotional categories … Each clip is categorized into one or more of the 11 widely-used emotions."

— Liu, Y. et al. (2022). *MAFW: A Large-scale, Multi-modal, Compound Affective Database.* ACM MM. (p. 1)  [PDF](../../paper/20-%20Liu%202022%20-%20MAFW%20Multi-modal%20Compound%20Affective%20Database.pdf#page=1)

**Penjelasan:** MAFW mengkritik bahwa kebanyakan dataset FER (*Facial Expression Recognition*) masih memakai label saling eksklusif ("hanya satu emosi per gambar"), padahal kenyataannya ekspresi majemuk (*compound*) sangat umum. Sistem ini mengambil pendekatan yang benar: tiap dari 4 emosi belajar berlabel independen.

---

**Mengapa SigLIP (sigmoid) bukan CLIP (softmax) untuk multi-label:**

> "the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization."

— Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV. (p. 1)  [PDF](../../paper/09-%20Zhai%202023%20-%20Sigmoid%20Loss%20for%20Language%20Image%20Pre-Training%20%28SigLIP%29.pdf#page=1)

**Penjelasan:** Perbedaan kritis SigLIP vs CLIP:
- **SigLIP (sigmoid)**: menilai tiap pasangan gambar-teks *secara independen*, menghasilkan skor 0–1 per emosi yang berdiri sendiri → **multi-label** (beberapa emosi bisa 1 sekaligus)
- **CLIP (softmax)**: menormalisasi skor *antar semua kandidat* (jumlah semua skor = 1) → **single-label** (hanya satu pemenang)

Untuk sistem ini yang perlu 4 emosi berlabel independen, SigLIP adalah pilihan yang tepat secara arsitektural.

---

**Mengapa zero-shot + multi-frame untuk anotasi:**

> "we explored a strategy that integrates multiple frames within 1-2 second video clips to enhance labeling performance and reduce costs."

— Zhang, H. & Fu, X. (2025). *Zero-shot Emotion Annotation in Facial Images Using Large Multimodal Models.* arXiv:2502.12454. (abstract)  [PDF](../../paper/33-%20Zhang%202025%20-%20Zero-shot%20Emotion%20Annotation%20in%20Facial%20Images%20Using%20Large%20Multimodal%20Models.pdf#page=1)

**Penjelasan:** Zhang & Fu (2025) memvalidasi paradigma *zero-shot annotation* (anotasi tanpa data latih domain-spesifik) menggunakan beberapa frame per klip. Sistem ini mengikuti pendekatan serupa: beberapa frame per video diproses bersama, hasilnya dirata-rata. Catatan: Zhang & Fu menguji pada 7 emosi dasar (single-label) — justifikasi *zero-shot + multi-frame* bisa diekstrapolasi, tapi multi-label dan SigLIP spesifik disandarkan ke paper lain di atas.

> **Jembatan → 11. Pelabelan Frame-Level (Statis) untuk Melatih Model Realtime:** Output multi-label per frame sudah tepat secara arsitektur (SigLIP sigmoid, bukan softmax). Untuk apa label statis ini? Untuk melatih model realtime hilir.

---

## 11. Pelabelan Frame-Level (Statis) untuk Melatih Model Realtime

**Konteks:** Sistem ini adalah *teacher/annotator* yang berat (menggunakan SigLIP + AU) untuk menghasilkan label per-frame. Label ini dipakai melatih model *student* yang ringan dan realtime (mis. MobileNet) dan/atau dikirim ke LLM. Mengapa label per-frame (statis) cukup?

> "most of the information about the appearance of engagement is contained in the **static pixels, not the motion per se**."
> "engagement labels of 10-second video clips can be reliably predicted from the **average labels of their constituent frames** (Pearson r = 0.85)."

— Whitehill et al. (2014). (§2.4 / abstract)  [PDF](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf#page=5)

**Penjelasan:** Dua temuan Whitehill ini memvalidasi frame-level labeling:
1. *Static pixels contain bulk of information* → tidak perlu video temporal untuk engagement — frame statis sudah cukup
2. *r=0.85 antara label klip dan rata-rata label frame* → jika kita label tiap frame lalu rata-rata, hasilnya sangat dekat dengan label klip secara keseluruhan

Ini sekaligus menjawab pertanyaan tentang *temporal paper* (D'Mello, dll.): meski D'Mello mengukur transisi temporal menit ke menit, Whitehill menunjukkan bahwa *rata-rata* frame-level sudah menangkap pola temporal tersebut (r=0.85). Artinya: basis temporal D'Mello dapat dioperasionalkan via rata-rata prediksi per-frame.

---

> "employing the GPT-4o-mini model for rapid, zero-shot labeling of **key frames** extracted from video segments … offering new avenues for **reducing labeling costs**."

— Zhang & Fu (2025). (abstract)  [PDF](../../paper/33-%20Zhang%202025%20-%20Zero-shot%20Emotion%20Annotation%20in%20Facial%20Images%20Using%20Large%20Multimodal%20Models.pdf#page=1)

**Penjelasan:** Zhang & Fu mengekstrak *key frames* dari video untuk anotasi — pendekatan yang persis sama dengan sistem ini (mengambil beberapa frame dari tiap video klip).

---

**Mengapa praktis maksimal 2 label aktif (Boredom/Engagement saling eksklusif):**

> "When engagement is low, boredom is generally high and vice-versa. In the instances that both boredom and engagement were low for a video snippet, the subject displayed high levels of confusion or frustration."

— Gupta et al. (2016). *DAiSEE.* (§5, "Complementary Labels")  [PDF](../../paper/02-%20Gupta%202016%20-%20DAiSEE%20Towards%20User%20Engagement%20Recognition%20in%20the%20Wild.pdf#page=10)

> "The model assumes that learners are typically in a prolonged state of either (a) engagement/flow as they pursue the superordinate learning goal of mastering the material in the learning environment or (b) disengagement (boredom) when they abandon pursuit of the superordinate learning goal."

— D'Mello & Graesser (2012). (p. 146)  [PDF](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf#page=2)

**Penjelasan:** Dua paper menunjukkan Boredom dan Engagement adalah dua *kondisi dasar* yang berlawanan: (a) sedang mengejar tujuan belajar = Engagement, (b) melepaskan diri dari tujuan belajar = Boredom. DAiSEE mengamati secara empiris bahwa keduanya cenderung berlawanan dalam data nyata. Gabungan dua paper ini membenarkan memperlakukan Boredom/Engagement sebagai pasangan yang *cenderung eksklusif* — meski dengan pengecualian (keduanya bisa rendah bersamaan saat Frustration tinggi, per DAiSEE Fig. 14).

Untuk Confusion/Frustration: D'Mello menemukan transisi Confusion→Frustration signifikan (berurutan), bukan eksklusivitas simultan. Perlakuan keduanya sebagai cenderung eksklusif di sistem ini adalah *penyederhanaan desain*, bukan ketentuan paper.

> **Jembatan → Daftar Lengkap Referensi:** Label frame-level berkualitas inilah produk akhir aplikasi — bahan latih untuk model realtime/LLM hilir. Seluruh klaim di atas bersumber pada referensi berikut.

---

## Daftar Lengkap Referensi

1. **Craig, S.D., D'Mello, S., Witherspoon, A., & Graesser, A. (2008).** Emote aloud during learning with AutoTutor: Applying the Facial Action Coding System to cognitive–affective states during learning. *Cognition & Emotion*, 22(5), 777–788.
   [PDF lokal](../../paper/08-%20Craig%202008%20-%20Emote%20Aloud%20during%20Learning%20with%20AutoTutor%20%28FACS%29.pdf)

2. **D'Mello, S.K. & Graesser, A. (2012).** Dynamics of Affective States during Complex Learning. *Learning and Instruction*, 22, 145–157.
   [PDF lokal](../../paper/01-%20DMello%20Graesser%202012%20-%20Dynamics%20of%20Affective%20States%20during%20Complex%20Learning.pdf) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0959475211000806)

3. **D'Mello, S., Craig, S., Fike, K., & Graesser, A. (2009).** Responding to Learners' Cognitive-Affective States with Supportive and Shakeup Dialogues. *AIED 2009.*
   [PDF lokal](../../paper/23-%20DMello%202009%20-%20Responding%20to%20Learners%20Cognitive-Affective%20States.pdf)

4. **D'Mello, S.K., Blanchard, N., Baker, R., Ocumpaugh, J., & Brawner, K. (2014).** I Feel Your Pain: A Selective Review of Affect-Sensitive Instructional Strategies.
   [PDF lokal](../../paper/24-%20DMello%202014%20-%20I%20Feel%20Your%20Pain%20Affect-Sensitive%20Instructional%20Strategies.pdf)

5. **Whitehill, J., Serpell, Z., Lin, Y-C., Foster, A., & Movellan, J.R. (2014).** The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions. *IEEE Transactions on Affective Computing.*
   [PDF lokal](../../paper/04-%20Whitehill%202014%20-%20The%20Faces%20of%20Engagement.pdf)

6. **Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016).** DAiSEE: Towards User Engagement Recognition in the Wild. *arXiv:1609.01885.*
   [PDF lokal](../../paper/02-%20Gupta%202016%20-%20DAiSEE%20Towards%20User%20Engagement%20Recognition%20in%20the%20Wild.pdf) · [arXiv](https://arxiv.org/abs/1609.01885)

7. **Bartlett, M.S., Hager, J.C., Ekman, P., & Sejnowski, T.J. (1999).** Measuring facial expressions by computer image analysis. *Psychophysiology*, 36, 253–263.
   [PDF lokal](../../paper/21-%20Bartlett%201999%20-%20Measuring%20Facial%20Expressions%20by%20Computer%20Image%20Analysis.pdf)

8. **Bartlett, M.S., Littlewort, G.C., Frank, M.G., Lainscsek, C., Fasel, I.R., & Movellan, J.R. (2006).** Automatic Recognition of Facial Actions in Spontaneous Expressions. *Journal of Multimedia*, 1(6), 22–35.
   [PDF lokal](../../paper/22-%20Bartlett%202006%20-%20Automatic%20Recognition%20of%20Facial%20Actions%20in%20Spontaneous%20Expressions.pdf)

9. **D'Mello, S., Olney, A., Williams, C., & Hays, P. (2012).** Gaze Tutor: A gaze-reactive intelligent tutoring system. *International Journal of Human-Computer Studies*, 70(5), 377–398.
   [PDF lokal](../../paper/12-%20DMello%202012%20-%20Gaze%20Tutor%20A%20Gaze-Reactive%20Intelligent%20Tutoring%20System.pdf) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1071581912000250)

10. **Sümer, Ö., Goldberg, P., D'Mello, S., Gerjets, P., Trautwein, U., & Kasneci, E. (2021).** Multimodal Engagement Analysis from Facial Videos in the Classroom. *IEEE Transactions on Affective Computing* (arXiv:2101.04215).
    [PDF lokal](../../paper/03-%20Sumer%202021%20-%20Multimodal%20Engagement%20Analysis%20from%20Facial%20Videos%20in%20the%20Classroom.pdf) · [arXiv](https://arxiv.org/abs/2101.04215)

11. **Grafsgaard, J.F., Boyer, K.E., & Lester, J.C. (2011).** Predicting Facial Indicators of Confusion with Hidden Markov Models. *ACII 2011, LNCS 6974.*
    [PDF lokal](../../paper/17-%20Grafsgaard%202011%20-%20Predicting%20Facial%20Indicators%20of%20Confusion%20with%20Hidden%20Markov%20Models.pdf)

12. **Grafsgaard, J.F., Wiggins, J.B., Boyer, K.E., Wiebe, E.N., & Lester, J.C. (2013).** Automatically Recognizing Facial Indicators of Frustration: A Learning-Centric Analysis. *ACII 2013.*
    [PDF lokal](../../paper/16-%20Grafsgaard%202013%20-%20Automatically%20Recognizing%20Facial%20Indicators%20of%20Frustration.pdf)

13. **Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023).** Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023* (arXiv:2303.15343).
    [PDF lokal](../../paper/09-%20Zhai%202023%20-%20Sigmoid%20Loss%20for%20Language%20Image%20Pre-Training%20%28SigLIP%29.pdf) · [arXiv](https://arxiv.org/abs/2303.15343)

14. **Li, S. & Deng, W. (2019).** Blended Emotion in-the-Wild: Multi-label Facial Expression Recognition Using Crowdsourced Annotations and Deep Locality Feature Learning (RAF-ML). *International Journal of Computer Vision.*
    [PDF lokal](../../paper/19-%20Li%20Deng%202019%20-%20Blended%20Emotion%20in-the-Wild%20Multi-label%20Facial%20Expression%20Recognition%20%28RAF-ML%29.pdf)

15. **Liu, Y. et al. (2022).** MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild. *ACM Multimedia 2022* (arXiv:2208.00847).
    [PDF lokal](../../paper/20-%20Liu%202022%20-%20MAFW%20Multi-modal%20Compound%20Affective%20Database.pdf) · [arXiv](https://arxiv.org/abs/2208.00847)

16. **Zhang, H. & Fu, X. (2025).** Zero-shot Emotion Annotation in Facial Images Using Large Multimodal Models. *arXiv:2502.12454.*
    [PDF lokal](../../paper/33-%20Zhang%202025%20-%20Zero-shot%20Emotion%20Annotation%20in%20Facial%20Images%20Using%20Large%20Multimodal%20Models.pdf) · [arXiv](https://arxiv.org/abs/2502.12454)

17. **Grafsgaard, J.F., Wiggins, J.B., Boyer, K.E., Wiebe, E.N., & Lester, J.C. (2013).** Embodied Affect in Tutorial Dialogue: Student Gesture and Posture. *AIED 2013.*
    [PDF lokal](../../paper/06-%20Grafsgaard%202013b%20-%20Embodied%20Affect%20in%20Tutorial%20Dialogue%20Student%20Gesture%20and%20Posture.pdf)

18. **Behera, A., Matthew, P., Keidel, A., Vangorp, P., Fang, H., & Canning, S. (2020).** Associating Facial Expressions and Upper-Body Gestures with Learning Tasks for Enhancing Intelligent Tutoring Systems. *Int. J. Artificial Intelligence in Education*, 30, 236–270.
    [PDF lokal](../../paper/15-%20Behera%202020%20-%20Associating%20Facial%20Expressions%20and%20Upper-Body%20Gestures%20with%20Learning%20Tasks.pdf)

19. **Mahmoud, M., Baltrušaitis, T., & Robinson, P. (2016).** Automatic Analysis of Naturalistic Hand-Over-Face Gestures. *ACM Trans. Interact. Intell. Syst.*, 6(2), Article 19.
    [PDF lokal](../../paper/18-%20Mahmoud%202016%20-%20Automatic%20Analysis%20of%20Naturalistic%20Hand-Over-Face%20Gestures.pdf)

20. **Namba, S., Sato, W., Namba, S., Diel, A., Ishi, C., & Minato, T. (2024).** How an Android Expresses "Now Loading…": Examining the Properties of Thinking Faces. *International Journal of Social Robotics*, 16, 1861–1877. — AU25+AU26 (mouth open) = most significant thinking face component (Component 2) saat menjawab pertanyaan sulit; mendasari cue Confusion di §5.
    [PDF lokal](../../paper/05-%20Namba%202024%20-%20How%20an%20Android%20Expresses%20Thinking%20Face%20Examining%20Properties%20of%20Thinking%20Faces.pdf)

21. **Mahmoud, M. & Robinson, P. (2011).** Interpreting Hand-Over-Face Gestures. *ACII 2011 (Doctoral Consortium), LNCS 6975.* — kuantitatif: jari-telunjuk-ke-wajah → 12 "thinking" + 2 "unsure" (≈Confusion); gestur pasif (bersandar) = relaxed. Link tangan→Confusion paling langsung (§8.6).
    [PDF lokal](../../paper/11-%20Mahmoud%202011%20-%20Interpreting%20Hand-Over-Face%20Gestures.pdf)

22. **Nojavanasghari, B., Hughes, C.E., Baltrušaitis, T., & Morency, L-P. (2017).** Hand2Face: Automatic Synthesis and Recognition of Hand Over Face Occlusions. *ACII 2017.* — hand-over-face → "curiosity, frustration and boredom". Mendukung tangan→Frustration (§8.6).
    [PDF lokal](../../paper/31-%20Nojavanasghari%202017%20-%20Hand2Face%20Synthesis%20and%20Recognition%20of%20Hand%20Over%20Face%20Occlusions.pdf)

23. **Dong, L., Wang, X., Frank, M., Setlur, S., Govindaraju, V., & Nwogu, I. (2026).** ConfusionBench: An Expert-Validated Benchmark for Confusion Recognition and Localization in Educational Videos. *arXiv:2603.17267.* — konfirmasi terbaru: AU4+AU7 = "most reliable facial correlates of confusion"; hand-to-face → thinking/frustration/hesitation (§5, §8.6).
    [PDF lokal](../../paper/13-%20Dong%202026%20-%20ConfusionBench%20Expert-Validated%20Benchmark%20for%20Confusion%20Recognition%20in%20Educational%20Videos.pdf)

24. **Bosch, E., Käthner, D., Jipp, M., Drewitz, U., & Ihme, K. (2023).** Fifty Shades of Frustration: Intra- and Interindividual Variances in Expressing Frustration. *Transportation Research Part F*, 94, 436–452. — konfirmasi AU frustrasi lintas-domain (Brow Lowerer/Dimpler/Brow Raiser/Smile/Lip Press) + variasi antar-individu → dasar kalibrasi baseline per-orang (§4).
    [PDF lokal](../../paper/10-%20Bosch%202023%20-%20Fifty%20Shades%20of%20Frustration%20Intra%20and%20Interindividual%20Variances%20in%20Expressing%20Frustration.pdf)

25. **D'Mello, S., Lehman, B., Pekrun, R., & Graesser, A. (2014).** Confusion can be beneficial for learning. *Learning and Instruction*, 29, 153–170. — confusion = emosi PALING SERING saat belajar kompleks + menguntungkan bila terselesaikan. Validasi: under-deteksi confusion = isu kalibrasi, bukan kelangkaan (§5).
    [PDF lokal](../../paper/25-%20DMello%202014b%20-%20Confusion%20Can%20Be%20Beneficial%20for%20Learning.pdf)

26. **Richey, J.E., Andres-Bray, J.M.L., Mogessie, M., Scruggs, R., Andres, J.M.A.L., Star, J.R., Baker, R.S., & McLaren, B.M. (2019).** More confusion and frustration, better learning: The impact of erroneous examples. *Computers & Education*, 139, 173–190. — confusion+frustration sering co-occur ("confrustion") & berkorelasi learning gain → dukung multi-label Conf+Frus (§5). Catatan: log-based detector, bukan AU.
    [PDF lokal](../../paper/26-%20Richey%202019%20-%20More%20Confusion%20and%20Frustration%20Better%20Learning%20Impact%20of%20Erroneous%20Examples.pdf)

27. **Turrisi, R., Iacono Isidoro, S., Bruschetta, R., Famà, F., Campisi, A., Aiello, S., Cusimano, G., Ruta, L., Pioggia, G., & Tartarisco, G. (2026).** Blendshape features meet action units: a clinical mapping for enhancing facial expression analysis. *Computers in Human Behavior Reports*, 22, 101125. — expert-validated mapping 52 MediaPipe blendshapes → AU FACS oleh 10 psikolog klinis bersertifikat (κ=0.92, 95.5% agreement). Menegaskan: `eyeLookDownL/R` = AU64 (gaze direction) — BUKAN AU7 (lid tightener) atau AU43 (blink). Dasar **chain rule** sistem ini (§Blendshape Features) dan **eyeLookDown gating** di `blendshape_features.py`.
    [PDF lokal](../../paper/07-%20Turrisi%202026%20-%20Blendshape%20Features%20Meet%20Action%20Units.pdf)

28. **Aldenhoven, C.M., Nissen, L., Heinemann, M., Doğdu, C., Hanke, A., Jonas, S., & Reimer, L.M. (2026).** Real-time emotion recognition performance of mobile devices: A detailed analysis of camera and TrueDepth sensors using Apple's ARKit. *Sensors*, 26, 1060. — native ARKit/MediaPipe blendshapes dipakai langsung untuk deteksi 7 emosi dasar via cosine similarity; akurasi 68.3% (melebihi human rater rata-rata 58.9%); validasi bahwa blendshape native MediaPipe cukup akurat tanpa model aproksimasi tambahan. Mendukung `BLENDSHAPE_SOURCE=mediapipe` sebagai sumber primer (§Blendshape Features).
    [PDF lokal](../../paper/14-%20Aldenhoven%202026%20-%20Real-Time%20Emotion%20Recognition%20Performance%20of%20Mobile%20Devices%20ARKit.pdf)
