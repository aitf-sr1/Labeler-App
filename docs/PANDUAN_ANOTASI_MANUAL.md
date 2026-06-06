# Panduan Anotasi Manual — Pemeriksaan Hasil Sistem

Dokumen ini digunakan saat **memeriksa hasil pelabelan aplikasi secara manual**. Untuk tiap video/frame, penilai melihat tampilan wajah dan menentukan emosi mana yang aktif berdasarkan ciri-ciri di bawah ini.

Semua ciri yang dicantumkan **berdasarkan verbatim dari paper** — bukan opini. Setiap ciri dilengkapi sumber papernya.

> **Cara pakai:** Buka video di aplikasi → lihat frame yang dilabeli AI → bandingkan dengan daftar ciri di bawah → jika ciri terpenuhi = emosi tersebut harusnya aktif.

---

## SISTEM LABEL

Tiap frame punya **4 label independen** (bisa lebih dari satu aktif bersamaan):

| Label | Aktif = 1 | Tidak aktif = 0 |
|---|---|---|
| **Boredom** | Tanda kebosanan terlihat | Tidak ada tanda kebosanan |
| **Engagement** | Tanda terlibat/fokus terlihat | Tidak ada tanda keterlibatan |
| **Confusion** | Tanda kebingungan terlihat | Tidak ada tanda kebingungan |
| **Frustration** | Tanda frustrasi terlihat | Tidak ada tanda frustrasi |

Boredom dan Engagement biasanya **tidak aktif bersamaan** — kalau satu tinggi, yang lain rendah (DAiSEE: *"When engagement is low, boredom is generally high and vice-versa"*). Confusion dan Engagement **bisa aktif bersamaan** (*productive confusion*, D'Mello 2012).

---

## 1. BOREDOM (Kebosanan)

### Definisi (Craig 2008)
> *"Boredom — the state of being weary and restless through lack of interest."*

Siswa terlihat **lelah/malas karena tidak tertarik** — bukan sedih, bukan mengantuk karena habis olahraga. Boredom spesifik untuk situasi belajar.

### Ciri Utama — HARUS ada setidaknya satu:

**A. Mata berat / menutup (AU43 Eye Closure)**
> *"While boredom displayed a significant association with action unit 43 (eye closure)."*
> — Craig et al. (2008), Table 2

- Kelopak mata **terlihat berat atau setengah menutup** tanpa alasan fisik
- Mata yang seharusnya terbuka normal terlihat droopy/lesu
- **Bukan** berkedip normal (terlalu singkat)
- **Bukan** menyipit karena cahaya atau fokus (itu AU7 → Confusion)

**B. Pandangan menjauh dari layar (Gaze Away)**
> *"monitor a student's gaze patterns and identify when the student is bored, disengaged, or is zoning out."*
> — D'Mello et al. (2012), GazeTutor

> *"1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task."*
> — Whitehill et al. (2014)

- Kepala/mata menoleh **jauh dari arah kamera/layar** — ke samping, ke atas melamun
- **Bukan** menunduk membaca (itu bisa Engagement juga)
- **Bukan** menoleh sesaat (harus jelas tidak memperhatikan)

**Tabel ARAH PANDANG — patokan cepat (penting!):**

| Arah pandang | Label | Alasan & paper |
|---|---|---|
| **Lurus ke depan/layar** | Engagement | Whitehill 2014: forward gaze = engaged |
| **Ke SAMPING** (menoleh kiri/kanan) | **Boredom** | GazeTutor/Whitehill: "looking away from computer" = disengaged |
| **Ke ATAS** (mendongak, melamun) | **Boredom** | Whitehill: looking away/zoning out |
| **Ke BAWAH** (nunduk ke keyboard/catatan) | **Engagement** ✓ | Sümer 2021: "head-down = taking notes or reading learning material" = on-task |

⚠️ **Catatan paling sering keliru:** nunduk ke bawah = **Engagement** (baca/ketik), BUKAN Boredom. Hanya **samping & atas** yang Boredom. Sistem sudah dikalibrasi persis begini.

### Ciri yang TIDAK dipakai (tidak ada basis paper):
- ~~Menguap~~ — Craig menyebut "non-significant trends"
- ~~Bergerak-gerak~~ — tidak divalidasi Craig untuk boredom

### Contoh JELAS Boredom:
- Siswa menatap langit-langit, mata setengah tutup
- Siswa menoleh ke jendela, ekspresi kosong
- Siswa terlihat mengantuk, kepala nyaris ke bawah

### Contoh yang BUKAN Boredom:
- Siswa menunduk membaca/menulis → lebih ke Engagement
- Siswa berkedip normal → bukan AU43 signifikan
- Siswa fokus tapi wajah datar → bisa Engagement tanpa ekspresi berlebihan

---

## 2. ENGAGEMENT (Keterlibatan/Fokus)

### Definisi (D'Mello & Graesser 2012)
> *"Engagement/flow — a state of interest that results from involvement in an activity."*

### Definisi level (Whitehill 2014)
> *"3: Engaged in task – student requires no admonition to 'stay on task'."*
> *"4: Very engaged – student could be 'commended' for his/her level of engagement in task."*

### Ciri Utama — HARUS ada:

**A. Menatap ke arah layar/kamera**
> *"1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task."*
> — Whitehill et al. (2014)

> *"looking away from the computer"* = sinyal TIDAK engaged.
> — Whitehill et al. (2014)

- Kepala relatif tegak menghadap layar
- Pandangan ke arah kamera (proxy layar)
- **Minimal tidak terlihat jelas menoleh jauh**

**B. Mata terbuka (tidak droopy)**
> *"2: Nominally engaged – e.g., eyes barely open, clearly not 'into' the task."*
> — Whitehill et al. (2014)

- Mata terbuka normal — tidak setengah tertutup
- Terlihat "ada" secara kognitif, bukan melamun

### Catatan penting:
- Engagement **tidak perlu** ekspresi dramatis — wajah datar tapi menatap layar = valid Engagement
- Whitehill (2014): *"most of the information about engagement is contained in the static pixels, not the motion"* → penampilan umum wajah, bukan gerakan spesifik
- Engagement **bisa co-occur** dengan Confusion (*productive confusion*)

**Kasus khusus: Lihat ke bawah ≠ Boredom**
> *"Students can still focus on content when looking around or taking notes."*
> *"head-down (i.e., taking notes or reading learning material)"*
> — Sümer et al. (2021)

Lihat ke bawah (menunduk ke keyboard/catatan) **tidak** berarti Boredom — bisa sangat Engaged sedang baca soal atau menulis. Yang menjadi sinyal Boredom adalah lihat ke **samping** atau **atas** (melamun/zoning out), bukan menunduk ke materi. Sistem sudah menangani ini: komponen vertikal ke bawah tidak masuk ke perhitungan gaze Boredom.

### Contoh JELAS Engagement:
- Siswa menatap lurus ke layar, mata terbuka penuh
- Siswa sedikit menunduk membaca, kepala tidak menoleh
- Siswa dengan ekspresi serius memperhatikan

### Contoh yang BUKAN Engagement:
- Kepala menoleh jauh ke samping
- Mata sangat menutup/droopy
- Jelas tidak memperhatikan materi

---

## 3. CONFUSION (Kebingungan)

### Definisi (D'Mello & Graesser 2012)
> *"Learners experience cognitive disequilibrium when they are confronted with a contradiction, anomaly, system breakdown, or error, and when they are uncertain about what to do next. Confusion is a key signature of the cognitive disequilibrium that occurs when an impasse is detected."*

Bukan sekadar "tidak tahu" — confusion adalah kondisi aktif mencoba **memproses sesuatu yang bertentangan atau tidak dipahami**.

### Ciri Utama dari FACS — setidaknya satu:

**A. Alis turun/mengerut (AU4 Brow Lowerer) — 95% coverage**
> *"Confusion displayed associations with AUs 4, 7, and 12... Action units 4 and 7 occur simultaneously."*
> — Craig et al. (2008), p. 784

- Alis ditarik ke **bawah dan tengah** — terlihat mengerut/berkerut
- Biasanya muncul di antara kedua alis (galur vertikal di tengah dahi)
- **Bukan** alis naik (itu Frustration/terkejut)
- Coverage 95%: muncul di hampir semua episode confusion yang dikode

**B. Kelopak mata sedikit menyipit (AU7 Lid Tightener) — 78% coverage**
> *"the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow)."*
> — Craig et al. (2008), p. 784

- Kelopak **sedikit menegang/menyipit** — bukan menutup, tapi terlihat lebih kecil dari normal
- Sering muncul bersamaan dengan AU4
- **Bukan** squinting karena cahaya terang
- **Bukan** mata menutup karena mengantuk (itu AU43 → Boredom)

**C. Kombinasi AU4+AU7 — 73% coverage**
- Alis turun **DAN** kelopak menyipit bersamaan = sinyal confusion paling kuat
- Ini ekspresi "mengernyit fokus mencoba memahami sesuatu"

### Ciri Tambahan (cue KUAT — setara sinyal AU utama, tetap idealnya disertai sinyal lain):

**D. Mulut sedikit terbuka (AU25+AU26)**
> *"Component 2 indicated opening the mouth (AU25, AU26)... can be considered the most significant component"* saat menjawab pertanyaan sulit.
> — Namba et al. (2024), Study 1 Discussion

- Bibir terbuka sedikit atau rahang sedikit turun
- Terlihat saat orang **berpikir keras** menjawab pertanyaan sulit
- Cue **KUAT** (Namba 2024: "komponen paling signifikan" thinking face, `mouth_open_conf_w=0.78`) — tapi pertimbangkan konteks (mulut mangap bisa juga sebab lain)

**E. Tangan di wajah — POSISI yang menandakan Confusion**
> *"There is a prominent increase in hand-over-face gestures when the difficulty level of the given exercise increases."* (sulit 30.46% vs mudah 23.79%)
> — Behera et al. (2020)

> *"index finger touching face appeared in 12 thinking segments and 2 unsure segments out of a total of 15 segments in this category... actions like stroking, tapping and touching facial regions - especially with index finger - are all associated with cognitive mental states, namely thinking and unsure."*
> — Mahmoud & Robinson (2011)

> *"Hand-to-face actions such as touching the chin, pressing the forehead, and covering the mouth may indicate thinking, frustration, or hesitation."*
> — Dong et al. (2026, ConfusionBench)

**Posisi tangan yang condong ke Confusion (pose "berpikir", kontak RINGAN, biasanya 1 tangan):**
- **Jari telunjuk** menyentuh / mengetuk / mengusap wajah (terutama dagu/pipi) → "thinking/unsure" (Mahmoud — 12–14/15 segmen kognitif)
- **Menyangga atau menyentuh dagu** sambil menatap soal → "touching the chin = thinking" (ConfusionBench)
- **Menutupi mulut ringan** dengan jari/tangan → "covering the mouth = hesitation" (ragu, masih memproses)

> ⚠️ **Catatan untuk anotator:** detektor otomatis app ini hanya **menghitung jumlah tangan** dekat wajah — ia TIDAK bisa membedakan posisi. **Kamu (manusia) bisa.** Maka penilaian posisi ini adalah nilai tambah anotator manual di atas skor AI.

- Tetap hanya **menguatkan**, tidak cukup sendiri (harus ada sinyal wajah/gaze pendukung).

### Catatan penting:
- Confusion **bisa co-occur dengan Engagement** (*productive confusion*) — D'Mello 2012 membuktikan transisi Confusion→Engagement signifikan. Siswa yang bingung tapi masih engaged = normal!
- Confusion **tidak berkaitan dengan Boredom** — D'Mello: *"Confusion → Boredom transition occurred at chance levels"*

### Contoh JELAS Confusion:
- Alis berkerut kuat, mata sedikit menyipit, menatap layar → alis turun + AU7
- Ekspresi "mikir keras", dahi berkerut, mulut sedikit terbuka → AU4+AU25
- Tangan di dagu sambil menatap soal → hand + gaze

### Contoh yang BUKAN Confusion:
- Alis naik (terkejut/frustasi — bukan AU4)
- Mata menutup penuh (boredom — AU43, bukan AU7)
- Ekspresi datar menatap layar (bisa Engagement saja)

---

## 4. FRUSTRATION (Frustrasi)

### Definisi (Craig 2008)
> *"Frustration — making vain or ineffectual efforts, however vigorous; a deep chronic sense or state of insecurity and dissatisfaction arising from unresolved problems or unfulfilled needs."*

### Konteks kemunculan (D'Mello & Graesser 2012)
> *"Hopeless confusion occurs when the impasse cannot be resolved, the student gets stuck, there is no available plan, and important goals are blocked. The model hypothesizes that learners will experience frustration in these situations."*

Frustrasi muncul saat siswa **stuck dan tidak bisa maju** — berbeda dari confusion (masih mencoba memproses). Frustrasi = sudah menyerah mencari jalan / sangat tidak puas.

### Ciri Utama — KEDUANYA harus aktif untuk sinyal kuat:

**A. Alis bagian dalam naik (AU1 Inner Brow Raiser) — 100% coverage**
> *"AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together."*
> — Craig et al. (2008), p. 784

- Bagian **dalam** alis (dekat tengah hidung) terangkat ke atas
- Membentuk sudut V terbalik di tengah dahi
- **Bukan** keseluruhan alis naik rata (itu terkejut/AU2 saja)

**B. Alis bagian luar naik (AU2 Outer Brow Raiser) — 100% coverage**
- Keseluruhan alis terangkat, termasuk bagian luar
- Craig: *"these AUs mutually trigger each other"* — satu muncul, yang lain ikut
- **Coverage 100%**: muncul di SEMUA episode frustration yang dikode Craig

**→ Ekspresi gabungan AU1+AU2: kedua bagian alis terangkat = "alis naik penuh"**

### Ciri Tambahan (sinyal sekunder, Grafsgaard 2013):

**C. Alis turun sekunder (AU4 Brow Lowerer)**
> *"The present work investigates whether an automated system can identify facial features related to a learning-centric measure of student frustration."* — AU4 berkorelasi positif.
> — Grafsgaard et al. (2013)

- Bisa muncul bersamaan dengan AU1+AU2 → ekspresi "mengernyit marah/frustrasi"

**D. Lesung pipit (AU14 Dimpler)**
- Sudut bibir tertarik membentuk lesung pipit kecil
- Sering muncul saat menahan emosi negatif
- Lebih halus, sulit dilihat manual

**E. Tangan di wajah — POSISI yang menandakan Frustration**
> *"two-hands-to-face gestures occurred significantly more frequently among students with low self-efficacy."*
> — Grafsgaard et al. (2013b)

> *"Hand-to-face actions such as touching the chin, pressing the forehead, and covering the mouth may indicate thinking, frustration, or hesitation."*
> — Dong et al. (2026, ConfusionBench)

**Posisi tangan yang condong ke Frustration (pose "kewalahan/menyerah", kontak BERAT/menekan):**
- **Menekan dahi** (telapak/jari menekan dahi atau pelipis) → "pressing the forehead = frustration" (ConfusionBench)
- **Kedua tangan** ke wajah / **menutupi wajah** (facepalm, menutup mata, menopang kepala dengan dua tangan) → low self-efficacy ≈ frustrasi (Grafsgaard 2013b)
- Bedanya dengan Confusion: lebih **menekan/menutup** dan cenderung **2 tangan**, bukan sentuhan ringan 1 jari.

> ⚠️ **Catatan untuk anotator:** sama seperti Confusion, detektor otomatis hanya menghitung jumlah tangan (2-tangan → cue Frustration pendukung). Posisi "menekan dahi / menutup wajah" hanya bisa kamu nilai secara manual.

- Tetap **menguatkan**, tidak cukup sendiri (idealnya disertai alis naik AU1+AU2 atau ekspresi distressed).

### Perbedaan Confusion vs Frustration — penting untuk tidak salah:

| Aspek | Confusion | Frustration |
|---|---|---|
| Alis | **Turun** (AU4, mengerut) | **Naik** (AU1+AU2, terangkat) |
| Ekspresi | Mengernyit fokus, berpikir | Alis naik, tampak distressed |
| Kondisi | Masih mencoba memproses | Stuck, tidak bisa maju |
| **Posisi tangan** | **Dagu / telunjuk**, kontak ringan, biasanya 1 tangan ("mikir") | **Dahi ditekan / menutup wajah**, kontak berat, cenderung 2 tangan ("kewalahan") |
| Gaze | Ke layar (masih mencoba) | Bisa ke mana saja |
| D'Mello | Cognitive disequilibrium | Goals blocked, hopeless |

### Contoh JELAS Frustration:
- Kedua alis terangkat penuh, wajah tampak distressed → AU1+AU2
- Alis naik sambil menghela napas / gerak kepala negatif
- Ekspresi "mau menyerah" — alis naik, mungkin sedikit gelengan kepala

### Contoh yang BUKAN Frustration:
- Alis turun/mengerut (itu Confusion — AU4 bukan AU1/AU2)
- Senyum → bukan frustration
- Ekspresi datar → tidak cukup sinyal

---

## Tabel Ringkasan Cepat

| Ciri yang dilihat | Emosi yang aktif | Paper |
|---|---|---|
| Mata berat/setengah menutup | **Boredom** | Craig 2008 (AU43) |
| Menatap layar + mata terbuka | **Engagement** | Whitehill 2014 |
| Jelas tidak lihat layar (menoleh jauh) | **Boredom** / tidak Engagement | GazeTutor, Whitehill |
| Alis **turun/mengerut** + kelopak sedikit menyipit | **Confusion** | Craig 2008 (AU4+AU7) |
| Mulut sedikit terbuka sambil menatap soal | **Confusion** (tambah) | Namba 2024 (AU25+AU26) |
| Telunjuk/tangan di **dagu** (ringan, "mikir") saat materi sulit | **Confusion** (tambah) | Behera 2020, Mahmoud 2011, ConfusionBench 2026 |
| Alis **naik** (inner + outer) | **Frustration** | Craig 2008 (AU1+AU2) |
| **Menekan dahi** / **2 tangan** menutup wajah | **Frustration** (tambah) | ConfusionBench 2026, Grafsgaard 2013b |
| Alis naik + turun bersamaan ("konfliktif") | Confusion **DAN** Frustration | Craig+Grafsgaard |
| Bingung tapi masih menatap layar | Confusion + Engagement | D'Mello 2012 |

---

## Co-occurrence & Suppression — Emosi Mana yang Bisa Bersamaan?

Ini patokan utama yang harus diingat sebelum menilai. Setiap baris punya **dasar paper** yang menjelaskan kenapa aturannya begitu.

### Tabel Patokan Utama

| Kombinasi | Boleh bersamaan? | Patokan untuk penilai | Basis paper |
|---|:---:|---|---|
| **Boredom + Engagement** | ❌ Hampir tidak mungkin | Kalau salah satu jelas tinggi, yang lain harusnya rendah. Kalau keduanya terlihat, nilai yang paling dominan = 1, yang lain = 0. | DAiSEE: *"When engagement is low, boredom is generally high and vice-versa"* — Gupta 2016 |
| **Confusion + Engagement** | ✅ Bisa & normal | Siswa yang bingung tapi masih menatap layar/mencoba = **keduanya aktif**. Ini kondisi belajar yang produktif. JANGAN suppress salah satu. | D'Mello 2012: *"productive confusion hypothesis"* — Confusion→Engagement transisi signifikan |
| **Confusion + Frustration** | ✅ Bisa | Confusion yang tidak terselesaikan → Frustration. Keduanya bisa terlihat bersamaan di frame transisi. | D'Mello 2012: Confusion→Frustration signifikan (*"hopeless confusion"*) |
| **Frustration + Boredom** | ⚠️ Jarang, tapi bisa | Frustrasi berkepanjangan bisa mengarah ke bosan. Kalau keduanya terlihat, biasanya frustrasi lebih jelas. | D'Mello 2012: Frustration→Boredom marginal (*"persistent frustration → disengagement"*) |
| **Frustration + Engagement** | ❓ Tidak ada aturan jelas | Bisa terjadi (masih mencoba tapi frustrasi). Tidak ada paper yang melarang atau mendukung eksplisit. Nilai apa yang terlihat. | D'Mello 2012: Engagement→Frustration "at chance" — tidak ada hubungan kuat |
| **Confusion + Boredom** | ❌ Tidak berkaitan | Kebingungan dan kebosanan tidak saling berhubungan. Jangan assume salah satu menyebabkan yang lain. Nilai berdasarkan ciri masing-masing secara independen. | D'Mello 2012: Confusion→Boredom *"occurred at chance levels"* — tidak ada hubungan |

---

### Tabel Cepat: Apa yang Suppress Apa di Sistem?

Ini yang dilakukan sistem secara otomatis (kode), bukan keputusan penilai manual:

| Mekanisme | Cara kerja | Basis |
|---|---|---|
| **Boredom suppress Engagement** | Kalau Boredom tinggi → skor Engagement dikurangi | DAiSEE "complementary" + D'Mello near-mutually exclusive |
| **Smile gate Confusion** | Kalau senyum sangat kuat (AU12 tinggi) → skor Confusion dibatasi (bukan di-nol-kan, ada floor 30%) | Craig 2008: AU12 non-diskriminatif (muncul di Confusion DAN Frustration) |
| **Gaze gate Engagement** | Kalau lihat jauh dari layar → Engagement berkurang | Whitehill 2014: "looking away from computer" = not engaged |
| **Lihat ke bawah** | TIDAK mengurangi Engagement | Sümer 2021: "head-down = taking notes or reading" — on-task |
| Lainnya (Conf/Frus suppress, dll.) | **TIDAK ADA** — sengaja dihapus | D'Mello: Conf↔Bore "at chance", Conf↔Eng co-occur → suppress tidak berdasar |

---

### Contoh Aplikasi Patokan

**Contoh 1:** Siswa mengerutkan alis (AU4), menatap soal, kepala condong → **Confusion=1, Engagement=1** (productive confusion, bukan salah satu)

**Contoh 2:** Siswa melamun ke samping, mata berat → **Boredom=1, Engagement=0** (bukan keduanya 1)

**Contoh 3:** Alis naik (AU1+AU2), tampak distressed tapi masih menatap layar → **Frustration=1, Engagement=1** (tidak ada aturan melarang, nilai apa yang terlihat)

**Contoh 4:** Siswa jelas bosan (lihat ke atas/samping) tapi juga ada sedikit alis naik → **Boredom=1, Frustration bisa 1** (Frustration→Boredom diakui D'Mello)

---

## Alur Keputusan untuk Penilai

```
Lihat frame →

1. Mata berat / menutup?
   YA → cek Boredom
   
2. Lihat ke layar (kepala relatif tegak)?
   YA → cek Engagement
   TIDAK (menoleh jelas) → kurangi Engagement, tambah kemungkinan Boredom
   
3. Alis bergerak?
   TURUN/mengerut → cek Confusion
   NAIK (terangkat) → cek Frustration
   
4. Ada tanda lain?
   Mulut sedikit terbuka → tambah Confusion
   Tangan di wajah → tambah Confusion atau Frustration
```

---

## Limitasi yang Perlu Diketahui Penilai

Berdasarkan DESIGN_RATIONALE §15 dan §17:

1. **Threshold adalah kalibrasi empiris** — batas 0.45 bukan dari paper, bisa disesuaikan
2. **Gaze dari webcam** lebih kasar dari eye-tracker (Sümer 2021: metode webcam sebagai pendekatan)
3. **Variasi individual** — Bosch (2023) menemukan ekspresi frustrasi sangat bervariasi antar individu
4. **Wajah datar** — siswa yang terlibat bisa terlihat flat/datar (Whitehill: engagement tidak selalu ekspresif)
5. **AU dari MediaPipe blendshape** = perkiraan FACS (ARKit↔FACS), bisa salah pada wajah dari sisi, terlalu jauh, atau terhalang sebagian
6. **Confusion + Engagement bisa co-occur** — ini normal dan diharapkan (*productive confusion*)
