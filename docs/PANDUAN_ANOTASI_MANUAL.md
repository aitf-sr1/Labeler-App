# Panduan Anotasi Manual — Pemeriksaan Hasil Sistem

Dipakai saat **memeriksa hasil pelabelan aplikasi secara manual**. Untuk tiap frame: lihat wajah → cocokkan dengan ciri di tabel → tentukan emosi mana yang aktif.

Semua ciri **verbatim dari paper** (sudah diverifikasi kata-per-kata ke 37 PDF sumber — lihat catatan verifikasi di akhir). Tiap ciri diberi **kolom Kekuatan** supaya kamu tahu mana sinyal yang bisa dipercaya sendirian dan mana yang cuma pendukung.

> **Cara pakai:** Buka video → lihat frame berlabel AI → cocokkan ke tabel emosi → kalau ciri **KUAT** terpenuhi, emosi itu harusnya aktif. Ciri **LEMAH** tidak cukup berdiri sendiri.

---

## Legenda Kekuatan Bukti

| Tanda | Arti | Boleh jadi dasar sendiri? |
|---|---|---|
| **KUAT** | Banyak studi independen / coverage tinggi / sinyal langsung | Ya — kalau ciri ini jelas, label boleh aktif |
| **SEDANG** | Satu studi, atau coverage menengah, atau rantai argumen tak-langsung tapi didukung | Hati-hati — idealnya ada 1 ciri pendukung |
| **LEMAH** | Sampel kecil / rantai panjang / sulit dilihat / keterbatasan detektor | TIDAK — hanya menguatkan, jangan jadi alasan tunggal |

> **Tentang TANGAN (sering ditanya):** semua sinyal tangan **LEMAH untuk deteksi OTOMATIS** — detektor app cuma menghitung *jumlah* tangan dekat wajah, tidak bisa bedakan posisi (dagu vs dahi) atau aktif vs pasif (Mahmoud 2011: tangan menyangga pasif = *relaxed*, bukan berpikir). Tapi **kamu (manusia) bisa melihat posisinya** → di tangan anotator manual, sinyal tangan naik jadi **SEDANG**. Tetap: jangan melabeli hanya karena ada tangan.

---

## SISTEM LABEL

4 label **independen** (boleh lebih dari satu aktif). Boredom & Engagement biasanya **tidak bersamaan**; Confusion & Engagement **boleh bersamaan** (*productive confusion*).

| Label | Aktif = 1 | Tidak aktif = 0 |
|---|---|---|
| **Boredom** | Tanda kebosanan terlihat | Tidak ada |
| **Engagement** | Tanda terlibat/fokus terlihat | Tidak ada |
| **Confusion** | Tanda kebingungan terlihat | Tidak ada |
| **Frustration** | Tanda frustrasi terlihat | Tidak ada |

---

## TABEL INDUK — Semua Sinyal Sekilas (cheat sheet)

| Yang terlihat di wajah | Emosi | Kekuatan | Paper |
|---|---|---|---|
| Alis **turun/mengerut** (di antara alis berkerut) | **Confusion** | KUAT | Craig 2008 (95%), Grafsgaard 2011, ConfusionBench 2026 |
| Kelopak mata **menyipit** (bukan menutup) | **Confusion** | KUAT | Craig 2008 (AU7) |
| Alis turun **+** kelopak menyipit bersama | **Confusion** | KUAT | Craig 2008 (AU4+AU7), ConfusionBench |
| Menatap layar + mata terbuka (wajah "ada") | **Engagement** | KUAT | Whitehill 2014 (κ=0.96) |
| Alis **naik penuh** (dalam + luar terangkat) | **Frustration** | KUAT* | Craig 2008 (100%) — *lihat catatan |
| Mulut sedikit terbuka, "mikir keras" | **Confusion** | SEDANG | Namba 2024 (AU25+AU26) |
| Mata **berat/setengah menutup** (lesu) | **Boredom** | SEDANG | Craig 2008 (AU43, 40%) |
| Menoleh jauh ke **samping / atas** (melamun) | **Boredom** | SEDANG | GazeTutor, Whitehill, Sümer |
| Nunduk ke catatan/keyboard | **Engagement** (bukan Boredom) | SEDANG | Sümer 2021 |
| Alis turun sekunder saat distressed | **Frustration** | SEDANG | Grafsgaard 2013 (AU4) |
| **2 tangan** menekan dahi / menutup wajah | **Frustration** | SEDANG | Grafsgaard 2013b, ConfusionBench |
| **1 tangan/telunjuk** di dagu (kontak ringan) | **Confusion** | LEMAH→SEDANG (manual) | Mahmoud 2011, Behera 2020 |
| Lesung pipit (sudut bibir tertarik) | **Frustration** | LEMAH | Grafsgaard 2013, Bosch 2023 |

> **\*Catatan Frustration:** arah alis **NAIK** (AU1+AU2) punya coverage 100% di Craig 2008, **tapi hanya dari satu studi**. Grafsgaard 2013 (deteksi otomatis, konteks belajar) justru menemukan alis **TURUN** (AU4) berkorelasi positif dengan frustrasi. Praktik aman: **alis naik penuh + wajah distressed** = Frustration; kalau cuma alis turun, pertimbangkan Confusion dulu.

---

## 1. BOREDOM (Kebosanan)

**Definisi (Craig 2008):** *"the state of being weary and restless through lack of interest."* — lelah/malas karena tidak tertarik, khusus situasi belajar.

| Ciri | Yang dilihat | Kekuatan | Catatan |
|---|---|---|---|
| **Mata berat / menutup (AU43)** | Kelopak berat / setengah menutup tanpa sebab fisik, lesu/droopy | SEDANG | Craig: satu-satunya AU signifikan utk boredom, **tapi coverage cuma 40%, 1 studi** |
| **Pandang ke samping / atas** | Kepala/mata menoleh jauh dari layar, melamun, *zoning out* | SEDANG | Konstruk didukung 3 paper; **tapi gaze webcam kasar** |

**BUKAN Boredom:**
- Nunduk membaca/menulis → itu **Engagement** (lihat §2)
- Berkedip normal (terlalu singkat) → bukan AU43
- Menyipit karena cahaya/fokus → itu AU7 (Confusion)
- ~~Menguap~~, ~~gerak-gerak~~ → Craig: "non-significant", **tidak dipakai**

**Tabel arah pandang (sering keliru!):**

| Arah pandang | Label | Alasan |
|---|---|---|
| Lurus ke layar | Engagement | forward gaze = engaged (Whitehill) |
| Ke **samping** | **Boredom** | "looking away from computer" (GazeTutor/Whitehill) |
| Ke **atas** (mendongak) | **Boredom** | zoning out (Whitehill lvl 1) |
| Ke **bawah** (nunduk baca/ketik) | **Engagement** | "head-down = taking notes/reading" (Sümer 2021) |

> ⚠️ Nunduk = **Engagement**, BUKAN Boredom. Hanya **samping & atas** yang Boredom.

---

## 2. ENGAGEMENT (Keterlibatan/Fokus)

**Definisi (D'Mello & Graesser 2012):** *"state of interest that results from involvement in an activity."*

| Ciri | Yang dilihat | Kekuatan | Catatan |
|---|---|---|---|
| **Menatap layar + mata terbuka** | Kepala tegak menghadap layar, mata terbuka normal, "ada" secara kognitif | KUAT | Whitehill: manusia κ=0.96 menilai engaged dari wajah; info ada di "static pixels" |
| **Nunduk ke materi** | Menunduk baca soal / catat / keyboard | SEDANG | Sümer 2021: head-down = on-task, **bukan** boredom |

**Penting:**
- Engagement **tidak perlu** ekspresi dramatis — wajah datar tapi menatap layar = valid (Whitehill).
- **Boleh co-occur dengan Confusion** (*productive confusion*).

**BUKAN Engagement:** kepala menoleh jauh ke samping; mata sangat droopy/menutup; jelas tidak memperhatikan.

---

## 3. CONFUSION (Kebingungan)

**Definisi (D'Mello & Graesser 2012):** *"cognitive disequilibrium... uncertain about what to do next."* — aktif memproses sesuatu yang bertentangan/tidak dipahami (bukan sekadar "tidak tahu").

| Ciri | Yang dilihat | Kekuatan | Catatan |
|---|---|---|---|
| **Alis turun/mengerut (AU4)** | Alis ditarik ke bawah & tengah, galur vertikal di dahi | **KUAT** | Craig 95% + Grafsgaard 2011 (HMM) + ConfusionBench 2026 "most reliable" — **3 studi**, sinyal terkuat sistem |
| **Kelopak menyipit (AU7)** | Kelopak menegang/menyipit, bukan menutup | **KUAT** | Craig 78%; AU7→AU4 |
| **AU4 + AU7 bersama** | Alis turun **dan** kelopak menyipit = "mengernyit fokus" | **KUAT** | Craig 73%; ConfusionBench: kombinasi paling reliable |
| **Mulut sedikit terbuka (AU25+26)** | Bibir/rahang sedikit turun, "mikir keras" jawab soal sulit | SEDANG | Namba 2024 "komponen paling signifikan" thinking face — **1 studi, ambigu** (mangap banyak sebab) |
| **1 tangan/telunjuk di dagu** | Telunjuk menyentuh/mengetuk/mengusap dagu-pipi, kontak ringan | LEMAH (auto) → SEDANG (manual) | Mahmoud 2011 (N=15 kecil) + Behera 2020 (chain sulit→bingung) |

**BUKAN Confusion:**
- Alis **naik** → Frustration/terkejut (AU1+AU2, bukan AU4)
- Mata **menutup penuh** → Boredom (AU43, bukan AU7)
- Mulut mangap **tanpa** alis berkerut → cek konteks dulu (bisa menguap/ngomong)
- Wajah datar menatap layar → bisa Engagement saja

> **Inti:** jangkar Confusion = **alis berkerut (AU4)**. Mulut terbuka & tangan hanya **menambah**, tidak cukup sendiri.

---

## 4. FRUSTRATION (Frustrasi)

**Definisi (Craig 2008):** *"making vain or ineffectual efforts... dissatisfaction arising from unresolved problems."* Muncul saat siswa **stuck, tidak bisa maju** (beda dari Confusion yang masih mencoba).

| Ciri | Yang dilihat | Kekuatan | Catatan |
|---|---|---|---|
| **Alis naik penuh (AU1+AU2)** | Alis dalam **dan** luar terangkat, wajah distressed | KUAT* | Craig 100% coverage — **tapi 1 studi**; Grafsgaard malah soroti AU4 turun |
| **Alis turun sekunder (AU4)** | Bisa muncul bareng alis-naik → "mengernyit marah" | SEDANG | Grafsgaard 2013 (auto, konteks belajar) |
| **2 tangan menekan dahi / menutup wajah** | Telapak menekan dahi/pelipis, facepalm, dua tangan | SEDANG | Grafsgaard 2013b (2-tangan = self-efficacy rendah, "significant") |
| **Lesung pipit (AU14)** | Sudut bibir tertarik, lesung kecil | LEMAH | Halus, sulit dilihat manual |

**BUKAN Frustration:**
- Alis **turun/mengerut** saja → itu Confusion (AU4, bukan AU1/AU2)
- Senyum → bukan frustration
- Wajah datar → tidak cukup sinyal

### Confusion vs Frustration — pembeda kunci

| Aspek | Confusion | Frustration |
|---|---|---|
| **Alis** | **Turun** (AU4, mengerut) | **Naik** (AU1+AU2, terangkat) |
| Kondisi | Masih mencoba memproses | Stuck, tidak bisa maju |
| **Tangan** | **Dagu / telunjuk**, ringan, biasanya 1 tangan | **Dahi ditekan / wajah ditutup**, berat, cenderung 2 tangan |
| Gaze | Cenderung ke layar | Bisa ke mana saja |

---

## 5. Posisi Tangan — Tabel Khusus (jawaban "kalau tangan gimana")

Tangan **tidak pernah** jadi alasan tunggal. Gunakan hanya untuk **menguatkan** setelah ada sinyal wajah. Yang membedakan Confusion vs Frustration adalah **posisi + jumlah tangan** — dan itu hanya bisa **kamu** nilai (detektor otomatis buta posisi).

| Posisi tangan | Condong ke | Kekuatan | Paper |
|---|---|---|---|
| Telunjuk menyentuh/mengetuk **dagu-pipi**, kontak ringan, 1 tangan | **Confusion** ("mikir/unsure") | LEMAH→SEDANG | Mahmoud 2011 (12 thinking + 2 unsure / 15) |
| Menyangga/menyentuh **dagu** sambil menatap soal | **Confusion** | SEDANG (manual) | ConfusionBench 2026 "touching chin = thinking" |
| Menutupi mulut ringan dengan jari | **Confusion** (ragu) | LEMAH | ConfusionBench "covering mouth = hesitation" |
| **Menekan dahi** (telapak/jari menekan dahi/pelipis) | **Frustration** | SEDANG | ConfusionBench "pressing forehead = frustration" |
| **2 tangan** ke wajah / menutup wajah / menopang kepala | **Frustration** | SEDANG | Grafsgaard 2013b (self-efficacy rendah, "significant") |
| Tangan menyangga kepala **pasif, santai** | **Bukan sinyal kognitif** | — | Mahmoud 2011: gestur pasif = *relaxed mood* |

> ⚠️ **Jebakan:** banyak siswa menyangga kepala karena lelah/santai, bukan berpikir. Kalau tidak ada alis berkerut (Confusion) atau alis naik/distressed (Frustration), **jangan** melabeli dari tangan saja.

---

## 6. Co-occurrence — Emosi Mana Boleh Bersamaan

| Kombinasi | Boleh bersamaan? | Kekuatan aturan | Basis |
|---|---|---|---|
| **Boredom + Engagement** | ❌ Hampir tidak | SEDANG | DAiSEE: "engagement low → boredom high & vice-versa" (observasi, ada pengecualian) |
| **Confusion + Engagement** | ✅ Normal | KUAT | D'Mello: *productive confusion*, transisi signifikan |
| **Confusion + Frustration** | ✅ Bisa | KUAT | D'Mello (*hopeless confusion*) + Richey "confrustion" |
| **Frustration + Boredom** | ⚠️ Jarang | SEDANG | D'Mello: persistent frustration → disengagement (marginal) |
| **Frustration + Engagement** | ❓ Tidak ada aturan | LEMAH | D'Mello: "at chance" — nilai apa yang terlihat |
| **Confusion + Boredom** | ❌ Tidak berkaitan | KUAT | D'Mello: "at chance levels" — nilai independen |

### Yang di-suppress otomatis oleh SISTEM (bukan keputusanmu)

| Mekanisme | Cara kerja | Basis |
|---|---|---|
| Boredom suppress Engagement | Boredom tinggi → Engagement dikurangi (kecil, 0.40) | DAiSEE complementary |
| Smile gate Confusion | Senyum sangat kuat → Confusion dibatasi (floor 30%, tidak di-nol) | Craig: AU12 muncul di Conf & Frus (non-diskriminatif) |
| Gaze gate Engagement | Lihat jauh dari layar → Engagement berkurang | Whitehill "looking away" |
| Lihat ke bawah | TIDAK mengurangi Engagement | Sümer "head-down = on-task" |

---

## 7. Alur Keputusan Cepat

```
Lihat frame →
1. Mata berat/menutup?            YA → cek Boredom (SEDANG)
2. Menatap layar, kepala tegak?   YA → cek Engagement (KUAT)
                                  TIDAK (noleh samping/atas) → Boredom
3. Alis bergerak?
     TURUN/mengerut → Confusion (KUAT)
     NAIK penuh + distressed → Frustration (KUAT*)
4. Tanda tambahan (jangan jadi alasan tunggal):
     Mulut sedikit terbuka → +Confusion (SEDANG)
     1 tangan dagu → +Confusion (LEMAH)   |   2 tangan dahi → +Frustration (SEDANG)
```

---

## 8. Limitasi yang Perlu Diingat

1. **Threshold = kalibrasi empiris** (mis. 0.45), bukan dari paper — bisa disesuaikan.
2. **Gaze webcam** lebih kasar dari eye-tracker (Sümer 2021).
3. **Variasi antar-orang** besar untuk frustrasi (Bosch 2023) → sistem pakai baseline netral per-orang.
4. **Wajah datar** bisa tetap Engaged (Whitehill).
5. **AU dari blendshape MediaPipe** = perkiraan FACS — bisa salah pada wajah miring/jauh/terhalang.
6. **Tangan = sinyal terlemah** untuk deteksi otomatis (count-based, buta posisi) — andalkan penilaian manualmu.

---

## Catatan Verifikasi Verbatim

Seluruh kutipan di panduan ini dan di `ACADEMIC_BASIS.md` telah **dicek kata-per-kata** terhadap 37 PDF paper sumber (`2-anotasi-data/paper/`) memakai ekstraksi teks + pencocokan substring ternormalisasi. **Hasil: semua kutipan akurat.** Perbedaan kecil yang muncul saat pengecekan otomatis semuanya **artefak ekstraksi PDF** (soft-hyphen pemenggalan baris, penanda sitasi inline yang sengaja dihilangkan sesuai konvensi, header halaman nyelip) — bukan kesalahan kutipan. Satu paper (Bartlett 1999) adalah PDF hasil pindai (tanpa lapisan teks) sehingga tidak bisa dicek otomatis; kutipannya kanonik dan dibiarkan apa adanya.
