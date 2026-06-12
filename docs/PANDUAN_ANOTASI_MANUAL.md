# Panduan Anotasi Manual — Pemeriksaan Hasil Sistem

Dipakai saat **memeriksa hasil pelabelan aplikasi secara manual**. Untuk tiap frame: lihat wajah → cocokkan dengan ciri di tabel → tentukan emosi mana yang aktif.

Semua ciri **verbatim dari paper** (sudah diverifikasi kata-per-kata ke 37 PDF sumber — lihat catatan verifikasi di akhir). Tiap ciri diberi **kolom Kekuatan** supaya kamu tahu mana sinyal yang bisa dipercaya sendirian dan mana yang cuma pendukung.

> **Cara pakai:** Buka video → lihat frame berlabel AI → cocokkan ke tabel emosi → kalau ciri **KUAT** terpenuhi, emosi itu harusnya aktif. Ciri **LEMAH** tidak cukup berdiri sendiri.

---

## Legenda Kekuatan Bukti

| Tanda | Arti | Boleh jadi dasar sendiri? |
|---|---|---|
| **KUAT** | Banyak studi independen / coverage tinggi / sinyal langsung | Ya — kalau ciri ini jelas, label boleh aktif |
| **SEDANG** | Satu studi, atau coverage menengah, atau rantai argumen tak-langsung tapi didukung | **Cukup** kalau ciri **JELAS**; idealnya + 1 cue pendukung kalau ragu |
| **LEMAH** | Sampel kecil / rantai panjang / sulit dilihat / keterbatasan detektor | TIDAK — hanya menguatkan, jangan jadi alasan tunggal |

> **"SEDANG itu sudah cukup atau harus KUAT?"** — **SEDANG sudah cukup** untuk melabeli, asalkan cirinya **terlihat JELAS** (bukan samar). KUAT artinya boleh yakin walau hanya 1 ciri; SEDANG artinya yakin kalau jelas, dan **lebih yakin lagi kalau ada 2 cue**. **Boredom** sengaja tak punya cue KUAT (lihat §1) — jadi untuk Boredom kamu memang mengandalkan cue SEDANG, idealnya 2 sekaligus. LEMAH **tidak pernah** cukup sendiri.

> **Tentang TANGAN (sering ditanya):** semua sinyal tangan **LEMAH untuk deteksi OTOMATIS** — detektor app cuma menghitung *jumlah* tangan dekat wajah, tidak bisa bedakan posisi (dagu vs dahi) atau aktif vs pasif (Mahmoud 2011: tangan menyangga pasif = *relaxed*, bukan berpikir). Tapi **kamu (manusia) bisa melihat posisinya** → di tangan anotator manual, sinyal tangan naik jadi **SEDANG**. Tetap: jangan melabeli hanya karena ada tangan.

---

## Dasar Label Kekuatan — dari Jurnalnya Sendiri (bukan opini)

Label KUAT/SEDANG/LEMAH **bukan tebakan** — ia mengikuti **peringkat yang dibuat jurnalnya sendiri** (terutama ConfusionBench 2026, benchmark *expert-validated* oleh 10 psikolog) dan **angka coverage** dari Craig 2008. Berikut buktinya verbatim:

| Klaim label | Apa kata jurnal (verbatim) | Sumber |
|---|---|---|
| Alis (AU4/AU7) = **KUAT** (tier 1) | *"brow lowering (AU4), eyelid tightening (AU7), and especially their combination (AU4+AU7) as **the most reliable indicators**"* | ConfusionBench 2026, protokol anotasi |
| Tangan = **auxiliary** (tier 4 dari 5) | *"(4) **auxiliary** hand-to-face behaviors, such as chin touching, forehead pressing, and covering the mouth"* | ConfusionBench 2026 |
| Gaze = pendukung | *"gaze direction and head pose **may also provide supportive evidence**"* | ConfusionBench 2026 |
| "1 tangan→confusion, 2 tangan→frustration" itu **inferensi**, bukan klaim langsung | *"one-hand-to-face gestures **may be** associated with less negative affect, while two-hands-to-face gestures **may be** indicative of reduced focus"* — confusion/frustration adalah simpulan KITA, jurnal cuma bilang "thoughtful" & "reduced focus" + korelasi **self-efficacy rendah** (sifat, bukan emosi sesaat) | Grafsgaard 2013b |
| Gestur tangan = pendekatan **spekulatif** | *"**speculative approach** to find the correlation between gestures and context"* · *"it is still **unclear if these states can be detected**"* | Behera 2020 |
| Mulut terbuka = **SEDANG** (1 studi, konteks "berpikir") | *"a **novel form** that has not been **previously reported**"* — komponen paling signifikan untuk *thinking face* (android), bukan confusion-belajar langsung; ConfusionBench bahkan **tidak** memasukkan mulut-terbuka ke kriteria confusion | Namba 2024 |
| Coverage = dasar angka | Craig 2008 Table 2: AU4 **95%**, AU7 **78%**, AU4+AU7 **73%**, AU1+AU2 **100%**, AU43 (boredom) **40%** | Craig 2008 |

**Kesimpulan:** menaikkan tangan/mulut ke **KUAT** justru **bertentangan** dengan jurnalnya — mereka sendiri menyebutnya *auxiliary / may be / speculative*. Jadi LEMAH/SEDANG itu **setia ke sumber**, bukan meremehkan. (Untuk anotasi manual, lihat aturan **"gestalt berpikir"** di §3 & §5 — kombinasi tangan+mulut+gaze boleh jadi pemicu walau tidak satu-satu.)

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

> **Kenapa Boredom "SEDANG semua" (tidak ada KUAT)?** Bukan diremehkan — **jurnalnya memang begitu.** Craig 2008 menemukan AU43 sebagai **satu-satunya** AU signifikan untuk boredom, tapi coverage-nya **cuma 40%** (terendah di Table 2; bandingkan Confusion AU4 **95%**, Frustration AU1+AU2 **100%**) dan *"no association rules between AUs were observed"*. Boredom lebih merupakan **disengagement** (gaze/postur lintas-waktu) daripada satu ekspresi wajah sesaat — D'Mello: *"disengagement (boredom) when they abandon pursuit of the... learning goal"*. Jadi memang tak ada sinyal wajah tunggal yang kuat untuk boredom.
>
> **Cara melabelinya (karena tak ada cue KUAT):**
> - **Paling yakin:** 2 cue SEDANG bareng → **mata berat (AU43) + pandang ke samping/atas**.
> - **Boleh:** 1 cue SEDANG yang **jelas** + konteks (mis. mata lesu droopy yang nyata, berlangsung beberapa saat).
> - **Jangan:** dari 1 cue **samar/ambigu** saja (mata sedikit menyipit bisa AU7/cahaya; nunduk = malah Engagement).

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
| **1 tangan/telunjuk di dagu (AKTIF)** | Telunjuk **menyentuh/mengetuk/mengusap** dagu-pipi (bukan menyangga pasif) | LEMAH (auto) → SEDANG (manual) | Mahmoud 2011: aktif = *thinking/unsure*; **pasif = relaxed** (N=15 kecil) |

**BUKAN Confusion:**
- Alis **naik** → Frustration/terkejut (AU1+AU2, bukan AU4)
- Mata **menutup penuh** → Boredom (AU43, bukan AU7)
- **Tangan + mulut terbuka TAPI alis/sipit NOL & pose berpikir tenang** → **bukan** Confusion (zona "Unsure"); butuh minimal jejak alis/sipit ATAU pose "berjuang" jelas (lihat aturan di bawah)
- Tangan **menyangga kepala pasif** (santai/lelah) → bukan kognitif (Mahmoud: *relaxed*)
- Wajah datar menatap layar → bisa Engagement saja

> **Inti:** jangkar Confusion = **alis berkerut (AU4)**. Mulut terbuka & tangan hanya **menambah**, tidak cukup sendiri.

### Aturan cue lemah (tangan & mulut): kapan CUKUP, kapan TIDAK

Tangan-saja **tidak cukup**, mulut-saja **tidak cukup** — itu posisi jurnal. Pertanyaannya: kalau **tangan + mulut ada tapi alis/sipit (AU4/AU7) NOL**, boleh Confusion atau tidak? Pakai tabel ini:

| Alis turun (AU4) / sipit (AU7) | Ditambah tangan & mulut | Confusion? |
|---|---|---|
| **JELAS** (mengerut/menyipit terlihat) | tidak perlu | **YA — KUAT.** Alis/sipit sudah cukup sendiri. |
| **SAMAR** (sedikit tegang, ragu-ragu) | tangan dagu **aktif** + mulut sedikit buka + tatap soal | **YA — tapi lemah** (gestalt; keputusan anotator). |
| **NOL** (sama sekali tak ada) | hanya tangan + mulut | **TIDAK (default)** — ini zona **"Unsure"**. Baca aturan di bawah. |

**Jawaban untuk kasus "ada tangan, ada mulut, TAPI alis & mata sipit tidak ada":**

- **Default: JANGAN tandai Confusion.** Tangan + mulut hanyalah sinyal **"berpikir"** — dan berpikir **tanpa** tanda *disequilibrium* (alis turun / sipit) belum tentu bingung: bisa berpikir tenang (= **Engagement** kalau on-task), bisa ngobrol/menguap. ConfusionBench 2026 menaruh kasus persis begini di kategori **"Unsure"** (*"additional information is needed for a more confident judgment"*) — **bukan "Yes"** (di datasetnya: 224 Yes vs 46 Unsure).
- **Boleh "lemah-Confusion" HANYA bila ketiganya terpenuhi:**
  1. Tangan **AKTIF** — telunjuk **menyentuh/mengetuk/mengusap** dagu-pipi (Mahmoud 2011: ini *"thinking and unsure"*), **BUKAN** menyangga kepala pasif (Mahmoud 2011: bersandar pasif = *"relaxed mood"*, **bukan** kognitif);
  2. Mulut sedikit terbuka **sambil menatap soal sulit** (bukan menguap/ngobrol);
  3. Kamu benar-benar membaca pose **"sedang berjuang memahami"** (bukan berpikir santai).
  → Kalau ya semua, tandai Confusion tapi anggap **ragu/lemah**.
- Kalau pose tampak **berpikir tenang / santai / pasif** → **Confusion = 0** (kalau menatap layar/soal = Engagement).

> **Ringkas:** **minimal harus ada JEJAK alis/sipit, ATAU pose "berjuang" yang jelas.** Tangan + mulut **tanpa** keduanya = *Unsure*, default **tidak** Confusion. Jangan kejar kuantitas sampel dengan melabeli "thinking tenang" sebagai Confusion — itu menambah false-positive ke dataset.

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

> ✅ **Pengecualian "gestalt berpikir" (aturan lengkap di §3):** **tangan dagu AKTIF + mulut sedikit terbuka + tatap soal** → boleh Confusion **bila ada minimal JEJAK alis/sipit ATAU pose "berjuang" jelas**. **Tapi kalau alis/sipit NOL dan pose berpikir tenang → default TIDAK Confusion** (zona *Unsure*). Tangan menyangga **pasif** = *relaxed*, jangan dihitung.

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
4. Cue LEMAH (menambah, TIDAK jadi alasan tunggal):
     Mulut sedikit terbuka → +Confusion (SEDANG)
     1 tangan dagu AKTIF → +Confusion (LEMAH)   |   2 tangan dahi → +Frustration (SEDANG)
5. KASUS tangan+mulut TAPI alis/sipit NOL?
     pose "berjuang" jelas / tangan aktif → Confusion LEMAH (ragu)
     pose berpikir tenang / tangan pasif  → Confusion = 0  (zona "Unsure"; on-task = Engagement)
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

Seluruh kutipan di panduan ini dan di `ACADEMIC_BASIS.md` telah **dicek kata-per-kata** terhadap 37 PDF paper sumber (`2-anotasi-data/paper/`) memakai ekstraksi teks + pencocokan substring ternormalisasi. **Hasil: semua kutipan akurat.** Perbedaan kecil yang muncul saat pengecekan otomatis semuanya **artefak ekstraksi PDF** (soft-hyphen pemenggalan baris, penanda sitasi inline yang sengaja dihilangkan sesuai konvensi, header halaman nyelip) — bukan kesalahan kutipan. Satu paper (**Bartlett 1999**) adalah PDF hasil **pindai** (tanpa lapisan teks); ia di-**OCR** (EasyOCR, 300 dpi, 11 halaman → 9.701 kata) dan kutipannya **TERKONFIRMASI cocok** (*"…is an objective method for quantifying facial movement in terms of component actions"*). Jadi **tidak ada satu pun kutipan yang tidak terverifikasi.**
