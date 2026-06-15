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
| Alis (AU4/AU7) = **KUAT** (tier 1) | *"brow lowering or frowning (AU4), eyelid tightening or squinting (AU7), and especially their combination (AU4+AU7) as **the most reliable indicators**"* | ConfusionBench 2026, protokol anotasi |
| Tangan = **auxiliary** (tier 4 dari 5) | *"(4) **auxiliary** hand-to-face behaviors, such as chin touching, forehead pressing, and covering the mouth"* | ConfusionBench 2026 |
| Gaze = pendukung | *"gaze direction and head pose **may also provide supportive evidence**"* | ConfusionBench 2026 |
| "1 tangan→confusion, 2 tangan→frustration" itu **inferensi**, bukan klaim langsung | *"one-hand-to-face gestures **may be** associated with less negative affect, while two-hands-to-face gestures **may be** indicative of reduced focus"* — confusion/frustration adalah simpulan KITA, jurnal cuma bilang "thoughtful" & "reduced focus" + korelasi **self-efficacy rendah** (sifat, bukan emosi sesaat) | Grafsgaard 2013b |
| Gestur tangan = pendekatan **spekulatif** | *"**speculative approach** to find the correlation between gestures and context"* · *"it is still **unclear if these states can be detected**"* | Behera 2020 |
| Mulut terbuka = **SEDANG** (1 studi, konteks "berpikir") | *"a **novel form** that has not been **previously reported**"* — komponen paling signifikan untuk *thinking face* (android), bukan confusion-belajar langsung; ConfusionBench bahkan **tidak** memasukkan mulut-terbuka ke kriteria confusion | Namba 2024 |
| Coverage = dasar angka | Craig 2008 Table 2: AU4 **95%**, AU7 **78%**, AU4+AU7 **73%**, AU1+AU2 **100%**, AU43 (boredom) **40%** | Craig 2008 |

**Kesimpulan:** menaikkan tangan/mulut ke **KUAT** justru **bertentangan** dengan jurnalnya — mereka sendiri menyebutnya *auxiliary / may be / speculative*. Jadi LEMAH/SEDANG itu **setia ke sumber**, bukan meremehkan. **Aturan ketat (lihat §3 & §4):** tangan & mulut **hanya MENGUATKAN** label saat sinyal alis sudah ada — **tanpa sinyal alis, tidak pernah memicu Confusion/Frustration**, bahkan kalau tangan + mulut muncul bersamaan.

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

## Cara Memutuskan Saat Ciri "Beririsan" — Tiap Label Punya CUE Sendiri

**Masalah yang sering bikin bingung:** ciri tampak beririsan. Mis. orang bingung **juga** menghadap layar — lalu "hadap layar" itu bukti Engagement atau Confusion? Jawabannya: **"hadap layar" HANYA bukti Engagement.** Confusion dibuktikan oleh **alis**, bukan oleh gaze. Begitu tiap cue dikembalikan ke "pemiliknya", irisannya hilang.

**Aturannya: jawab DUA pertanyaan TERPISAH, masing-masing dari cue-nya sendiri:**

| Pertanyaan | Lihat CUE | Hasil |
|---|---|---|
| **1. ATENSI** — ke mana perhatiannya? | **arah pandang + mata** (gaze) | Engagement / Boredom / **tak tentu** |
| **2. KOGNITIF** — ekspresi alisnya? | **arah alis** | Confusion / Frustration / **tak ada** |

- **Gaze** (hadap layar, mata terbuka/lesu, arah menoleh) → **hanya** menentukan **Engagement/Boredom**.
- **Alis** (turun=Confusion, naik=Frustration) → **hanya** menentukan **Confusion/Frustration**.
- Lalu **gabungkan** jawaban kedua pertanyaan. Tiap pertanyaan paling banyak menghasilkan **satu** label (karena Eng↔Bore & Conf↔Frus saling eksklusif).

**Makanya bisa muncul label tunggal maupun ganda — semuanya konsisten:**

| Gaze (atensi) | Alis (kognitif) | Label akhir |
|---|---|---|
| Hadap layar, mata terbuka | netral | **Engagement saja** |
| Hadap layar, mata terbuka | **turun/berkerut** | **Engagement + Confusion** (productive confusion) |
| Hadap layar, mata terbuka | **naik/tertekan** | **Engagement + Frustration** |
| Menoleh ke samping/atas, mata lesu | netral | **Boredom saja** |
| **Tak tentu** (gaze ambigu) | **turun/berkerut** | **Confusion saja** |
| **Tak tentu** | **naik/tertekan** | **Frustration saja** |

> Jadi **"Confusion saja"** terjadi saat **alis jelas berkerut TAPI atensinya tak bisa ditentukan** (gaze ambigu) — itu sebabnya ia langka. **"Engagement saja"** = hadap layar + **alis netral**. Tidak ada yang beririsan kalau tiap cue dikembalikan ke pertanyaannya.

**Irisan ASLI yang perlu hati-hati (jujur, ini memang mirip):**
- **Mata menyipit (AU7, Confusion) vs mata lesu/setengah menutup (AU43, Boredom)** — sama-sama di mata. Bedakan: **menegang/menyipit aktif** (usaha kognitif → Confusion) vs **turun/lemas pasif** (mengantuk → Boredom).
- **Senyum (AU12)** muncul di Confusion **dan** Frustration → **jangan** pakai senyum untuk memutuskan keduanya.
- **Menunduk** = bukan Boredom (itu baca/catat = Engagement); hanya **samping/atas** yang Boredom.

**Dasar jurnalnya:** tiap label memang punya **kanal ukur** berbeda di literatur — Engagement/atensi dari **gaze/penampilan** (Whitehill, GazeTutor, Sümer); Confusion/Frustration dari **AU alis** (Craig). Karena kanalnya beda, keduanya **diputuskan terpisah lalu digabung** — bukan diadu di cue yang sama. (Eng↔Bore & Conf↔Frus dipaksa eksklusif karena tiap pasangan **berebut cue yang sama**: gaze untuk yang pertama, arah alis untuk yang kedua.)

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
| Menunduk ke catatan/keyboard | **Engagement** (bukan Boredom) | SEDANG | Sümer 2021 |
| Alis turun sekunder saat tertekan | **Frustration** | SEDANG | Grafsgaard 2013 (AU4) |
| **2 tangan** menekan dahi / menutup wajah | **Frustration** | SEDANG | Grafsgaard 2013b, ConfusionBench |
| **1 tangan/telunjuk** di dagu (kontak ringan) | **Confusion** | LEMAH→SEDANG (manual) | Mahmoud 2011, Behera 2020 |
| Lesung pipit (sudut bibir tertarik) | **Frustration** | LEMAH | Grafsgaard 2013, Bosch 2023 |

> **\*Catatan Frustration:** arah alis **NAIK** (AU1+AU2) punya coverage 100% di Craig 2008, **tapi hanya dari satu studi**. Grafsgaard 2013 (deteksi otomatis, konteks belajar) justru menemukan alis **TURUN** (AU4) berkorelasi positif dengan frustrasi. Praktik aman: **alis naik penuh + wajah tertekan** = Frustration; kalau cuma alis turun, pertimbangkan Confusion dulu.

---

## 1. BOREDOM (Kebosanan)

**Definisi (Craig 2008):** *"the state of being weary and restless through lack of interest."* — lelah/malas karena tidak tertarik, khusus situasi belajar.

| Ciri | Tanda fisik yang terlihat | Kekuatan | Catatan |
|---|---|---|---|
| **Mata berat / setengah menutup (AU43)** | Kelopak atas **turun/sayu**, mata setengah tertutup tanpa sebab fisik, tatapan **kosong/mengantuk** | SEDANG | Craig: satu-satunya AU signifikan untuk boredom, **tapi coverage 40%, 1 studi** |
| **Pandangan ke samping / atas** | Kepala atau mata **menoleh menjauh** dari layar, melamun ke samping/atas | SEDANG | Konstruk didukung 3 paper; pengukuran lewat webcam kasar |

> **Kenapa Boredom "SEDANG semua" (tidak ada KUAT)?** Bukan diremehkan — **jurnalnya memang begitu.** Craig 2008 menemukan AU43 sebagai **satu-satunya** AU signifikan untuk boredom, tapi coverage-nya **cuma 40%** (terendah di Table 2; bandingkan Confusion AU4 **95%**, Frustration AU1+AU2 **100%**) dan *"no association rules between AUs were observed"*. Boredom lebih merupakan **disengagement** (gaze/postur lintas-waktu) daripada satu ekspresi wajah sesaat — D'Mello: *"disengagement (boredom) when they abandon pursuit of the... learning goal"*. Jadi memang tak ada sinyal wajah tunggal yang kuat untuk boredom.
>
> **Cara melabelinya (karena tak ada cue KUAT):**
> - **Paling yakin:** 2 cue SEDANG bareng → **mata berat (AU43) + pandang ke samping/atas**.
> - **Boleh:** 1 cue SEDANG yang **jelas** + konteks (mis. mata lesu sayu yang nyata, berlangsung beberapa saat).
> - **Jangan:** dari 1 cue **samar/ambigu** saja (mata sedikit menyipit bisa AU7/cahaya; menunduk = malah Engagement).

**BUKAN Boredom:**
- Menunduk membaca/menulis → itu **Engagement** (lihat §2)
- Berkedip normal (terlalu singkat) → bukan AU43
- Menyipit karena cahaya/fokus → itu AU7 (Confusion)
- ~~Menguap~~, ~~gerak-gerak~~ → Craig: "nonsignificant", **tidak dipakai**

**Tabel arah pandang (sering keliru!):**

| Arah pandang | Label | Alasan |
|---|---|---|
| Lurus ke layar | Engagement | forward gaze = engaged (Whitehill) |
| Ke **samping** | **Boredom** | "looking away from computer" (GazeTutor/Whitehill) |
| Ke **atas** (mendongak) | **Boredom** | melamun (Whitehill lvl 1) |
| Ke **bawah** (menunduk baca/ketik) | **Engagement** | menunduk = baca/catat = on-task (Sümer 2021) |

> ⚠️ Menunduk = **Engagement**, BUKAN Boredom. Hanya **samping & atas** yang Boredom.

---

## 2. ENGAGEMENT (Keterlibatan/Fokus)

**Definisi (D'Mello & Graesser 2012):** *"state of interest that results from involvement in an activity."*

| Ciri | Tanda fisik yang terlihat | Kekuatan | Catatan |
|---|---|---|---|
| **Menatap layar + mata terbuka** | Kepala **tegak menghadap layar**, mata **terbuka normal**, tampak hadir/fokus | KUAT | Whitehill: manusia κ=0.96 menilai engaged dari wajah |
| **Menunduk ke materi** | Kepala **menunduk** ke arah soal/catatan/keyboard (membaca atau menulis) | SEDANG | Sümer 2021: menunduk = on-task, **bukan** boredom |

**Penting:**
- Engagement **tidak perlu** ekspresi dramatis — wajah datar tapi menatap layar = valid (Whitehill).
- **Boleh co-occur dengan Confusion** (*productive confusion*).

**BUKAN Engagement:** kepala menoleh jauh ke samping; mata sangat sayu/menutup; jelas tidak memperhatikan.

---

## 3. CONFUSION (Kebingungan)

**Definisi (D'Mello & Graesser 2012):** *"cognitive disequilibrium... uncertain about what to do next."* — aktif memproses sesuatu yang bertentangan/tidak dipahami (bukan sekadar "tidak tahu").

| Ciri | Tanda fisik yang terlihat | Kekuatan | Catatan |
|---|---|---|---|
| **Alis turun/mengerut (AU4)** | Kedua alis ditarik **ke bawah dan merapat ke tengah**; **bila ada**, muncul **kerutan VERTIKAL** (garis tegak seperti "11") di antara alis (kerutan boleh tak muncul — yang wajib alisnya turun-merapat) | **KUAT** | Craig 95% + Grafsgaard 2011 + ConfusionBench 2026 "most reliable" — 3 studi, sinyal terkuat |
| **Kelopak menyipit (AU7)** | Mata **sedikit menyipit/menegang** (lebih kecil dari biasa), kelopak bawah terangkat; **mata tetap terbuka**, bukan menutup | **KUAT** | Craig 78% |
| **AU4 + AU7 bersama** | Alis turun **dan** mata menyipit sekaligus = ekspresi **"mengernyit memikirkan sesuatu"** | **KUAT** | Craig 73%; ConfusionBench: kombinasi paling andal |
| **Mulut sedikit terbuka (AU25+26)** | Bibir sedikit membuka / rahang sedikit turun, seperti **sedang berpikir keras** menjawab soal sulit | SEDANG (penguat) | Namba 2024 — 1 studi, ambigu (mulut terbuka banyak sebab) |
| **Tangan di dagu (aktif)** | Jari/telunjuk **menyentuh, mengetuk, atau mengusap** dagu–pipi (bukan kepala disangga pasif) | LEMAH→SEDANG (penguat) | Mahmoud 2011: aktif = berpikir; pasif = santai (N=15) |

**BUKAN Confusion:**
- Alis **naik** (kerutan mendatar di dahi) → Frustration/terkejut, bukan Confusion
- Mata **menutup penuh** → Boredom (AU43, bukan AU7)
- **Tangan + mulut terbuka tetapi alis netral** (tidak turun, tidak menyipit) → **bukan** Confusion; tangan & mulut hanya penguat, **wajib ada sinyal alis**
- Tangan **menyangga kepala secara pasif** (santai/lelah) → bukan tanda berpikir
- Wajah datar menatap layar → cenderung Engagement saja

> **Inti:** jangkar Confusion = **alis berkerut (AU4)**. Mulut terbuka & tangan hanya **menambah**, tidak cukup sendiri.

### ATURAN KETAT — Confusion WAJIB ada sinyal alis

**Confusion baru boleh aktif kalau ada sinyal ALIS:** alis **turun/mengerut** (AU4) **atau** mata **menyipit** (AU7). Tanpa salah satu itu → **BUKAN Confusion**, walau ada tangan dan mulut sekaligus.

| Yang terlihat | Confusion? |
|---|---|
| Alis turun (AU4) / mata sipit (AU7) **terlihat** | **YA** (tangan/mulut tinggal menguatkan) |
| Alis netral — **hanya** tangan / **hanya** mulut / **tangan + mulut** | **TIDAK** — zona *Unsure* (default = 0) |
| Alis **naik** (bukan turun) | bukan Confusion → cek **Frustration** |

**Kenapa ketat begini:**
- Jurnal menaruh alis (AU4/AU7) sebagai *"most reliable"*, tangan/mulut cuma *"auxiliary"* (ConfusionBench). Tanpa sinyal andal (alis), cue pendukung saja = kategori *"Unsure"* (*"additional information is needed"*) — **bukan "Yes"**.
- Tangan + mulut = sinyal **"berpikir"** umum; berpikir **tenang** tanpa *disequilibrium* bisa jadi **Engagement** (kalau on-task), menguap, atau berbicara → mudah jadi false-positive.

**Peran tangan & mulut (tetap berguna, tapi sebatas pendukung):**
- Mereka **menaikkan keyakinan** saat alis **sudah** ada (mis. alis berkerut **+** tangan di dagu = makin yakin Confusion).
- Mereka **tidak pernah** memicu label sendiri — bahkan gabungan tangan+mulut tanpa alis = **tidak** Confusion.
- Tangan **pasif** (menyangga kepala santai) = *relaxed*, **jangan** dihitung sama sekali.

> **Ringkas: tidak ada alis (turun/sipit) = tidak ada Confusion.** Tangan/mulut hanya bumbu, bukan bahan utama.

---

## 4. FRUSTRATION (Frustrasi)

**Definisi (Craig 2008):** *"making vain or ineffectual efforts... dissatisfaction arising from unresolved problems."* Muncul saat siswa **buntu, tidak bisa maju** (beda dari Confusion yang masih mencoba).

| Ciri | Tanda fisik yang terlihat | Kekuatan | Catatan |
|---|---|---|---|
| **Alis naik penuh (AU1+AU2)** | Kedua alis (bagian dalam **dan** luar) **terangkat ke atas**; **bila kulit tak terlalu kencang** muncul **kerutan MENDATAR** di dahi (kerutan **boleh tak muncul** — yang wajib alisnya terangkat); wajah tampak **tertekan/kesal**. *(Mata melotot saja ≠ frustrasi — itu AU5/terkejut.)* | KUAT* | Craig 100% — 1 studi; Grafsgaard soroti AU4 turun |
| **Alis turun + wajah tertekan (AU4 sekunder)** | Alis turun tetapi disertai ekspresi **kesal/mentok** (bukan sekadar berpikir) | SEDANG | Grafsgaard 2013 (deteksi otomatis, konteks belajar) |
| **2 tangan menekan dahi / menutup wajah** | Telapak/jari **menekan dahi atau pelipis**, menutup wajah, atau menopang kepala dengan **dua tangan** | SEDANG (penguat) | Grafsgaard 2013b (2 tangan = self-efficacy rendah) |
| **Lekuk di sudut bibir (AU14)** | Sudut bibir **tertarik ke samping-dalam**, membentuk lekuk/lesung kecil | LEMAH | Halus, sulit dilihat manual |

### ATURAN KETAT — Frustration WAJIB ada sinyal alis (atau pose tertekan jelas)

**Frustration baru boleh aktif kalau ada:** alis **naik penuh (AU1+AU2)** — ini patokan utama — **atau** alis **turun + wajah jelas tertekan/buntu** (AU4 sekunder, Grafsgaard). Tanpa salah satunya → **BUKAN Frustration**, walau ada 2 tangan menekan dahi.

| Yang terlihat | Frustration? |
|---|---|
| Alis **naik penuh** + tertekan | **YA** (patokan utama) |
| Alis **turun** + jelas **buntu/menyerah/tertekan** | **YA** (sekunder; kalau cuma alis turun tanpa tertekan → cek Confusion dulu) |
| Alis netral — **hanya** tangan (2-tangan/dahi) / lesung pipit | **TIDAK** — pendukung saja, tak cukup sendiri |

**BUKAN Frustration:**
- Alis **turun/mengerut** saja (tanpa tertekan) → itu **Confusion** (AU4, bukan AU1/AU2)
- **2 tangan menekan dahi** tapi alis netral & wajah tenang → **bukan** Frustration (cuma pendukung)
- **Senyum lebar tulus** (mata ikut berkerut) → bukan Frustration (lihat "Kasus Senyum"); tapi **senyum kaku + alis naik tertekan** → tetap Frustration
- Wajah datar → tidak cukup sinyal

> Tangan (2-tangan/dahi) & lesung pipit = **penguat saja**, persis seperti tangan/mulut di Confusion. Tanpa sinyal alis / pose tertekan → tidak memicu label.

### Confusion vs Frustration — pembeda kunci

| Aspek | Confusion | Frustration |
|---|---|---|
| **Arah alis** | **Turun & merapat** ke tengah (AU4) | **Naik/terangkat** ke atas (AU1+AU2) |
| **Kerutan dahi (penanda bantu, bisa TAK muncul)** | garis kerut **VERTIKAL** di antara kedua alis (seperti angka "11") | garis kerut **HORIZONTAL/mendatar** melintang dahi |
| **Mata** | sedikit **menyipit/menegang** (AU7) | **bukan** patokan — yang naik **alisnya**, *bukan* mata melotot (melotot = AU5/terkejut) |
| **Kondisi** | masih mencoba memahami | mentok / menyerah / kesal (tampak tertekan) |
| **Tangan** | menyentuh **dagu**, ringan, biasanya 1 tangan | **menekan dahi** / menutup wajah, biasanya 2 tangan |
| **Arah pandang** | cenderung ke layar/soal | bisa ke mana saja |

> 💡 **Trik cepat:** lihat **gerak ALIS** dulu (turun→Confusion, naik→Frustration). Kerutan dahi cuma **penanda bantu**: tegak antar-alis→Confusion, mendatar di dahi→Frustration.
>
> ⚠️ **Penting (jawaban kasus "alis naik tapi dahi tak berkerut" / "mata terbuka lebar"):**
> - **Kerutan dahi BOLEH tidak muncul** (kulit kencang/muda, atau alis terangkat tipis). Yang **wajib** dinilai adalah **posisi alisnya terangkat**, bukan ada-tidaknya kerutan. Jangan batalkan Frustration cuma karena dahi mulus.
> - **Mata terbuka lebar SAJA ≠ Frustration.** Mata melotot = **AU5 (Upper Lid Raiser)** — sinyal **terkejut/waspada**, **bukan** bagian sinyal frustrasi Craig (yang frustrasi itu **AU1+AU2 = alis terangkat**, bukan mata membesar). Frustration butuh **alis benar-benar terangkat (dalam+luar) + kesan tertekan/kesal**; kalau cuma mata lebar tanpa alis naik & tanpa kesan tertekan → kemungkinan **terkejut/netral**, bukan Frustration.
>
> *Dasar: morfologi AU standar FACS (Ekman & Friesen 1978 — sumber definisi AU yang dipakai Craig & Bartlett). AU4 (brow lowerer)→kerut tegak; AU1+AU2 (brow raiser)→kerut mendatar; AU5 (upper lid raiser)=mata membesar (terkejut), terpisah dari frustrasi. Deskripsi morfologi standar, bukan kutipan dari 37 paper.*

### Kasus Senyum (AU12) bersama alis naik/turun

**Senyum tidak menentukan** Confusion vs Frustration — ia muncul di **kedua**-nya (Craig 2008: *"AUs 12 and 14 that occur during expressions of both confusion and frustration"*). Jadi **alis tetap penentu**, tetapi **jenis senyum** mengubah keputusan:

| Yang terlihat | Putusan |
|---|---|
| **Senyum tulus/lebar** (pipi terangkat + **sudut mata ikut berkerut**) — walau alis sedikit naik/turun | **BUKAN** Confusion/Frustration. Senyum tulus = *exclusion cue* (ConfusionBench: *"smiling as an exclusion cue in most cases"*). Kalau menatap layar → **Engagement**. |
| **Senyum tipis/kaku/menyeringai** (hanya sudut bibir, **mata datar/tegang**) + **alis turun** (kerutan tegak) | **Confusion** — alis menang; senyumnya hanya menutupi |
| **Senyum tipis/kaku** + **alis naik + wajah tertekan** (kerutan mendatar) | **Frustration** — alis menang; senyum getir/kesal |

**Cara membedakan senyum:**
- **Tulus** → pipi naik **dan** ada kerutan di **sudut mata** (mata ikut tersenyum).
- **Kaku/sosial** → **hanya** sudut bibir tertarik, mata datar atau tegang.

> Sistem juga **membatasi** skor Confusion saat senyum **sangat kuat** (tidak di-nol-kan, ada batas bawah) — sebab senyum lebar lebih ke "menikmati", bukan bingung. Intinya: **senyum lebar tulus → jangan Confusion/Frustration; senyum kaku + alis bergerak → ikuti arah alis.**

---

## 5. Posisi Tangan — Tabel Khusus (jawaban "kalau tangan gimana")

Tangan **tidak pernah** jadi alasan tunggal. Gunakan hanya untuk **menguatkan** setelah ada sinyal wajah. Yang membedakan Confusion vs Frustration adalah **posisi + jumlah tangan** — dan itu hanya bisa **kamu** nilai (detektor otomatis buta posisi).

| Posisi tangan | Condong ke | Kekuatan | Paper |
|---|---|---|---|
| Telunjuk menyentuh/mengetuk **dagu-pipi**, kontak ringan, 1 tangan | **Confusion** ("mikir/unsure") | LEMAH→SEDANG | Mahmoud 2011 (12 thinking + 2 unsure / 15) |
| Menyangga/menyentuh **dagu** sambil menatap soal | **Confusion** | SEDANG (manual) | ConfusionBench 2026: chin-touch → thinking |
| Menutupi mulut ringan dengan jari | **Confusion** (ragu) | LEMAH | ConfusionBench: cover-mouth → hesitation |
| **Menekan dahi** (telapak/jari menekan dahi/pelipis) | **Frustration** | SEDANG | ConfusionBench: press-forehead → frustration |
| **2 tangan** ke wajah / menutup wajah / menopang kepala | **Frustration** | SEDANG | Grafsgaard 2013b (self-efficacy rendah, "significant") |
| Tangan menyangga kepala **pasif, santai** | **Bukan sinyal kognitif** | — | Mahmoud 2011: gestur pasif = *relaxed mood* |

> ⚠️ **Jebakan:** banyak siswa menyangga kepala karena lelah/santai, bukan berpikir. Tanpa sinyal alis (turun untuk Confusion, naik untuk Frustration), **jangan** melabeli dari tangan saja.

> 🔒 **Aturan ketat:** tangan **tidak pernah** memicu label sendiri — bahkan tangan + mulut bersamaan. Tangan hanya **menguatkan keyakinan** ketika sinyal alis **sudah** ada. Tanpa alis → label tetap 0. Tangan yang menyangga kepala secara **pasif** (santai) tidak dihitung sama sekali.

---

## 6. Co-occurrence — Emosi Mana Boleh Bersamaan

| Kombinasi | Boleh bersamaan? | Kekuatan aturan | Basis |
|---|---|---|---|
| **Boredom + Engagement** | ❌ Hampir tidak | SEDANG | DAiSEE: *"When engagement is low, boredom is generally high and vice-versa"* (observasi, ada pengecualian) |
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
| Lihat ke bawah | TIDAK mengurangi Engagement | Sümer 2021: menunduk = on-task |

---

## 7. Alur Keputusan Cepat

```
Lihat frame →
1. Mata berat/menutup?              YA → cek Boredom (SEDANG)
2. Menatap layar, kepala tegak?     YA → cek Engagement (KUAT)
                                    TIDAK (menoleh samping/atas) → Boredom
3. Lihat ALIS (penentu Confusion/Frustration):
     TURUN/mengerut, kerutan VERTIKAL di antara alis        → Confusion (KUAT)
     NAIK, kerutan HORIZONTAL di dahi, wajah tertekan        → Frustration (KUAT*)
     NETRAL (tidak turun, tidak naik)                        → BUKAN Confusion/Frustration
4. Cue penguat (HANYA menambah keyakinan; tak memicu sendiri):
     Mulut sedikit terbuka  → +Confusion
     Tangan di dagu (aktif) → +Confusion   |   2 tangan menekan dahi → +Frustration
5. Tangan/mulut ada TAPI alis netral?  → label kognitif = 0 (kalau menatap layar: Engagement).
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

Seluruh kutipan di panduan ini dan di `ACADEMIC_BASIS.md` telah **dicek kata-per-kata** terhadap 37 PDF paper sumber (`2-anotasi-data/paper/`) memakai ekstraksi teks + pencocokan substring ternormalisasi (semua teks paper diekstrak ke satu folder dulu, lalu tiap span dalam tanda kutip dicocokkan ke seluruh korpus). **Hasil: semua kutipan akurat.** Pemeriksaan menyeluruh menemukan & memperbaiki beberapa slip kecil (mis. *"looking away from **the** computer"* → *"…from computer"*; *"uncertainty"* → *"uncertain"*; ejaan *"nonsignificant"*; pemulihan *"or frowning/or squinting"* pada kutipan ConfusionBench) — parafrase di sel tabel kini **tidak** lagi diberi tanda kutip agar tak rancu dengan verbatim. Perbedaan kecil yang muncul saat pengecekan otomatis semuanya **artefak ekstraksi PDF** (soft-hyphen pemenggalan baris, penanda sitasi inline yang sengaja dihilangkan sesuai konvensi, header halaman nyelip) — bukan kesalahan kutipan. Satu paper (**Bartlett 1999**) adalah PDF hasil **pindai** (tanpa lapisan teks); ia di-**OCR** (EasyOCR, 300 dpi, 11 halaman → 9.701 kata) dan kutipannya **TERKONFIRMASI cocok** (*"…is an objective method for quantifying facial movement in terms of component actions"*). Jadi **tidak ada satu pun kutipan yang tidak terverifikasi.**
