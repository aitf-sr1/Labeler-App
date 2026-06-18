# Basis Akademis: Frame-Difference (Dwi-Aliran) untuk Confusion vs Frustration

Dokumen ini menjelaskan **mengapa** model memakai input dwi-aliran
**appearance + frame-difference** (selisih antar-frame), dan mengapa pendekatan
ini **valid secara FACS** meskipun tidak memakai deteksi landmark MediaPipe.
Semua kutipan di bawah **verbatim** dari paper asli (tervalidasi di
`docs/ACADEMIC_BASIS.md` aplikasi labeler, lengkap dengan tautan PDF + halaman).

## 1. Pembeda Confusion vs Frustration adalah ARAH GERAK alis

Craig 2008 (data mining FACS pada siswa belajar) menemukan pembeda intinya pada alis:

> "AUs 1, 2, and 14 were primarily associated with frustration, but a strong
> association was found for a link between AUs 1 and 2 occurring together.
> [...] Confusion displayed associations with AUs 4, 7, and 12. Action units 4
> and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to
> trigger AU4 (lowered brow)."

— Craig, S.D., D'Mello, S., Witherspoon, A., & Graesser, A. (2008). *Emote aloud
during learning with AutoTutor: Applying the Facial Action Coding System to
cognitive–affective states during learning.* Cognition & Emotion, 22(5), 777–788.

Jadi **Frustration = alis NAIK (AU1+AU2)**, **Confusion = alis TURUN (AU4)**.
Pembedanya adalah **arah** alis, yaitu sebuah **gerakan**, bukan keadaan diam.

## 2. FACS pada hakikatnya mengukur GERAKAN otot (jadi sinyalnya dinamis)

> "The Facial Action Coding System (Ekman & Friesen, 1978) is an objective
> method for quantifying facial movement in terms of component actions."

— Bartlett, M.S., Hager, J.C., Ekman, P., & Sejnowski, T.J. (1999). *Measuring
facial expressions by computer image analysis.* Psychophysiology, 36, 253–263.

Definisi FACS sendiri adalah pengukuran **facial movement** lewat **action**.
Action Unit = aksi/gerakan otot. Maka sinyal yang membedakan Confusion vs
Frustration (arah gerak alis) memang **inherently dinamis**, bukan statis.

## 3. Dinamika gerakan itu penting, khususnya untuk ekspresi SPONTAN

> "Spontaneous facial expressions differ from posed expressions in both which
> muscles are moved, and in the dynamics of the movement."

— Bartlett, M.S., Littlewort, G.C., Frank, M.G., Lainscsek, C., Fasel, I.R., &
Movellan, J.R. (2006). *Automatic Recognition of Facial Actions in Spontaneous
Expressions.* Journal of Multimedia, 1(6), 22–35.

Ekspresi siswa belajar adalah **spontan**. Bartlett 2006 menegaskan ekspresi
spontan dibedakan justru oleh **dynamics of the movement**. Satu frame statis
membuang informasi dinamika ini. **Frame-difference menangkapnya kembali.**

## 4. Sebaliknya, Engagement memang cukup dari penampilan STATIS

> "This accuracy is quite high and suggests that most of the information about
> the appearance of engagement is contained in the static pixels, not the motion
> per se."

— Whitehill, J., Serpell, Z., Lin, Y-C., Foster, A., & Movellan, J.R. (2014).
*The Faces of Engagement: Automatic Recognition of Student Engagement from Facial
Expressions.* IEEE Transactions on Affective Computing.

Engagement terbaca dari pixel statis. Maka model tetap butuh **aliran appearance
(RGB statis)** untuk Engagement (dan Boredom), bukan motion saja.

## 5. Desain yang mengikuti bukti di atas: input dwi-aliran 6-kanal

Menggabungkan poin 1 sampai 4:

| Aliran | Isi | Menangkap | Dasar |
|--------|-----|-----------|-------|
| Appearance | crop RGB ternormalisasi (3 kanal) | Engagement, Boredom (statis) | Whitehill 2014 (static pixels) |
| Motion | frame-difference bertanda (3 kanal) | Confusion vs Frustration (arah gerak alis) | Craig 2008 + Bartlett 2006 (dynamics of movement) |

Input model = konkatenasi keduanya = **6 kanal**. Backbone ConvNeXtV2 di-set
`in_chans=6` (bobot stem pretrained diadaptasi otomatis oleh timm).

## 6. Mengapa frame-difference, bukan AU dari MediaPipe

MediaPipe FaceLandmarker/blendshape **gagal terdeteksi saat ada distorsi kamera**,
sehingga AU eksplisit tidak andal di lapangan. Frame-difference dihitung
**langsung dari pixel** (selisih dua frame), **tidak butuh deteksi landmark**,
sehingga tahan terhadap distorsi.

Penting: ini **tetap "melihat dari AU"** secara sah. FACS mendefinisikan AU
sebagai **gerakan otot** (Bartlett 1999, poin 2), dan selisih antar-frame adalah
**manifestasi pixel-level dari gerakan otot itu** (poin 3, "dynamics of the
movement"). Jadi frame-difference adalah **proxy pixel** untuk dinamika AU, bukan
fitur ad-hoc. Yang ditinggalkan hanya **alat ukur AU yang rapuh (landmark)**,
bukan **konstruk AU-nya**.

## Catatan keterbatasan (jujur)

Dataset sekarang hanya 2 frame per video (posisi 25% dan 75%), jadi
frame-difference yang dipakai adalah selisih dua momen berjarak, bukan gerakan
antar-frame berdekatan. Ini sudah memberi sinyal arah perubahan, tetapi versi
yang lebih bersih menuntut **ekstraksi frame rapat** (mis. tiap 0.2–0.5 detik) di
sekitar momen ekspresif. Itu langkah peningkatan berikutnya, tanpa mengubah
arsitektur dwi-aliran ini.
