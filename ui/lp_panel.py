"""
ui/lp_panel.py — Panel "LP Transform" (augmentasi ekspresi LivePortrait di dalam aplikasi).

Alur pemakaian (urut dari atas ke bawah di panel):
  1. SUMBER         : frame wajah netral yang akan diubah ekspresinya (ditandai
                      "LP Transform" di galeri). Tombol Prev/Next pindah antar tanda.
  2. FOLDER DRIVING : taruh video acuan ekspresi di satu folder. Nama file menentukan
                      emosi + urutan: confuse1.mp4, frustration1.mp4, boredom1.mp4, dst.
                      Per emosi bisa pilih "Semua" atau satu video tertentu.
  3. EMOSI          : pilih emosi mana yang diproses (boleh lebih dari satu).
  4. PROSES         : "Proses Frame Ini" (1 frame, untuk dicoba & dipilih frame hasilnya),
                      atau "Batch Semua" (semua frame bertanda, dengan SATU video driving).
  5. PRATINJAU      : 3 gambar besar — SUMBER | DRIVING | HASIL — bisa diunduh satu-satu.
  6. PILIH FRAME    : geser slider video hasil, tandai frame mana yang mau diambil.
                      Frame index yang sama dipakai juga saat Batch (ekspresi konsisten).
  7. TINJAU & LABEL : periksa semua gambar hasil (besar), beri/cek label (AI atau manual),
                      tolak yang jelek (klik 2x), lalu Merge ke dataset (bisa di-Undo).

Catatan teknis penting:
  - Semua frame video DI-DECODE ke memori (list array BGR). TIDAK ada cv2.VideoCapture
    yang dibaca dari banyak thread sekaligus → mencegah crash libavcodec async_lock
    sekaligus membuat geser slider mulus tanpa lag.
  - Overlay landmark (viz) hanya dihitung saat saklar "Viz" di atas aktif, dan dengan
    penundaan singkat (debounce) agar menggeser slider tetap ringan.
"""

import os
import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from .constants import LABELS, LABEL_COLORS

WARNA_LP = "#f59e0b"            # amber — warna khas fitur LP Transform
UKURAN_PRATINJAU = 268         # sisi kotak gambar pratinjau (3 berjajar; pas di lebar panel)
UKURAN_THUMBNAIL = 150         # sisi thumbnail di grid tinjau
UKURAN_PEMERIKSA = 340         # sisi gambar BESAR di pemeriksa (untuk cek detail)
MAX_FRAME_DECODE = 600         # batas jumlah frame yang dimuat ke memori


# ── Pembantu tata letak ─────────────────────────────────────────────────────────

def _judul_seksi(induk, teks: str):
    """Buat judul seksi dengan garis pemisah horizontal di kanannya."""
    baris = ctk.CTkFrame(induk, fg_color="transparent")
    baris.pack(fill="x", padx=12, pady=(12, 4))
    ctk.CTkLabel(baris, text=teks, font=("Poppins", 10, "bold"),
                 text_color=("#374151", "#c4c4d4")).pack(side="left")
    ctk.CTkFrame(baris, fg_color=("#d1d5db", "#2a2a3a"), height=1).pack(
        side="left", fill="x", expand=True, padx=(8, 0))


def _aktifkan_scroll_mouse(area_scroll):
    """Aktifkan scroll roda mouse saat kursor berada di atas area scrollable."""
    kanvas = area_scroll._parent_canvas

    def _masuk(_):
        area_scroll.bind_all("<MouseWheel>",
                             lambda e: kanvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        area_scroll.bind_all("<Button-4>", lambda _e: kanvas.yview_scroll(-1, "units"))
        area_scroll.bind_all("<Button-5>", lambda _e: kanvas.yview_scroll(1, "units"))

    def _keluar(_):
        area_scroll.unbind_all("<MouseWheel>")
        area_scroll.unbind_all("<Button-4>")
        area_scroll.unbind_all("<Button-5>")

    area_scroll.bind("<Enter>", _masuk)
    area_scroll.bind("<Leave>", _keluar)


def _letterbox(bgr, lebar: int, tinggi: int):
    """Resize dengan RASIO DIJAGA lalu beri latar hitam (letterbox) sampai lebar x tinggi.

    Dipakai semua tampilan gambar/video di app supaya tidak gepeng/melar. Murni untuk
    tampilan — pemrosesan (crop wajah, LivePortrait) selalu membaca file asli.
    """
    import cv2
    import numpy as np
    h, w = bgr.shape[:2]
    skala = min(lebar / w, tinggi / h)
    baru_w, baru_h = max(1, int(w * skala)), max(1, int(h * skala))
    kecil = cv2.resize(bgr, (baru_w, baru_h),
                       interpolation=cv2.INTER_AREA if skala < 1 else cv2.INTER_LINEAR)
    kanvas = np.zeros((tinggi, lebar, 3), dtype=bgr.dtype)
    y = (tinggi - baru_h) // 2
    x = (lebar - baru_w) // 2
    kanvas[y:y + baru_h, x:x + baru_w] = kecil
    return kanvas


def _decode_video(path: str, batas_frame: int = MAX_FRAME_DECODE) -> list:
    """Baca SELURUH frame video ke list array BGR (satu thread saja).

    Memuat ke memori menghindari pemakaian cv2.VideoCapture lintas-thread (penyebab
    crash libavcodec) dan membuat geser slider menjadi instan.
    """
    import cv2
    frames = []
    if not path or not os.path.exists(path):
        return frames
    cap = cv2.VideoCapture(path)
    try:
        while len(frames) < batas_frame:
            berhasil, frame = cap.read()
            if not berhasil or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


# ── Panel utama ──────────────────────────────────────────────────────────────────

class LPPanel:
    """Panel LP Transform; dipasang di dalam wadah `_lp_container` pada LeftPanel.

    Hanya berisi tampilan + state ringan. Semua pekerjaan berat (menjalankan
    LivePortrait, melabel dengan AI, memuat thumbnail) dilakukan oleh app.py di
    thread latar, lalu hasilnya dikirim balik ke panel ini.
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app    = app
        self._sudah_dibangun = False

        # --- Sumber (frame netral yang diubah) ---
        self.sumber_bgr   = None
        self.sumber_uuid  = ""
        self.sumber_frame = 0

        # --- Frame hasil decode (di memori, bukan VideoCapture) ---
        self.frame_driving = []      # list BGR untuk pratinjau driving
        self.emosi_driving = ""
        self.frame_hasil   = []      # list BGR hasil LivePortrait (resolusi penuh)
        self.emosi_hasil   = ""
        self.index_hasil   = 0
        self.frame_terpilih = set()  # index frame hasil yang ditandai (absolut, video saat ini)
        # "Template" posisi tertanda sebagai FRAKSI 0-1 → dipakai memetakan ulang ke video
        # driving lain (jumlah sama, posisi proporsional). var_kunci_posisi dibuat di build().
        self.fraksi_terpilih = []
        self.var_kunci_posisi = None
        # Tanda frame DISIMPAN PER VIDEO DRIVING (tiap referensi beda pose/timing):
        # ganti video di dropdown → tanda untuk video itu dipulihkan otomatis.
        self.tanda_per_driving = {}    # {path_driving: [index, ...]}
        self.driving_aktif     = ""    # path driving yang sedang dipreview/dipakai

        # Gambar BGR yang sedang tampil di tiap kolom pratinjau (untuk tombol Unduh)
        self.bgr_tampil = [None, None, None]   # 0=sumber, 1=driving, 2=hasil

        # Indikator loading (animasi titik saat proses berjalan, biar tahu tidak hang)
        self._loading_aktif = False
        self._loading_teks  = ""
        self._loading_tick  = 0

        # --- Dataset wajah baru (tiap foto = 1 orang baru) ---
        folder_wajah = os.getenv("LP_FACES_DIR", "")
        self._folder_wajah_awal = folder_wajah
        self.var_folder_wajah = None               # tk.StringVar (dibuat di build)
        self.var_merge_mode   = None               # komposisi dataset: base|lp|lp_new
        self.wajah_paths      = []                 # semua path gambar di folder
        self.wajah_terpilih   = set()              # path gambar yang dipilih untuk diproses
        self.grid_wajah       = None
        self._ref_thumb_wajah = []
        self._thumb_wajah_cache = {}               # {path: thumb_bgr} agar toggle tak decode ulang
        self.label_ringkasan_wajah = None          # ringkasan setelan dekat tombol proses wajah

        # Chip label AI (read-only) di pemeriksa hasil
        self.chip_ai_periksa = {}

        # --- Folder driving ---
        folder_default = os.getenv(
            "LP_DRIVING_DIR",
            os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "4-Create", "refrensi")))
        self._folder_driving_awal = folder_default
        self.var_folder_driving = None             # tk.StringVar (dibuat di build)
        self.menu_driving   = {}                   # {emosi: CTkOptionMenu}
        self.pilihan_driving = {}                  # {emosi: tk.StringVar}
        self.video_driving  = {l: [] for l in LABELS}   # {emosi: [path video...]}

        # --- Pilihan emosi (pill multi-pilih) ---
        self.emosi_aktif  = {l: False for l in LABELS}
        self.tombol_emosi = {}

        # --- Tinjau & label hasil (mendukung RIBUAN gambar via navigasi + halaman) ---
        self.review_items  = []        # SELURUH hasil: list (path, emosi, rel)
        self.review_label  = {}        # rel -> label final manual {emosi:0/1}
        self.review_ai     = {}        # rel -> label deteksi AI {emosi:0/1}
        self.review_ditolak = set()    # rel yang ditolak
        self.idx_tinjau    = 0         # posisi aktif di SELURUH daftar
        self.PER_HAL       = 40        # thumbnail per halaman grid
        self._thumb_review = {}        # path -> thumb_bgr (cache halaman)
        self._hal_token    = 0         # anti-race saat ganti halaman cepat
        self._ref_thumbnail = []       # tahan referensi ImageTk agar tidak di-GC
        self.entri_loncat  = None
        self.label_posisi_tinjau = None
        self.label_hal     = None
        self._items_tinjau_penuh = []   # SEMUA item sebelum filter
        self.var_filter_tinjau = None   # Semua|Diterima|Ditolak|AI != target|per-emosi
        self._periksa_token = 0         # anti-balapan load full-res pemeriksa

        # --- Referensi widget (diisi saat build) ---
        self.label_sumber   = None
        self.kanvas_pratinjau = []
        self.label_pratinjau  = []
        self._ref_pratinjau   = [None, None, None]
        self.slider_hasil   = None
        self.label_info_hasil = None
        self.label_progres  = None
        self.label_jml_tanda = None
        self.label_frame_terpilih = None
        self.entri_jumlah   = None
        self._viz_terjadwal = None     # id after() untuk debounce viz
        # pemeriksa hasil
        self.kanvas_pemeriksa = None
        self.label_pemeriksa  = None
        self.pill_label_periksa = {}
        self.tombol_tolak_periksa = None
        self.grid_tinjau = None

    @property
    def _built(self):
        """Kompatibilitas: app.py mengecek `lp._built`."""
        return self._sudah_dibangun

    # ── Membangun UI (lazy: dipanggil saat panel pertama kali dibuka) ───────────

    def build(self):
        if self._sudah_dibangun:
            return
        self._sudah_dibangun = True

        self.var_folder_driving = tk.StringVar(value=self._folder_driving_awal)
        self.var_folder_wajah   = tk.StringVar(value=self._folder_wajah_awal)
        self.var_merge_mode     = tk.StringVar(value="lp")   # base | lp | lp_new
        self.var_kunci_posisi   = tk.BooleanVar(value=True)  # pertahankan posisi proporsional
        self.var_filter_tinjau  = tk.StringVar(value="Semua")
        for emosi in LABELS:
            self.pilihan_driving[emosi] = tk.StringVar(value="(tidak ada)")

        area = ctk.CTkScrollableFrame(self.parent, fg_color=("#f3f4f6", "#161622"),
                                      corner_radius=0)
        area.pack(fill="both", expand=True)
        _aktifkan_scroll_mouse(area)

        self._bangun_header_sumber(area)
        self._bangun_konfig_driving(area)
        self._bangun_pemilih_emosi(area)
        self._bangun_tombol_proses(area)
        self._bangun_pratinjau(area)
        self._bangun_pemilih_frame(area)
        self._bangun_dataset_wajah(area)
        self._bangun_tinjau_label(area)

        # Ringkasan setelan ikut berubah saat dropdown driving diganti
        for emosi in LABELS:
            self.pilihan_driving[emosi].trace_add("write", self._segarkan_ringkasan)

        self._pindai_folder_driving()   # isi menu driving otomatis
        self._segarkan_ringkasan()

    def _bangun_header_sumber(self, induk):
        kartu = ctk.CTkFrame(induk, fg_color=("#fef3c7", "#1c1400"), corner_radius=10)
        kartu.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(kartu, text="SUMBER", font=("Poppins", 9, "bold"),
                     text_color=("#d97706", "#fbbf24")).pack(side="left", padx=(12, 6), pady=8)
        self.label_sumber = ctk.CTkLabel(
            kartu, text="belum ada frame dipilih — tandai 'LP Transform' di galeri",
            font=("Poppins", 9), text_color=("#374151", "#e5e7eb"))
        self.label_sumber.pack(side="left", padx=(0, 8))
        ctk.CTkButton(kartu, text="Berikutnya ▶", width=92, height=26, corner_radius=20,
                      font=("Poppins", 9, "bold"), fg_color=WARNA_LP, hover_color="#d97706",
                      text_color="#1c1400",
                      command=lambda: self.app._goto_lp_mark(+1)).pack(side="right", padx=(2, 10))
        ctk.CTkButton(kartu, text="◀ Sebelumnya", width=92, height=26, corner_radius=20,
                      font=("Poppins", 9, "bold"), fg_color=WARNA_LP, hover_color="#d97706",
                      text_color="#1c1400",
                      command=lambda: self.app._goto_lp_mark(-1)).pack(side="right", padx=2)
        ctk.CTkButton(kartu, text="Hapus Semua Tanda", width=128, height=26, corner_radius=20,
                      font=("Poppins", 9, "bold"), fg_color=("#fee2e2", "#3a1414"),
                      hover_color=("#fecaca", "#4a1a1a"), text_color=("#b91c1c", "#fca5a5"),
                      command=self.app._lp_clear_all_marks).pack(side="right", padx=(2, 8))
        self.label_jml_tanda = ctk.CTkLabel(kartu, text="0 tertanda", font=("Poppins", 8),
                                            text_color=("#6b7280", "#9ca3af"))
        self.label_jml_tanda.pack(side="right", padx=(0, 8))

    def _bangun_konfig_driving(self, induk):
        _judul_seksi(induk, "Folder Video Driving (acuan ekspresi)")
        ctk.CTkLabel(
            induk,
            text="Taruh video acuan di folder ini. PENTING: beri nama file sesuai NAMA EMOSI "
                 "LENGKAP + nomor urut — Confusion1.mp4, Frustration1.mp4, Boredom1.mp4, "
                 "Engagement1.mp4 (bukan 'confuse'). Tambah variasi: Confusion2.mp4, dst.",
            font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
            wraplength=680, justify="left").pack(fill="x", padx=14, pady=(0, 4))

        baris = ctk.CTkFrame(induk, fg_color="transparent")
        baris.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkEntry(baris, textvariable=self.var_folder_driving,
                     font=("Poppins", 9), height=28, corner_radius=6).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(baris, text="Pilih…", width=58, height=28, corner_radius=6,
                      font=("Poppins", 9), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self._pilih_folder_driving).pack(side="left", padx=(0, 4))
        ctk.CTkButton(baris, text="Pindai", width=64, height=28, corner_radius=6,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=self._pindai_folder_driving).pack(side="left")

        # Tanpa opsi "Semua": tiap video referensi beda pose, jadi WAJIB dipilih dan
        # dipreview satu per satu. Ganti pilihan → preview DRIVING langsung dimuat +
        # tanda frame untuk video itu dipulihkan.
        for emosi in LABELS:
            r = ctk.CTkFrame(induk, fg_color="transparent")
            r.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(r, text=emosi[:4], font=("Poppins", 9, "bold"),
                         text_color=LABEL_COLORS[emosi], width=44, anchor="w").pack(side="left")
            menu = ctk.CTkOptionMenu(
                r, variable=self.pilihan_driving[emosi], values=["(tidak ada)"],
                font=("Poppins", 9), height=26, width=300, corner_radius=6,
                fg_color=("#e5e7eb", "#23233a"), button_color=LABEL_COLORS[emosi],
                button_hover_color="#d97706",
                command=lambda _v, e=emosi: self._saat_driving_berganti(e))
            menu.pack(side="left", padx=(4, 0))
            self.menu_driving[emosi] = menu

    def _bangun_pemilih_emosi(self, induk):
        _judul_seksi(induk, "Emosi yang Diproses (pilih SATU)")
        baris = ctk.CTkFrame(induk, fg_color="transparent")
        baris.pack(fill="x", padx=14, pady=(2, 8))
        for emosi in LABELS:
            warna = LABEL_COLORS[emosi]
            tombol = ctk.CTkButton(
                baris, text=emosi, width=92, height=28, corner_radius=20,
                font=("Poppins", 9, "bold"), fg_color="transparent", border_width=2,
                border_color=warna, text_color=warna, hover_color=("#e5e7eb", "#23233a"),
                command=lambda e=emosi: self._ganti_emosi(e))
            tombol.pack(side="left", padx=3)
            self.tombol_emosi[emosi] = tombol

    def _bangun_tombol_proses(self, induk):
        _judul_seksi(induk, "Proses LivePortrait")
        baris = ctk.CTkFrame(induk, fg_color="transparent")
        baris.pack(fill="x", padx=14, pady=(4, 4))
        ctk.CTkButton(baris, text="Proses Frame Ini", height=32, corner_radius=8,
                      font=("Poppins", 10, "bold"), fg_color=WARNA_LP, hover_color="#d97706",
                      text_color="#1c1400",
                      command=self.app._lp_process_current).pack(side="left", padx=(0, 8))
        ctk.CTkButton(baris, text="Batch Semua", height=32, corner_radius=8,
                      font=("Poppins", 10, "bold"), fg_color="#8b5cf6", hover_color="#7c3aed",
                      command=self.app._lp_process_batch).pack(side="left", padx=(0, 8))
        ctk.CTkButton(baris, text="Batal", height=32, corner_radius=8,
                      font=("Poppins", 10, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self.app._lp_cancel).pack(side="left", padx=(0, 8))
        ctk.CTkButton(baris, text="Reset", height=32, corner_radius=8,
                      font=("Poppins", 10, "bold"), fg_color=("#fee2e2", "#3a1414"),
                      hover_color=("#fecaca", "#4a1a1a"), text_color=("#b91c1c", "#fca5a5"),
                      command=self.reset).pack(side="left")
        self.label_progres = ctk.CTkLabel(induk, text="", font=("Poppins", 9),
                                          text_color="#10b981", anchor="w", justify="left",
                                          wraplength=680)
        self.label_progres.pack(fill="x", padx=14, pady=(0, 4))

    def _bangun_pratinjau(self, induk):
        _judul_seksi(induk, "Pratinjau  ·  SUMBER | DRIVING | HASIL  (viz ikut saklar Viz)")
        wadah = ctk.CTkFrame(induk, fg_color="transparent")
        wadah.pack(fill="x", padx=10, pady=(0, 4))
        for kolom in range(3):
            wadah.columnconfigure(kolom, weight=1)
        for kolom, judul in enumerate(["SUMBER", "DRIVING", "HASIL LP"]):
            sel = ctk.CTkFrame(wadah, fg_color="transparent")
            sel.grid(row=0, column=kolom, padx=5, pady=4, sticky="nsew")
            ctk.CTkLabel(sel, text=judul, font=("Poppins", 9, "bold"),
                         text_color=("#374151", "#9ca3af")).pack()
            kanvas = tk.Canvas(sel, width=UKURAN_PRATINJAU, height=UKURAN_PRATINJAU,
                               bg="#111", highlightthickness=2, highlightbackground="#2d2d40")
            kanvas.pack()
            kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                               text="—", fill="#4b5563", font=("Poppins", 10))
            self.kanvas_pratinjau.append(kanvas)
            label = ctk.CTkLabel(sel, text="", font=("Poppins", 8),
                                 text_color=("#6b7280", "#9ca3af"), wraplength=UKURAN_PRATINJAU)
            label.pack(pady=(2, 0))
            self.label_pratinjau.append(label)
            ctk.CTkButton(sel, text="Unduh Gambar", height=26, corner_radius=8,
                          font=("Poppins", 9), fg_color=("#e5e7eb", "#2a2a3a"),
                          hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                          command=lambda k=kolom: self._unduh_gambar(k)).pack(pady=(4, 0))

    def _bangun_pemilih_frame(self, induk):
        _judul_seksi(induk, "Pilih Frame Hasil untuk Dataset")
        kartu = ctk.CTkFrame(induk, fg_color=("#f3f4f6", "#1a1a2e"), corner_radius=10)
        kartu.pack(fill="x", padx=12, pady=(0, 4))
        self.label_info_hasil = ctk.CTkLabel(
            kartu, text="Belum ada hasil — klik 'Proses Frame Ini'",
            font=("Poppins", 9), text_color=("#6b7280", "#9ca3af"))
        self.label_info_hasil.pack(anchor="w", padx=12, pady=(8, 2))
        self.slider_hasil = ctk.CTkSlider(kartu, from_=0, to=100, number_of_steps=100,
                                          command=self._geser_slider,
                                          progress_color=WARNA_LP, button_color=WARNA_LP)
        self.slider_hasil.set(0)
        self.slider_hasil.pack(fill="x", padx=12, pady=(0, 4))
        self.slider_hasil.configure(state="disabled")

        baris_tanda = ctk.CTkFrame(kartu, fg_color="transparent")
        baris_tanda.pack(fill="x", padx=12, pady=(0, 2))
        ctk.CTkButton(baris_tanda, text="＋ Tandai Frame", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=WARNA_LP, hover_color="#d97706",
                      text_color="#1c1400", command=self._tandai_frame_sekarang).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(baris_tanda, text="Jumlah target:", font=("Poppins", 9),
                     text_color=("#6b7280", "#9ca3af")).pack(side="left", padx=(4, 2))
        self.entri_jumlah = ctk.CTkEntry(baris_tanda, width=40, height=28, font=("Poppins", 9),
                                         placeholder_text="4", corner_radius=6)
        self.entri_jumlah.pack(side="left", padx=(0, 6))
        ctk.CTkButton(baris_tanda, text="Tandai Merata", height=28, corner_radius=8,
                      font=("Poppins", 9), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self._tandai_merata).pack(side="left", padx=(0, 6))
        ctk.CTkButton(baris_tanda, text="Bersihkan", height=28, corner_radius=8,
                      font=("Poppins", 9), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self._bersihkan_tanda).pack(side="left")

        # Kunci posisi: saat video driving diganti & diproses ulang, frame tertanda otomatis
        # dipetakan ke posisi PROPORSIONAL di video baru — jumlah tetap, letak menyesuaikan.
        ctk.CTkCheckBox(
            kartu, text="Kunci posisi proporsional saat ganti driving (jumlah sama, letak menyesuaikan)",
            variable=self.var_kunci_posisi, font=("Poppins", 9),
            checkbox_width=18, checkbox_height=18, corner_radius=4,
            fg_color=WARNA_LP, hover_color="#d97706").pack(anchor="w", padx=12, pady=(4, 0))

        self.label_frame_terpilih = ctk.CTkLabel(
            kartu, text="Frame tertanda: (kosong)", font=("Poppins", 9),
            text_color=("#d97706", "#fbbf24"), anchor="w", justify="left", wraplength=680)
        self.label_frame_terpilih.pack(fill="x", padx=12, pady=(2, 4))

        baris_simpan = ctk.CTkFrame(kartu, fg_color="transparent")
        baris_simpan.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkButton(baris_simpan, text="Simpan Frame Tertanda", height=30, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#10b981", hover_color="#0ea372",
                      command=self._simpan_frame_tertanda).pack(side="left", padx=(0, 8))
        ctk.CTkButton(baris_simpan, text="Simpan Frame Ini Saja", height=30, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=self._simpan_frame_sekarang).pack(side="left")
        ctk.CTkLabel(
            kartu, text="Saat Batch, frame index yang ditandai ini dipakai untuk SEMUA video "
                        "(ekspresi sama karena driving sama).",
            font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
            wraplength=680, justify="left").pack(fill="x", padx=12, pady=(0, 8))

    def _bangun_dataset_wajah(self, induk):
        _judul_seksi(induk, "Dataset Wajah Baru  (opsional — tiap foto = 1 orang baru)")
        ctk.CTkLabel(
            induk,
            text="Untuk menambah RAGAM orang (mis. dari dataset wajah open-source) agar model "
                 "tidak overfit. Pilih folder berisi foto wajah; tiap foto dianggap ORANG BARU. "
                 "Centang foto yang ingin diproses, lalu 'Proses Wajah Terpilih' memakai emosi + "
                 "driving + frame index di atas.",
            font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
            wraplength=680, justify="left").pack(fill="x", padx=14, pady=(0, 4))

        baris = ctk.CTkFrame(induk, fg_color="transparent")
        baris.pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkEntry(baris, textvariable=self.var_folder_wajah, font=("Poppins", 9),
                     height=28, corner_radius=6,
                     placeholder_text="folder berisi foto wajah…").pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(baris, text="Pilih…", width=58, height=28, corner_radius=6,
                      font=("Poppins", 9), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self._pilih_folder_wajah).pack(side="left", padx=(0, 4))
        ctk.CTkButton(baris, text="Pindai", width=64, height=28, corner_radius=6,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=self._pindai_folder_wajah).pack(side="left")

        aksi = ctk.CTkFrame(induk, fg_color="transparent")
        aksi.pack(fill="x", padx=14, pady=(2, 2))
        ctk.CTkButton(aksi, text="Pilih Semua", height=26, corner_radius=8, font=("Poppins", 9),
                      fg_color=("#e5e7eb", "#2a2a3a"), hover_color=("#d1d5db", "#3a3a4a"),
                      text_color=("#374151", "#9ca3af"),
                      command=self._pilih_semua_wajah).pack(side="left", padx=(0, 6))
        ctk.CTkButton(aksi, text="Kosongkan", height=26, corner_radius=8, font=("Poppins", 9),
                      fg_color=("#e5e7eb", "#2a2a3a"), hover_color=("#d1d5db", "#3a3a4a"),
                      text_color=("#374151", "#9ca3af"),
                      command=self._kosongkan_pilihan_wajah).pack(side="left", padx=(0, 6))
        ctk.CTkButton(aksi, text="Proses Wajah Terpilih", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=WARNA_LP, hover_color="#d97706",
                      text_color="#1c1400",
                      command=self.app._lp_process_faces).pack(side="left")
        self.label_jml_wajah = ctk.CTkLabel(aksi, text="0 dipilih", font=("Poppins", 8),
                                            text_color=("#6b7280", "#9ca3af"))
        self.label_jml_wajah.pack(side="left", padx=(8, 0))

        # Ringkasan setelan aktif — supaya user TIDAK perlu scroll ke atas untuk
        # mengecek emosi/driving/frame sebelum memproses wajah.
        self.label_ringkasan_wajah = ctk.CTkLabel(
            induk, text="", font=("Poppins", 8),
            text_color=("#d97706", "#fbbf24"), anchor="w", justify="left", wraplength=680)
        self.label_ringkasan_wajah.pack(fill="x", padx=14, pady=(0, 2))

        self.grid_wajah = ctk.CTkFrame(induk, fg_color="transparent")
        self.grid_wajah.pack(fill="both", expand=True, padx=10, pady=(2, 8))

    def _bangun_tinjau_label(self, induk):
        _judul_seksi(induk, "Tinjau & Label Hasil  (cek sebelum buat dataset)")
        bar = ctk.CTkFrame(induk, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkButton(bar, text="Muat / Refresh", height=28, width=120, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=self.refresh_review).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Deteksi AI Semua", height=28, width=130, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#6366f1", hover_color="#4f46e5",
                      command=self.app._lp_label_all_ai).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Buang Ditolak → _trash", height=28, width=156, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#ef4444", hover_color="#dc2626",
                      command=self.app._lp_delete_rejected).pack(side="left", padx=(0, 6))

        # Baris alat QA: filter tampilan + auto-tolak mismatch AI + statistik dataset
        bar2 = ctk.CTkFrame(induk, fg_color="transparent")
        bar2.pack(fill="x", padx=14, pady=(2, 2))
        ctk.CTkLabel(bar2, text="Filter:", font=("Poppins", 9),
                     text_color=("#6b7280", "#9ca3af")).pack(side="left", padx=(0, 4))
        self.menu_filter = ctk.CTkOptionMenu(
            bar2, variable=self.var_filter_tinjau,
            values=["Semua", "Diterima", "Ditolak", "AI != target"] + LABELS,
            font=("Poppins", 9), height=26, width=130, corner_radius=6,
            fg_color=("#e5e7eb", "#23233a"), button_color="#3b82f6",
            command=lambda _v: self._terapkan_filter())
        self.menu_filter.pack(side="left", padx=(0, 10))
        ctk.CTkButton(bar2, text="Auto-Tolak (AI != target)", height=26, width=170, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=("#fee2e2", "#3a1414"),
                      hover_color=("#fecaca", "#4a1a1a"), text_color=("#b91c1c", "#fca5a5"),
                      command=self.app._lp_auto_reject_mismatch).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar2, text="Statistik Dataset", height=26, width=130, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#10b981", hover_color="#0ea372",
                      command=self.app._lp_show_stats).pack(side="left")

        ctk.CTkLabel(
            induk, text="Klik 1x thumbnail = lihat besar + bandingkan deteksi AI vs label manual.  "
                        "Klik 2x = tolak/terima (merah = ditolak, tidak ikut dataset).  "
                        "Auto-Tolak menandai tolak semua hasil yang menurut AI tidak mengandung "
                        "emosi targetnya (jalankan 'Deteksi AI Semua' dulu).",
            font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
            wraplength=680, justify="left").pack(fill="x", padx=14, pady=(2, 4))

        # ── Buat Dataset (pilih komposisi via bullet/radio) ──
        kartu_buat = ctk.CTkFrame(induk, fg_color=("#ecfdf5", "#0f1f1a"), corner_radius=10)
        kartu_buat.pack(fill="x", padx=12, pady=(2, 6))
        ctk.CTkLabel(kartu_buat, text="Buat Dataset — pilih komposisi:", font=("Poppins", 9, "bold"),
                     text_color=("#059669", "#34d399")).pack(anchor="w", padx=12, pady=(8, 2))
        baris_radio = ctk.CTkFrame(kartu_buat, fg_color="transparent")
        baris_radio.pack(fill="x", padx=12, pady=(0, 2))
        opsi = [("base",   "Tanpa LP (asli saja)"),
                ("lp",     "Dengan LP (asli + LP)"),
                ("lp_new", "LP + Dataset Wajah Baru")]
        for nilai, teks in opsi:
            ctk.CTkRadioButton(baris_radio, text=teks, value=nilai, variable=self.var_merge_mode,
                               font=("Poppins", 9), radiobutton_width=16, radiobutton_height=16,
                               fg_color="#10b981", hover_color="#0ea372").pack(side="left", padx=(0, 14))
        baris_buat = ctk.CTkFrame(kartu_buat, fg_color="transparent")
        baris_buat.pack(fill="x", padx=12, pady=(2, 10))
        ctk.CTkButton(baris_buat, text="Buat Dataset", height=30, width=150, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#10b981", hover_color="#0ea372",
                      command=self.app._lp_merge_into_label2d).pack(side="left", padx=(0, 8))
        ctk.CTkButton(baris_buat, text="Undo / Hapus", height=30, width=120, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self.app._lp_undo_merge).pack(side="left")
        ctk.CTkLabel(kartu_buat,
                     text="Output ke Label2d_merged_{komposisi}/ (non-destruktif, kolom 'synthetic'). "
                          "Label2d asli tidak diubah.",
                     font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
                     wraplength=680, justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        # Pemeriksa: gambar BESAR + NAVIGASI (prev/next/loncat) + pill label manual + tolak
        kartu = ctk.CTkFrame(induk, fg_color=("#f3f4f6", "#1a1a2e"), corner_radius=10)
        kartu.pack(fill="x", padx=12, pady=(0, 6))
        kiri = ctk.CTkFrame(kartu, fg_color="transparent")
        kiri.pack(side="left", padx=10, pady=10)
        self.kanvas_pemeriksa = tk.Canvas(kiri, width=UKURAN_PEMERIKSA, height=UKURAN_PEMERIKSA,
                                          bg="#111", highlightthickness=2, highlightbackground="#2d2d40")
        self.kanvas_pemeriksa.pack()
        self.kanvas_pemeriksa.bind("<Double-Button-1>", lambda e: self._tolak_pemeriksa())
        self.kanvas_pemeriksa.create_text(UKURAN_PEMERIKSA // 2, UKURAN_PEMERIKSA // 2,
                                          text="klik Muat / Refresh", fill="#4b5563", font=("Poppins", 10))
        # Baris navigasi di bawah gambar besar
        nav = ctk.CTkFrame(kiri, fg_color="transparent")
        nav.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(nav, text="◀ Sebelumnya", width=96, height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=lambda: self._tinjau_geser(-1)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(nav, text="Berikutnya ▶", width=96, height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=lambda: self._tinjau_geser(+1)).pack(side="left", padx=(0, 8))
        self.entri_loncat = ctk.CTkEntry(nav, width=52, height=28, font=("Poppins", 9),
                                         placeholder_text="no.", corner_radius=6)
        self.entri_loncat.pack(side="left", padx=(0, 4))
        self.entri_loncat.bind("<Return>", lambda e: self._tinjau_loncat())
        ctk.CTkButton(nav, text="Loncat", width=64, height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self._tinjau_loncat).pack(side="left")
        self.label_posisi_tinjau = ctk.CTkLabel(kiri, text="0 / 0", font=("Poppins", 9, "bold"),
                                                text_color=("#374151", "#e5e7eb"))
        self.label_posisi_tinjau.pack(pady=(4, 0))

        kanan = ctk.CTkFrame(kartu, fg_color="transparent")
        kanan.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.label_pemeriksa = ctk.CTkLabel(kanan, text="—", font=("Poppins", 9, "bold"),
                                            text_color=("#374151", "#e5e7eb"), anchor="w",
                                            justify="left", wraplength=320)
        self.label_pemeriksa.pack(anchor="w", pady=(0, 6))

        # Hasil deteksi AI (read-only) — perbandingan dengan label manual
        ctk.CTkLabel(kanan, text="Deteksi AI (SigLIP+MediaPipe):", font=("Poppins", 8),
                     text_color=("#6b7280", "#9ca3af")).pack(anchor="w")
        baris_ai = ctk.CTkFrame(kanan, fg_color="transparent")
        baris_ai.pack(anchor="w", pady=(2, 6))
        for emosi in LABELS:
            warna = LABEL_COLORS[emosi]
            chip = ctk.CTkLabel(baris_ai, text=emosi[:4], width=54, height=22, corner_radius=11,
                                font=("Poppins", 8, "bold"), fg_color="transparent",
                                text_color=("#9ca3af", "#4b5563"))
            chip.pack(side="left", padx=2)
            self.chip_ai_periksa[emosi] = chip

        ctk.CTkLabel(kanan, text="Label final manual (dipakai saat merge — klik untuk ubah):",
                     font=("Poppins", 8), text_color=("#6b7280", "#9ca3af")).pack(anchor="w")
        baris_pill = ctk.CTkFrame(kanan, fg_color="transparent")
        baris_pill.pack(anchor="w", pady=(2, 8))
        for emosi in LABELS:
            warna = LABEL_COLORS[emosi]
            pill = ctk.CTkButton(
                baris_pill, text=emosi, width=86, height=26, corner_radius=20,
                font=("Poppins", 9, "bold"), fg_color="transparent", border_width=2,
                border_color=warna, text_color=warna, hover_color=("#e5e7eb", "#23233a"),
                command=lambda e=emosi: self._ganti_label_periksa(e))
            pill.pack(side="left", padx=3)
            self.pill_label_periksa[emosi] = pill
        self.tombol_tolak_periksa = ctk.CTkButton(
            kanan, text="Tolak Gambar Ini", height=28, corner_radius=8,
            font=("Poppins", 9, "bold"), fg_color=("#fee2e2", "#3a1414"),
            hover_color=("#fecaca", "#4a1a1a"), text_color=("#b91c1c", "#fca5a5"),
            command=self._tolak_pemeriksa)
        self.tombol_tolak_periksa.pack(anchor="w")

        # Grid thumbnail (BERHALAMAN) untuk lompat cepat — klik = buka di pemeriksa
        bar_hal = ctk.CTkFrame(induk, fg_color="transparent")
        bar_hal.pack(fill="x", padx=14, pady=(2, 0))
        ctk.CTkButton(bar_hal, text="◀ Halaman", width=84, height=24, corner_radius=8,
                      font=("Poppins", 8, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=lambda: self._geser_halaman(-1)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(bar_hal, text="Halaman ▶", width=84, height=24, corner_radius=8,
                      font=("Poppins", 8, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=lambda: self._geser_halaman(+1)).pack(side="left", padx=(0, 8))
        self.label_hal = ctk.CTkLabel(bar_hal, text="", font=("Poppins", 8),
                                      text_color=("#6b7280", "#9ca3af"))
        self.label_hal.pack(side="left")
        self.grid_tinjau = ctk.CTkFrame(induk, fg_color="transparent")
        self.grid_tinjau.pack(fill="both", expand=True, padx=10, pady=(2, 12))

    # ── Menampilkan gambar (semua dari memori → tanpa lag, tanpa crash) ─────────

    def _tampilkan(self, kolom: int, bgr):
        """Tampilkan array BGR ke kanvas pratinjau ke-`kolom` (0=sumber,1=driving,2=hasil).
        Letterbox (rasio dijaga) supaya video driving/hasil non-persegi tidak gepeng."""
        import cv2
        self.bgr_tampil[kolom] = bgr
        rgb = cv2.cvtColor(_letterbox(bgr, UKURAN_PRATINJAU, UKURAN_PRATINJAU), cv2.COLOR_BGR2RGB)
        gambar = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._ref_pratinjau[kolom] = gambar
        kanvas = self.kanvas_pratinjau[kolom]
        kanvas.delete("all")
        kanvas.create_image(0, 0, anchor="nw", image=gambar)

    def _viz_jika_aktif(self, bgr):
        """Tambahkan overlay landmark HANYA bila saklar Viz aktif; selain itu kembalikan apa adanya."""
        try:
            if not self.app.show_viz.get():
                return bgr
        except Exception:
            return bgr
        try:
            from core.landmark_analyzer import (analyze_frame, compute_emotion_scores,
                                                draw_landmark_viz)
            hasil = analyze_frame(bgr)
            if not hasil.face_found:
                return bgr
            return draw_landmark_viz(bgr.copy(), hasil,
                                     compute_emotion_scores(hasil, self.app.rules))
        except Exception:
            return bgr

    def _render_index(self, index: int):
        """Tampilkan DRIVING + HASIL pada index frame tertentu (cepat, dari memori)."""
        if self.frame_hasil:
            ih = min(index, len(self.frame_hasil) - 1)
            self._tampilkan(2, self.frame_hasil[ih])
            self.label_pratinjau[2].configure(
                text=f"{self.emosi_hasil}  ·  frame {ih+1}/{len(self.frame_hasil)}")
        if self.frame_driving:
            idr = min(index, len(self.frame_driving) - 1)
            self._tampilkan(1, self.frame_driving[idr])
            self.label_pratinjau[1].configure(
                text=f"{self.emosi_driving}  ·  frame {idr+1}/{len(self.frame_driving)}")

    def _total_aktif(self) -> int:
        """Jumlah frame yang bisa di-scrub/ditandai: video HASIL bila ada,
        kalau belum diproses pakai video DRIVING (pratinjau pose sebelum proses)."""
        return len(self.frame_hasil) or len(self.frame_driving)

    def _geser_slider(self, nilai):
        """Saat slider digeser: tampilkan frame mentah (instan), viz dihitung belakangan.
        Bekerja juga SEBELUM proses (scrub video driving untuk melihat pose tiap frame)."""
        total = self._total_aktif()
        if not total:
            return
        index = int(float(nilai))
        self.index_hasil = index
        tanda = "  ●ditandai" if index in self.frame_terpilih else ""
        mode = self.emosi_hasil if self.frame_hasil else f"pratinjau driving {self.emosi_driving}"
        self.label_info_hasil.configure(text=f"Frame {index+1} / {total}  ·  {mode}{tanda}")
        self._render_index(index)               # gambar mentah dulu (tanpa lag)
        self._jadwalkan_viz(index)              # viz menyusul bila saklar aktif

    def _jadwalkan_viz(self, index: int):
        """Debounce: hitung viz hanya saat slider berhenti ~250ms (agar geser tetap mulus)."""
        try:
            if not self.app.show_viz.get():
                return
        except Exception:
            return
        if self._viz_terjadwal is not None:
            try:
                self.app.root.after_cancel(self._viz_terjadwal)
            except Exception:
                pass

        def _terapkan(idx=index):
            self._viz_terjadwal = None
            if idx != self.index_hasil:
                return
            if self.frame_hasil:
                ih = min(idx, len(self.frame_hasil) - 1)
                self._tampilkan(2, self._viz_jika_aktif(self.frame_hasil[ih]))
            if self.frame_driving:
                idr = min(idx, len(self.frame_driving) - 1)
                self._tampilkan(1, self._viz_jika_aktif(self.frame_driving[idr]))

        self._viz_terjadwal = self.app.root.after(250, _terapkan)

    # ── Tandai & simpan frame hasil ─────────────────────────────────────────────

    def _simpan_fraksi(self):
        """Rekam posisi tertanda sebagai fraksi 0-1 (template lintas-driving)."""
        total = self._total_aktif()
        if total > 1 and self.frame_terpilih:
            self.fraksi_terpilih = [i / (total - 1) for i in sorted(self.frame_terpilih)]
        elif not self.frame_terpilih:
            self.fraksi_terpilih = []

    def _simpan_tanda_driving(self):
        """Simpan tanda frame untuk video driving yang sedang aktif (per-video,
        karena tiap referensi beda pose — ganti video, tanda ikut video itu)."""
        if self.driving_aktif:
            self.tanda_per_driving[self.driving_aktif] = sorted(self.frame_terpilih)

    def _tandai_frame_sekarang(self):
        if not self._total_aktif():
            return
        self.frame_terpilih.add(self.index_hasil)
        self._simpan_fraksi()
        self._simpan_tanda_driving()
        self._segarkan_tanda()

    def _tandai_merata(self):
        total = self._total_aktif()
        if not total:
            return
        jumlah = self.get_target_n()
        self.frame_terpilih = {int(round(i * (total - 1) / max(jumlah - 1, 1))) for i in range(jumlah)}
        self._simpan_fraksi()
        self._simpan_tanda_driving()
        self._segarkan_tanda()

    def _bersihkan_tanda(self):
        self.frame_terpilih.clear()
        self.fraksi_terpilih = []
        self._simpan_tanda_driving()
        self._segarkan_tanda()

    def _segarkan_tanda(self):
        if not self.frame_terpilih:
            self.label_frame_terpilih.configure(text="Frame tertanda: (kosong)")
        else:
            urut = ", ".join(str(i + 1) for i in sorted(self.frame_terpilih))
            self.label_frame_terpilih.configure(
                text=f"Frame tertanda ({len(self.frame_terpilih)}): {urut}")
        self._geser_slider(self.index_hasil)
        self._segarkan_ringkasan()

    def _simpan_frame_tertanda(self):
        if not self.frame_hasil:
            self.update_progress("Belum ada video hasil", "#ef4444"); return
        if not self.frame_terpilih:
            self.update_progress("Tandai dulu frame yang mau disimpan", "#f59e0b"); return
        frames = [(i, self.frame_hasil[i]) for i in sorted(self.frame_terpilih)
                  if i < len(self.frame_hasil)]
        self.app._lp_save_frames(frames, self.emosi_hasil)

    def _simpan_frame_sekarang(self):
        if not self.frame_hasil:
            self.update_progress("Belum ada video hasil", "#ef4444"); return
        i = self.index_hasil
        self.app._lp_save_frames([(i, self.frame_hasil[i])], self.emosi_hasil)

    # ── Tinjau & label hasil ────────────────────────────────────────────────────

    def set_review_data(self, items: list, labels: dict, ai_labels: dict, ditolak: set):
        """Terima SELURUH daftar hasil + state (label/ai/tolak). Tidak decode thumbnail di
        sini (cepat walau ribuan). items: list (path, emosi, rel)."""
        self._items_tinjau_penuh = items or []
        self.review_label = labels or {}
        self.review_ai = ai_labels or {}
        self.review_ditolak = set(ditolak or set())
        self._thumb_review.clear()
        self._terapkan_filter()

    def _cocok_filter(self, item) -> bool:
        """Apakah satu item lolos filter tampilan yang sedang dipilih."""
        _path, emo, rel = item
        f = self.var_filter_tinjau.get() if self.var_filter_tinjau else "Semua"
        if f == "Ditolak":
            return rel in self.review_ditolak
        if f == "Diterima":
            return rel not in self.review_ditolak
        if f == "AI != target":
            ai = self.review_ai.get(rel)
            return bool(ai) and ai.get(emo, 0) == 0   # AI tidak mendeteksi emosi target
        if f in LABELS:
            return emo == f
        return True   # "Semua"

    def _terapkan_filter(self):
        """Saring daftar tinjau sesuai filter, lalu render ulang (posisi di-clamp)."""
        self.review_items = [it for it in self._items_tinjau_penuh if self._cocok_filter(it)]
        self.idx_tinjau = min(self.idx_tinjau, len(self.review_items) - 1) if self.review_items else 0
        self._render_pemeriksa()
        self._render_halaman()

    def _label_item(self, rel, emosi):
        return self.review_label.get(rel) or self.app._lp_default_label(emosi)

    # ── Navigasi pemeriksa (menjangkau SELURUH daftar, bukan cuma 1 halaman) ──
    def _tinjau_geser(self, delta: int):
        if not self.review_items:
            return
        self.idx_tinjau = max(0, min(len(self.review_items) - 1, self.idx_tinjau + delta))
        self._render_pemeriksa()
        self._render_halaman()   # pindah halaman bila perlu

    def _tinjau_loncat(self):
        if not self.review_items:
            return
        try:
            n = int(self.entri_loncat.get().strip())
        except (ValueError, AttributeError):
            self.update_progress("Isi nomor untuk loncat (1.." + str(len(self.review_items)) + ")", "#f59e0b")
            return
        self.idx_tinjau = max(0, min(len(self.review_items) - 1, n - 1))
        self._render_pemeriksa()
        self._render_halaman()

    def _buka_di_pemeriksa(self, idx: int):
        """Loncat ke index absolut (dipanggil dari klik thumbnail grid)."""
        if not self.review_items:
            return
        self.idx_tinjau = max(0, min(len(self.review_items) - 1, idx))
        self._render_pemeriksa()
        self._render_halaman()

    def _gambar_pemeriksa(self, bgr, ditolak: bool, emosi: str):
        """Gambar satu bgr ke kanvas pemeriksa (+ overlay DITOLAK + warna tepi)."""
        import cv2
        self.kanvas_pemeriksa.delete("all")
        if bgr is not None:
            rgb = cv2.cvtColor(_letterbox(bgr, UKURAN_PEMERIKSA, UKURAN_PEMERIKSA), cv2.COLOR_BGR2RGB)
            self._ref_periksa = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.kanvas_pemeriksa.create_image(0, 0, anchor="nw", image=self._ref_periksa)
        self.kanvas_pemeriksa.configure(highlightbackground="#ef4444" if ditolak
                                        else LABEL_COLORS.get(emosi, "#333"))
        if ditolak:
            self.kanvas_pemeriksa.create_rectangle(0, 0, UKURAN_PEMERIKSA, UKURAN_PEMERIKSA,
                                                   fill="#2a0000", stipple="gray50")
            self.kanvas_pemeriksa.create_text(UKURAN_PEMERIKSA // 2, 20, text="DITOLAK",
                                              fill="#ef4444", font=("Poppins", 11, "bold"))

    def _render_pemeriksa(self):
        """Tampilkan gambar BESAR pada idx_tinjau + chip AI + pill manual + status tolak.

        Supaya navigasi cepat (klik Berikutnya beruntun) TIDAK tersendat: thumbnail dari
        cache ditampilkan SEKETIKA, lalu gambar full-res dimuat di thread latar dan
        menggantikannya (dengan token anti-balapan)."""
        import threading
        n = len(self.review_items)
        self.label_posisi_tinjau.configure(text=f"{(self.idx_tinjau + 1) if n else 0} / {n}")
        if not n:
            self.kanvas_pemeriksa.delete("all")
            self.kanvas_pemeriksa.create_text(UKURAN_PEMERIKSA // 2, UKURAN_PEMERIKSA // 2,
                                              text="(belum ada hasil)", fill="#4b5563", font=("Poppins", 10))
            self.label_pemeriksa.configure(text="—")
            return
        path, emosi, rel = self.review_items[self.idx_tinjau]
        ditolak = rel in self.review_ditolak
        label = self._label_item(rel, emosi)
        ai_label = self.review_ai.get(rel, {})

        # 1) tampilkan thumbnail dari cache dulu (instan); full-res menyusul
        self._gambar_pemeriksa(self._thumb_review.get(path), ditolak, emosi)
        self._periksa_token += 1
        token = self._periksa_token

        def _muat_penuh(p=path, t=token, tolak=ditolak, emo=emosi):
            import cv2
            bgr = cv2.imread(p)
            if bgr is None:
                return
            def _pasang():
                if t == self._periksa_token:        # masih gambar yang sama?
                    self._gambar_pemeriksa(bgr, tolak, emo)
            try:
                self.app.root.after(0, _pasang)
            except (RuntimeError, tk.TclError):
                pass
        threading.Thread(target=_muat_penuh, daemon=True).start()

        self.label_pemeriksa.configure(text=f"target: {emosi}\n{os.path.basename(path)}")
        ada_ai = bool(ai_label) and sum(ai_label.get(l, 0) for l in LABELS) > 0
        for emo2, chip in self.chip_ai_periksa.items():
            warna = LABEL_COLORS[emo2]
            if ada_ai and ai_label.get(emo2, 0) == 1:
                chip.configure(fg_color=warna, text_color="#0b0b12")
            else:
                chip.configure(fg_color="transparent", text_color=("#9ca3af", "#4b5563"))
        for emo2, pill in self.pill_label_periksa.items():
            warna = LABEL_COLORS[emo2]
            if label.get(emo2, 0) == 1:
                pill.configure(fg_color=warna, text_color="#0b0b12", border_color=warna)
            else:
                pill.configure(fg_color="transparent", text_color=warna, border_color=warna)
        self.tombol_tolak_periksa.configure(text="Batal Tolak" if ditolak else "Tolak Gambar Ini")

    # ── Grid berhalaman (thumbnail di-decode di thread latar) ──
    def _geser_halaman(self, delta: int):
        if not self.review_items:
            return
        n_hal = (len(self.review_items) - 1) // self.PER_HAL + 1
        hal_kini = self.idx_tinjau // self.PER_HAL
        hal_baru = max(0, min(n_hal - 1, hal_kini + delta))
        # pindahkan idx ke awal halaman baru lalu render
        self.idx_tinjau = hal_baru * self.PER_HAL
        self._render_pemeriksa()
        self._render_halaman()

    def _render_halaman(self):
        """Render satu halaman thumbnail (yang memuat idx_tinjau). Decode di thread latar."""
        import threading
        for anak in self.grid_tinjau.winfo_children():
            anak.destroy()
        self._ref_thumbnail.clear()
        n = len(self.review_items)
        if not n:
            self.label_hal.configure(text="")
            return
        hal = self.idx_tinjau // self.PER_HAL
        n_hal = (n - 1) // self.PER_HAL + 1
        awal = hal * self.PER_HAL
        chunk = self.review_items[awal:awal + self.PER_HAL]
        self.label_hal.configure(text=f"halaman {hal+1}/{n_hal}  ·  total {n} gambar")
        self._hal_token += 1
        token = self._hal_token

        def _decode():
            import cv2
            hasil = []
            for (path, emosi, rel) in chunk:
                t = self._thumb_review.get(path)
                if t is None:
                    bgr = cv2.imread(path)
                    if bgr is not None:
                        t = _letterbox(bgr, UKURAN_THUMBNAIL, UKURAN_THUMBNAIL)
                        self._thumb_review[path] = t
                hasil.append((path, emosi, rel, t))
            # Batasi cache thumbnail (FIFO ~12 halaman) supaya RAM tidak membengkak
            # saat menjelajah ribuan gambar.
            MAKS = self.PER_HAL * 12
            while len(self._thumb_review) > MAKS:
                self._thumb_review.pop(next(iter(self._thumb_review)), None)
            try:   # root mungkin sudah ditutup saat shutdown → abaikan
                self.app.root.after(0, lambda: self._gambar_halaman(hasil, awal, token))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_decode, daemon=True).start()

    def _gambar_halaman(self, hasil, awal, token):
        if token != self._hal_token:
            return   # halaman sudah berganti
        import cv2
        KOLOM = 5
        for w in self.grid_tinjau.winfo_children():
            w.destroy()
        self._ref_thumbnail.clear()
        for j, (path, emosi, rel, thumb) in enumerate(hasil):
            idx_abs = awal + j
            r, c = divmod(j, KOLOM)
            sel = tk.Frame(self.grid_tinjau, bg="#161622")
            sel.grid(row=r, column=c, padx=3, pady=3)
            ditolak = rel in self.review_ditolak
            aktif = (idx_abs == self.idx_tinjau)
            warna_tepi = (WARNA_LP if aktif else ("#ef4444" if ditolak
                          else LABEL_COLORS.get(emosi, "#333")))
            kanvas = tk.Canvas(sel, width=UKURAN_THUMBNAIL, height=UKURAN_THUMBNAIL, bg="#111",
                               highlightthickness=3 if aktif else 2,
                               highlightbackground=warna_tepi, cursor="hand2")
            kanvas.pack()
            if thumb is not None:
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._ref_thumbnail.append(img)
                kanvas.create_image(0, 0, anchor="nw", image=img)
            if ditolak:
                kanvas.create_rectangle(0, 0, UKURAN_THUMBNAIL, UKURAN_THUMBNAIL,
                                        fill="#2a0000", stipple="gray50")
                kanvas.create_text(UKURAN_THUMBNAIL // 2, UKURAN_THUMBNAIL // 2,
                                   text="DITOLAK", fill="#ef4444", font=("Poppins", 9, "bold"))
            kanvas.bind("<Button-1>", lambda e, i=idx_abs: self._buka_di_pemeriksa(i))
            kanvas.bind("<Double-Button-1>", lambda e, p=path: self.app._lp_toggle_reject(p))
            lab = self._label_item(rel, emosi)
            ringkas = "·".join(l[:1] for l in LABELS if lab.get(l, 0) == 1) or "-"
            ctk.CTkLabel(sel, text=f"#{idx_abs+1} {emosi[:4]} [{ringkas}]", font=("Poppins", 7),
                         text_color=(WARNA_LP if aktif else ("#6b7280", "#9ca3af"))).pack()

    def terapkan_state(self, rel: str, label: dict = None, ditolak: bool = None):
        """Dipanggil app setelah menyimpan label/tolak → update lokal + render ulang TANPA
        kehilangan posisi (tidak reload seluruh daftar)."""
        if label is not None:
            self.review_label[rel] = label
        if ditolak is not None:
            if ditolak:
                self.review_ditolak.add(rel)
            else:
                self.review_ditolak.discard(rel)
        # re-apply filter: status tolak/label bisa mengubah keanggotaan daftar tersaring
        self._terapkan_filter()

    def _ganti_label_periksa(self, emosi: str):
        """Toggle label manual gambar yang sedang diperiksa → app simpan + update lokal."""
        if not self.review_items:
            return
        _path, emo_target, rel = self.review_items[self.idx_tinjau]
        label = self._label_item(rel, emo_target)
        nilai_baru = 0 if label.get(emosi, 0) == 1 else 1
        self.app._lp_set_label(rel, emosi, nilai_baru)

    def _tolak_pemeriksa(self):
        if not self.review_items:
            return
        path = self.review_items[self.idx_tinjau][0]
        self.app._lp_toggle_reject(path)

    def refresh_review(self):
        if not self._sudah_dibangun:
            self.build()
        self.app._lp_refresh_review()

    # ── API yang dipanggil app.py ───────────────────────────────────────────────

    def get_selected_emotions(self) -> list:
        return [l for l in LABELS if self.emosi_aktif[l]]

    def get_merge_mode(self) -> str:
        return self.var_merge_mode.get() if self.var_merge_mode else "lp"

    def get_target_n(self) -> int:
        try:
            return max(1, int((self.entri_jumlah.get() or "4").strip()))
        except (ValueError, AttributeError):
            return 4

    def get_picked_indices(self) -> list:
        return sorted(self.frame_terpilih)

    def get_picked_fractions(self) -> list:
        """Template posisi tertanda sebagai fraksi 0-1. Saat Batch, tiap video hasil
        memetakan fraksi ini ke index-nya sendiri → jumlah sama, posisi proporsional
        walau panjang/driving video beda. Kosong = jatuh ke 'Tandai Merata'."""
        if self.var_kunci_posisi and self.var_kunci_posisi.get() and self.fraksi_terpilih:
            return list(self.fraksi_terpilih)
        # kunci mati → pakai posisi absolut frame yang sedang tertanda, dinormalkan
        total = self._total_aktif()
        if total > 1 and self.frame_terpilih:
            return [i / (total - 1) for i in sorted(self.frame_terpilih)]
        return []

    def get_driving_folder(self) -> str:
        return self.var_folder_driving.get().strip() if self.var_folder_driving else self._folder_driving_awal

    def get_driving_choice(self, emosi: str) -> str:
        """Path video driving yang DIPILIH untuk emosi ini ('' = belum dipilih).
        Tidak ada mode 'Semua' — tiap video harus dipreview dulu (beda pose)."""
        if emosi not in self.pilihan_driving:
            return ""
        pilihan = self.pilihan_driving[emosi].get()
        if pilihan in ("(tidak ada)", ""):
            return ""
        for path in self.video_driving.get(emosi, []):
            if os.path.basename(path) == pilihan:
                return path
        return ""

    def get_driving_list(self, emosi: str) -> list:
        """Maks 1 video (yang dipilih di dropdown). Kosong = belum dipilih."""
        satu = self.get_driving_choice(emosi)
        return [satu] if satu else []

    def update_marks_count(self, jumlah: int):
        if self.label_jml_tanda:
            self.label_jml_tanda.configure(text=f"{jumlah} tertanda")

    def update_progress(self, pesan: str, warna: str = "#10b981"):
        if self.label_progres:
            self.label_progres.configure(text=pesan, text_color=warna)

    def start_loading(self, pesan: str):
        """Tampilkan indikator loading beranimasi selama proses (biar tahu tidak hang)."""
        self._loading_teks = pesan
        if self._loading_aktif:
            return
        self._loading_aktif = True
        self._loading_tick = 0
        self._animasi_loading()

    def _animasi_loading(self):
        if not self._loading_aktif:
            return
        titik = "." * (1 + self._loading_tick % 3)
        spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._loading_tick % 10]
        self.update_progress(f"{spin}  {self._loading_teks}{titik}", "#f59e0b")
        self._loading_tick += 1
        try:
            self.app.root.after(180, self._animasi_loading)
        except Exception:
            self._loading_aktif = False

    def stop_loading(self, pesan: str = "", warna: str = "#10b981"):
        """Hentikan animasi loading dan tampilkan pesan akhir."""
        self._loading_aktif = False
        if pesan:
            self.update_progress(pesan, warna)

    def set_source_frame(self, bgr, uuid: str, nomor_frame: int):
        self.sumber_bgr, self.sumber_uuid, self.sumber_frame = bgr, uuid, nomor_frame
        self._tampilkan(0, self._viz_jika_aktif(bgr))
        self.label_pratinjau[0].configure(
            text=f"UUID {uuid[:8] if uuid else '—'}  ·  frame {nomor_frame}")
        self.kanvas_pratinjau[0].configure(highlightbackground=WARNA_LP)
        self.set_source_label(uuid, nomor_frame)

    def set_source_label(self, uuid: str, nomor_frame: int):
        if self.label_sumber:
            self.label_sumber.configure(
                text=f"UUID {uuid[:8] if uuid else '—'}  ·  frame {nomor_frame}")

    def clear_source(self, pesan: str):
        self.sumber_bgr = None
        if self.label_sumber:
            self.label_sumber.configure(text=pesan)
        if self.kanvas_pratinjau:
            kanvas = self.kanvas_pratinjau[0]
            kanvas.delete("all")
            kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                               text="—", fill="#4b5563", font=("Poppins", 10))
            self.label_pratinjau[0].configure(text="")

    def set_driving_frames(self, frames: list, emosi: str, path: str = ""):
        """Tampilkan video driving. Bila `path` diberikan: jadikan driving AKTIF —
        tanda frame untuk video itu dipulihkan, dan slider bisa dipakai men-scrub
        driving SEBELUM diproses (lihat pose per frame dulu)."""
        self.frame_driving, self.emosi_driving = frames or [], emosi
        if path:
            self.driving_aktif = path
            # pulihkan tanda khusus video ini (beda video = beda pose = beda tanda)
            self.frame_terpilih = set(self.tanda_per_driving.get(path, []))
            self._simpan_fraksi()
        if self.frame_driving:
            self._tampilkan(1, self._viz_jika_aktif(self.frame_driving[0]))
            self.label_pratinjau[1].configure(text=f"{emosi}  ·  frame 1/{len(self.frame_driving)}")
            self.kanvas_pratinjau[1].configure(highlightbackground=LABEL_COLORS.get(emosi, "#6b7280"))
            # Belum ada hasil? Aktifkan slider untuk PRATINJAU driving + tandai frame.
            if not self.frame_hasil:
                total = len(self.frame_driving)
                langkah = max(total - 1, 1)
                self.slider_hasil.configure(state="normal", to=langkah, number_of_steps=langkah)
                self.slider_hasil.set(0)
                self.index_hasil = 0
                self.label_info_hasil.configure(
                    text=f"Frame 1 / {total}  ·  pratinjau driving {emosi} — geser untuk lihat pose")
            self._segarkan_tanda()
        else:
            kanvas = self.kanvas_pratinjau[1]
            kanvas.delete("all")
            kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                               text="driving\nkosong", fill="#ef4444", font=("Poppins", 9), justify="center")
            self.label_pratinjau[1].configure(text=f"{emosi}  ·  tidak ada")

    def set_result_frames(self, frames: list, emosi: str):
        self.frame_hasil, self.emosi_hasil, self.index_hasil = frames or [], emosi, 0
        total = len(self.frame_hasil)
        # Prioritas tanda frame saat hasil baru dimuat:
        # 1) tanda TERSIMPAN untuk video driving aktif (persis — hasil ~1:1 dgn driving)
        # 2) kunci posisi proporsional (fraksi) bila aktif
        # 3) kosong
        tersimpan = self.tanda_per_driving.get(self.driving_aktif, []) if self.driving_aktif else []
        if tersimpan and total > 0:
            self.frame_terpilih = {min(total - 1, max(0, i)) for i in tersimpan}
        elif self.var_kunci_posisi.get() and self.fraksi_terpilih and total > 0:
            self.frame_terpilih = {min(total - 1, max(0, int(round(f * (total - 1)))))
                                   for f in self.fraksi_terpilih}
        else:
            self.frame_terpilih.clear()
        self._segarkan_tanda()
        if not total:
            self.update_progress("Video hasil kosong / gagal dibaca", "#ef4444")
            self.slider_hasil.configure(state="disabled")
            return
        self._tampilkan(2, self._viz_jika_aktif(self.frame_hasil[0]))
        self.label_pratinjau[2].configure(text=f"{emosi}  ·  frame 1/{total}")
        self.kanvas_pratinjau[2].configure(highlightbackground=LABEL_COLORS.get(emosi, "#6b7280"))
        langkah = max(total - 1, 1)
        self.slider_hasil.configure(state="normal", to=langkah, number_of_steps=langkah)
        self.slider_hasil.set(0)
        self.label_info_hasil.configure(text=f"Frame 1 / {total}  ·  {emosi}")

    def populate_driving_menus(self, pemetaan: dict):
        self.video_driving = {l: list(pemetaan.get(l, [])) for l in LABELS}
        for emosi in LABELS:
            nama = [os.path.basename(p) for p in self.video_driving[emosi]]
            nilai = nama if nama else ["(tidak ada)"]
            self.menu_driving[emosi].configure(values=nilai)
            if self.pilihan_driving[emosi].get() not in nilai:
                self.pilihan_driving[emosi].set(nilai[0])   # default: video pertama
        self._segarkan_ringkasan()

    def reset(self):
        """Kosongkan state kerja (pilihan emosi, frame tertanda, hasil) — tanpa hapus file."""
        for emosi in LABELS:
            self.emosi_aktif[emosi] = False
            self.tombol_emosi[emosi].configure(
                fg_color="transparent", text_color=LABEL_COLORS[emosi],
                border_color=LABEL_COLORS[emosi])
        self.frame_hasil, self.frame_driving = [], []
        self.frame_terpilih.clear()
        self.index_hasil = 0
        self._segarkan_tanda()
        for kolom in range(3):
            kanvas = self.kanvas_pratinjau[kolom]
            kanvas.delete("all")
            kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                               text="—", fill="#4b5563", font=("Poppins", 10))
            self.label_pratinjau[kolom].configure(text="")
        self.slider_hasil.configure(state="disabled")
        self.label_info_hasil.configure(text="Belum ada hasil — klik 'Proses Frame Ini'")
        self.update_progress("Reset selesai.", "#6b7280")

    def cleanup(self):
        self.frame_driving, self.frame_hasil = [], []

    # ── Internal ────────────────────────────────────────────────────────────────

    def _pilih_folder_driving(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Folder video driving",
                                         initialdir=self.get_driving_folder() or None)
        if folder:
            self.var_folder_driving.set(folder)
            self._pindai_folder_driving()

    def _saat_driving_berganti(self, emosi: str):
        """Dipanggil saat dropdown driving diganti: muat PREVIEW video itu (di thread
        latar), pulihkan tanda frame untuk video itu, dan kosongkan hasil lama
        (hasil dari driving sebelumnya tidak berlaku untuk pose video baru)."""
        import threading
        path = self.get_driving_choice(emosi)
        if not path:
            return
        # simpan tanda video sebelumnya sebelum pindah
        self._simpan_tanda_driving()
        # hasil lama dari driving lain tidak relevan → kosongkan kolom HASIL
        self.frame_hasil = []
        kanvas = self.kanvas_pratinjau[2]
        kanvas.delete("all")
        kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                           text="(belum diproses\nuntuk driving ini)", fill="#4b5563",
                           font=("Poppins", 9), justify="center")
        self.label_pratinjau[2].configure(text="")
        self.start_loading(f"Memuat pratinjau driving {os.path.basename(path)}")

        def _muat(p=path, e=emosi):
            from ui.lp_panel import _decode_video
            frames = _decode_video(p)
            def _pasang():
                if self.get_driving_choice(e) != p:
                    return   # user sudah ganti lagi
                self.set_driving_frames(frames, e, path=p)
                n_tanda = len(self.tanda_per_driving.get(p, []))
                self.stop_loading(
                    f"Driving {os.path.basename(p)}: {len(frames)} frame"
                    + (f", {n_tanda} tanda dipulihkan" if n_tanda else
                       " — geser slider untuk lihat pose, lalu Tandai Frame"),
                    "#10b981")
            try:
                self.app.root.after(0, _pasang)
            except (RuntimeError, tk.TclError):
                pass
        threading.Thread(target=_muat, daemon=True).start()

    def _pindai_folder_driving(self):
        pemetaan = self.app._lp_scan_driving(self.get_driving_folder())
        self.populate_driving_menus(pemetaan)
        total = sum(len(v) for v in pemetaan.values())
        self.update_progress(
            f"Folder driving: {total} video — "
            + "  ".join(f"{l[:4]}:{len(pemetaan.get(l, []))}" for l in LABELS),
            "#10b981" if total else "#f59e0b")

    def _ganti_emosi(self, emosi: str):
        """SATU emosi aktif pada satu waktu (single-select). Generate memang bekerja
        per-emosi dengan satu video driving yang harus dipreview dulu — multi-pilih
        hanya membingungkan. Memilih emosi langsung memuat preview driving-nya."""
        aktif = not self.emosi_aktif[emosi]
        for e in LABELS:
            self.emosi_aktif[e] = False
            self.tombol_emosi[e].configure(fg_color="transparent",
                                           text_color=LABEL_COLORS[e],
                                           border_color=LABEL_COLORS[e])
        if aktif:
            warna = LABEL_COLORS[emosi]
            self.emosi_aktif[emosi] = True
            self.tombol_emosi[emosi].configure(fg_color=warna, text_color="#0b0b12",
                                               border_color=warna)
            self._saat_driving_berganti(emosi)   # langsung muat preview driving emosi ini
        self._segarkan_ringkasan()

    # ── Dataset wajah baru ──────────────────────────────────────────────────────

    def get_face_folder(self) -> str:
        return self.var_folder_wajah.get().strip() if self.var_folder_wajah else ""

    def get_selected_faces(self) -> list:
        return sorted(self.wajah_terpilih)

    def _pilih_folder_wajah(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Folder dataset wajah",
                                         initialdir=self.get_face_folder() or None)
        if folder:
            self.var_folder_wajah.set(folder)
            self._pindai_folder_wajah()

    def _pindai_folder_wajah(self):
        self.wajah_terpilih.clear()
        self.app._lp_refresh_faces()   # app baca folder + decode thumbnail di thread latar

    def _pilih_semua_wajah(self):
        self.wajah_terpilih = set(self.wajah_paths)
        self.render_faces([(p, p in self.wajah_terpilih, None) for p in self.wajah_paths],
                          dari_cache=True)

    def _kosongkan_pilihan_wajah(self):
        self.wajah_terpilih.clear()
        self.render_faces([(p, False, None) for p in self.wajah_paths], dari_cache=True)

    def _toggle_pilih_wajah(self, path: str):
        if path in self.wajah_terpilih:
            self.wajah_terpilih.discard(path)
        else:
            self.wajah_terpilih.add(path)
        self._segarkan_label_wajah()
        self.render_faces([(p, p in self.wajah_terpilih, None) for p in self.wajah_paths],
                          dari_cache=True)

    def _segarkan_label_wajah(self):
        if hasattr(self, "label_jml_wajah"):
            self.label_jml_wajah.configure(text=f"{len(self.wajah_terpilih)} dipilih")

    def _segarkan_ringkasan(self, *_):
        """Perbarui baris ringkasan setelan aktif (emosi · driving · frame hasil)."""
        if not self.label_ringkasan_wajah:
            return
        terpilih = self.get_selected_emotions()
        emos = ", ".join(e[:4] for e in terpilih) or "— (belum dipilih)"
        bagian_driving = []
        for e in terpilih:
            pilih = self.pilihan_driving[e].get()
            if pilih in ("(tidak ada)", ""):
                bagian_driving.append(f"{e[:4]}→(belum dipilih)")
            else:
                bagian_driving.append(f"{e[:4]}→{pilih}")
        driving = "  ".join(bagian_driving) or "—"
        frame_hasil = (", ".join(str(i + 1) for i in sorted(self.frame_terpilih))
                       if self.frame_terpilih else f"{self.get_target_n()} merata")
        self.label_ringkasan_wajah.configure(
            text=f"Setelan aktif — Emosi: {emos}   ·   Driving: {driving}   ·   "
                 f"Frame hasil: {frame_hasil}")

    def render_faces(self, item_wajah: list, dari_cache: bool = False, semua_path: list = None):
        """Render grid thumbnail wajah. item: (path, terpilih, thumb_bgr).
        dari_cache=True → pakai thumbnail yang sudah tersimpan (toggle pilih, tanpa decode ulang).
        semua_path → daftar LENGKAP file di folder (bisa > yang ditampilkan); 'Pilih Semua'
        memilih daftar lengkap ini, bukan hanya thumbnail yang tampil."""
        import cv2
        if not dari_cache:
            self.wajah_paths = list(semua_path) if semua_path else [p for (p, _s, _t) in item_wajah]
            self._thumb_wajah_cache = {p: t for (p, _s, t) in item_wajah}
        for anak in self.grid_wajah.winfo_children():
            anak.destroy()
        self._ref_thumb_wajah.clear()
        if not item_wajah:
            ctk.CTkLabel(self.grid_wajah, text="(folder kosong / belum dipindai)",
                         font=("Poppins", 9), text_color=("#6b7280", "#9ca3af")).pack(pady=8)
            self._segarkan_label_wajah()
            return
        KOLOM = 5
        UK = 120
        cache = getattr(self, "_thumb_wajah_cache", {})
        for nomor, (path, terpilih, thumb) in enumerate(item_wajah):
            if thumb is None:
                thumb = cache.get(path)
            if thumb is None:
                continue
            r, c = divmod(nomor, KOLOM)
            sel = tk.Frame(self.grid_wajah, bg="#161622")
            sel.grid(row=r, column=c, padx=3, pady=3)
            warna_tepi = WARNA_LP if terpilih else "#333"
            kanvas = tk.Canvas(sel, width=UK, height=UK, bg="#111", highlightthickness=3,
                               highlightbackground=warna_tepi, cursor="hand2")
            kanvas.pack()
            rgb = cv2.cvtColor(_letterbox(thumb, UK, UK), cv2.COLOR_BGR2RGB)
            gambar = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._ref_thumb_wajah.append(gambar)
            kanvas.create_image(0, 0, anchor="nw", image=gambar)
            if terpilih:
                kanvas.create_text(UK - 14, 14, text="✓", fill=WARNA_LP, font=("Poppins", 14, "bold"))
            kanvas.bind("<Button-1>", lambda e, p=path: self._toggle_pilih_wajah(p))
            ctk.CTkLabel(sel, text=os.path.basename(path)[:16], font=("Poppins", 7),
                         text_color=("#6b7280", "#9ca3af")).pack()
        sisa = len(self.wajah_paths) - len(self._thumb_wajah_cache)
        if sisa > 0:
            ctk.CTkLabel(
                self.grid_wajah,
                text=f"(+{sisa} foto lain tidak ditampilkan — 'Pilih Semua' tetap memilih semuanya)",
                font=("Poppins", 8), text_color=("#d97706", "#fbbf24"),
            ).grid(row=999, column=0, columnspan=5, sticky="w", padx=4, pady=(4, 0))
        self._segarkan_label_wajah()

    def _unduh_gambar(self, kolom: int):
        """Simpan gambar yang sedang tampil di kolom pratinjau ke file pilihan user."""
        import cv2
        bgr = self.bgr_tampil[kolom]
        if bgr is None:
            self.update_progress("Tidak ada gambar untuk diunduh di kolom itu", "#f59e0b")
            return
        from tkinter import filedialog
        nama_awal = ["sumber", "driving", "hasil"][kolom]
        path = filedialog.asksaveasfilename(
            title="Simpan gambar", defaultextension=".jpg", initialfile=f"{nama_awal}.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path:
            cv2.imwrite(path, bgr)
            self.update_progress(f"Gambar disimpan: {os.path.basename(path)}", "#10b981")
