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
UKURAN_PRATINJAU = 300         # sisi kotak gambar pratinjau (besar agar mudah dicek)
UKURAN_THUMBNAIL = 160         # sisi thumbnail di grid tinjau
UKURAN_PEMERIKSA = 340         # sisi gambar besar di pemeriksa hasil
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

    def __init__(self, induk, app):
        self.induk = induk
        self.app   = app
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
        self.frame_terpilih = set()  # index frame hasil yang ditandai untuk diambil

        # Gambar BGR yang sedang tampil di tiap kolom pratinjau (untuk tombol Unduh)
        self.bgr_tampil = [None, None, None]   # 0=sumber, 1=driving, 2=hasil

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

        # --- Tinjau & label hasil ---
        self.item_tinjau  = []         # [(path, emosi, ditolak, label, thumb_bgr)]
        self.index_tinjau = 0
        self._ref_thumbnail = []       # tahan referensi ImageTk agar tidak di-GC

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
        for emosi in LABELS:
            self.pilihan_driving[emosi] = tk.StringVar(value="Semua")

        area = ctk.CTkScrollableFrame(self.induk, fg_color=("#f3f4f6", "#161622"),
                                      corner_radius=0)
        area.pack(fill="both", expand=True)
        _aktifkan_scroll_mouse(area)

        self._bangun_header_sumber(area)
        self._bangun_konfig_driving(area)
        self._bangun_pemilih_emosi(area)
        self._bangun_tombol_proses(area)
        self._bangun_pratinjau(area)
        self._bangun_pemilih_frame(area)
        self._bangun_tinjau_label(area)

        self._pindai_folder_driving()   # isi menu driving otomatis

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
        self.label_jml_tanda = ctk.CTkLabel(kartu, text="0 tertanda", font=("Poppins", 8),
                                            text_color=("#6b7280", "#9ca3af"))
        self.label_jml_tanda.pack(side="right", padx=(0, 8))

    def _bangun_konfig_driving(self, induk):
        _judul_seksi(induk, "Folder Video Driving (acuan ekspresi)")
        ctk.CTkLabel(
            induk,
            text="Taruh video acuan di folder ini. Nama file menentukan emosi + urutan: "
                 "confuse1.mp4, confuse2.mp4, frustration1.mp4, boredom1.mp4 …",
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

        for emosi in LABELS:
            r = ctk.CTkFrame(induk, fg_color="transparent")
            r.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(r, text=emosi[:4], font=("Poppins", 9, "bold"),
                         text_color=LABEL_COLORS[emosi], width=44, anchor="w").pack(side="left")
            menu = ctk.CTkOptionMenu(
                r, variable=self.pilihan_driving[emosi], values=["Semua"],
                font=("Poppins", 9), height=26, width=300, corner_radius=6,
                fg_color=("#e5e7eb", "#23233a"), button_color=LABEL_COLORS[emosi],
                button_hover_color="#d97706")
            menu.pack(side="left", padx=(4, 0))
            self.menu_driving[emosi] = menu

    def _bangun_pemilih_emosi(self, induk):
        _judul_seksi(induk, "Emosi yang Diproses")
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

    def _bangun_tinjau_label(self, induk):
        _judul_seksi(induk, "Tinjau & Label Hasil  (cek sebelum merge)")
        bar = ctk.CTkFrame(induk, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkButton(bar, text="Muat / Refresh", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#3b82f6", hover_color="#2563eb",
                      command=self.refresh_review).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Label Semua (AI)", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#6366f1", hover_color="#4f46e5",
                      command=self.app._lp_label_all_ai).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Hapus Ditolak", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#ef4444", hover_color="#dc2626",
                      command=self.app._lp_delete_rejected).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Merge ke Dataset", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color="#10b981", hover_color="#0ea372",
                      command=self.app._lp_merge_into_label2d).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Undo Merge", height=28, corner_radius=8,
                      font=("Poppins", 9, "bold"), fg_color=("#e5e7eb", "#2a2a3a"),
                      hover_color=("#d1d5db", "#3a3a4a"), text_color=("#374151", "#9ca3af"),
                      command=self.app._lp_undo_merge).pack(side="left")
        ctk.CTkLabel(
            induk, text="Klik 1x thumbnail = lihat besar & label manual.  Klik 2x = tolak/terima "
                        "(merah = ditolak, tidak ikut merge).",
            font=("Poppins", 8), text_color=("#6b7280", "#9ca3af"),
            wraplength=680, justify="left").pack(fill="x", padx=14, pady=(2, 4))

        # Pemeriksa: gambar BESAR + pill label manual + tombol tolak
        kartu = ctk.CTkFrame(induk, fg_color=("#f3f4f6", "#1a1a2e"), corner_radius=10)
        kartu.pack(fill="x", padx=12, pady=(0, 6))
        kiri = ctk.CTkFrame(kartu, fg_color="transparent")
        kiri.pack(side="left", padx=10, pady=10)
        self.kanvas_pemeriksa = tk.Canvas(kiri, width=UKURAN_PEMERIKSA, height=UKURAN_PEMERIKSA,
                                          bg="#111", highlightthickness=2, highlightbackground="#2d2d40")
        self.kanvas_pemeriksa.pack()
        self.kanvas_pemeriksa.bind("<Double-Button-1>", lambda e: self._tolak_pemeriksa())
        self.kanvas_pemeriksa.create_text(UKURAN_PEMERIKSA // 2, UKURAN_PEMERIKSA // 2,
                                          text="pilih thumbnail", fill="#4b5563", font=("Poppins", 10))
        kanan = ctk.CTkFrame(kartu, fg_color="transparent")
        kanan.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self.label_pemeriksa = ctk.CTkLabel(kanan, text="—", font=("Poppins", 9, "bold"),
                                            text_color=("#374151", "#e5e7eb"), anchor="w",
                                            justify="left", wraplength=300)
        self.label_pemeriksa.pack(anchor="w", pady=(0, 6))
        ctk.CTkLabel(kanan, text="Label final (yang dipakai saat merge):", font=("Poppins", 8),
                     text_color=("#6b7280", "#9ca3af")).pack(anchor="w")
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

        # Grid thumbnail untuk navigasi cepat
        self.grid_tinjau = ctk.CTkFrame(induk, fg_color="transparent")
        self.grid_tinjau.pack(fill="both", expand=True, padx=10, pady=(0, 12))

    # ── Menampilkan gambar (semua dari memori → tanpa lag, tanpa crash) ─────────

    def _tampilkan(self, kolom: int, bgr):
        """Tampilkan array BGR ke kanvas pratinjau ke-`kolom` (0=sumber,1=driving,2=hasil)."""
        import cv2
        self.bgr_tampil[kolom] = bgr
        rgb = cv2.cvtColor(cv2.resize(bgr, (UKURAN_PRATINJAU, UKURAN_PRATINJAU)), cv2.COLOR_BGR2RGB)
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

    def _geser_slider(self, nilai):
        """Saat slider digeser: tampilkan frame mentah (instan), viz dihitung belakangan."""
        if not self.frame_hasil:
            return
        index = int(float(nilai))
        self.index_hasil = index
        total = len(self.frame_hasil)
        tanda = "  ●ditandai" if index in self.frame_terpilih else ""
        self.label_info_hasil.configure(text=f"Frame {index+1} / {total}  ·  {self.emosi_hasil}{tanda}")
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

    def _tandai_frame_sekarang(self):
        if not self.frame_hasil:
            return
        self.frame_terpilih.add(self.index_hasil)
        self._segarkan_tanda()

    def _tandai_merata(self):
        if not self.frame_hasil:
            return
        jumlah = self.get_target_n()
        total = len(self.frame_hasil)
        self.frame_terpilih = {int(round(i * (total - 1) / max(jumlah - 1, 1))) for i in range(jumlah)}
        self._segarkan_tanda()

    def _bersihkan_tanda(self):
        self.frame_terpilih.clear()
        self._segarkan_tanda()

    def _segarkan_tanda(self):
        if not self.frame_terpilih:
            self.label_frame_terpilih.configure(text="Frame tertanda: (kosong)")
        else:
            urut = ", ".join(str(i + 1) for i in sorted(self.frame_terpilih))
            self.label_frame_terpilih.configure(
                text=f"Frame tertanda ({len(self.frame_terpilih)}): {urut}")
        self._geser_slider(self.index_hasil)

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

    def render_review(self, item_tinjau: list):
        """Render grid thumbnail hasil. `item_tinjau`: list (path, emosi, ditolak, label, thumb_bgr).

        thumb_bgr sudah di-decode di thread latar oleh app.py → di sini hanya konversi cepat
        ke gambar Tk (tidak ada pembacaan file di thread UI → tidak ada lag).
        """
        import cv2
        self.item_tinjau = item_tinjau
        for anak in self.grid_tinjau.winfo_children():
            anak.destroy()
        self._ref_thumbnail.clear()
        if not item_tinjau:
            ctk.CTkLabel(self.grid_tinjau, text="(belum ada hasil generate)",
                         font=("Poppins", 9), text_color=("#6b7280", "#9ca3af")).pack(pady=10)
            return
        KOLOM = 4
        for nomor, (path, emosi, ditolak, label, thumb) in enumerate(item_tinjau):
            r, c = divmod(nomor, KOLOM)
            sel = tk.Frame(self.grid_tinjau, bg="#161622")
            sel.grid(row=r, column=c, padx=4, pady=4)
            warna_tepi = "#ef4444" if ditolak else LABEL_COLORS.get(emosi, "#333")
            kanvas = tk.Canvas(sel, width=UKURAN_THUMBNAIL, height=UKURAN_THUMBNAIL, bg="#111",
                               highlightthickness=2, highlightbackground=warna_tepi, cursor="hand2")
            kanvas.pack()
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            gambar = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._ref_thumbnail.append(gambar)
            kanvas.create_image(0, 0, anchor="nw", image=gambar)
            if ditolak:
                kanvas.create_rectangle(0, 0, UKURAN_THUMBNAIL, UKURAN_THUMBNAIL,
                                        fill="#2a0000", stipple="gray50")
                kanvas.create_text(UKURAN_THUMBNAIL // 2, UKURAN_THUMBNAIL // 2,
                                   text="DITOLAK", fill="#ef4444", font=("Poppins", 10, "bold"))
            kanvas.bind("<Button-1>", lambda e, n=nomor: self._buka_di_pemeriksa(n))
            kanvas.bind("<Double-Button-1>", lambda e, p=path: self.app._lp_toggle_reject(p))
            label_ringkas = "·".join(l[:1] for l in LABELS if label.get(l, 0) == 1) or "-"
            ctk.CTkLabel(sel, text=f"{emosi[:4]}  [{label_ringkas}]", font=("Poppins", 7),
                         text_color=("#6b7280", "#9ca3af")).pack()
        # tampilkan item pertama di pemeriksa
        self._buka_di_pemeriksa(min(self.index_tinjau, len(item_tinjau) - 1))

    def _buka_di_pemeriksa(self, nomor: int):
        """Tampilkan satu gambar hasil BESAR di pemeriksa + status label/tolaknya."""
        import cv2
        if not self.item_tinjau or nomor >= len(self.item_tinjau):
            return
        self.index_tinjau = nomor
        path, emosi, ditolak, label, thumb = self.item_tinjau[nomor]
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = thumb
        rgb = cv2.cvtColor(cv2.resize(bgr, (UKURAN_PEMERIKSA, UKURAN_PEMERIKSA)), cv2.COLOR_BGR2RGB)
        gambar = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._ref_periksa = gambar
        self.kanvas_pemeriksa.delete("all")
        self.kanvas_pemeriksa.create_image(0, 0, anchor="nw", image=gambar)
        self.kanvas_pemeriksa.configure(highlightbackground="#ef4444" if ditolak
                                        else LABEL_COLORS.get(emosi, "#333"))
        if ditolak:
            self.kanvas_pemeriksa.create_rectangle(0, 0, UKURAN_PEMERIKSA, UKURAN_PEMERIKSA,
                                                   fill="#2a0000", stipple="gray50")
            self.kanvas_pemeriksa.create_text(UKURAN_PEMERIKSA // 2, 20, text="DITOLAK",
                                              fill="#ef4444", font=("Poppins", 11, "bold"))
        self.label_pemeriksa.configure(
            text=f"{nomor+1}/{len(self.item_tinjau)}  ·  target: {emosi}\n{os.path.basename(path)}")
        for emo2, pill in self.pill_label_periksa.items():
            warna = LABEL_COLORS[emo2]
            if label.get(emo2, 0) == 1:
                pill.configure(fg_color=warna, text_color="#0b0b12", border_color=warna)
            else:
                pill.configure(fg_color="transparent", text_color=warna, border_color=warna)
        self.tombol_tolak_periksa.configure(
            text="Batal Tolak" if ditolak else "Tolak Gambar Ini")

    def _ganti_label_periksa(self, emosi: str):
        """Toggle label manual untuk gambar yang sedang diperiksa (lalu app simpan + refresh)."""
        if not self.item_tinjau or self.index_tinjau >= len(self.item_tinjau):
            return
        path, _emo, _tolak, label, _thumb = self.item_tinjau[self.index_tinjau]
        gen_dir = self.app._lp_generated_dir() or ""
        rel = os.path.relpath(path, gen_dir).replace("\\", "/") if gen_dir else path
        nilai_baru = 0 if label.get(emosi, 0) == 1 else 1
        self.app._lp_set_label(rel, emosi, nilai_baru)

    def _tolak_pemeriksa(self):
        if not self.item_tinjau or self.index_tinjau >= len(self.item_tinjau):
            return
        path = self.item_tinjau[self.index_tinjau][0]
        self.app._lp_toggle_reject(path)

    def refresh_review(self):
        if not self._sudah_dibangun:
            self.build()
        self.app._lp_refresh_review()

    # ── API yang dipanggil app.py ───────────────────────────────────────────────

    def get_selected_emotions(self) -> list:
        return [l for l in LABELS if self.emosi_aktif[l]]

    def get_target_n(self) -> int:
        try:
            return max(1, int((self.entri_jumlah.get() or "4").strip()))
        except (ValueError, AttributeError):
            return 4

    def get_picked_indices(self) -> list:
        return sorted(self.frame_terpilih)

    def get_driving_folder(self) -> str:
        return self.var_folder_driving.get().strip() if self.var_folder_driving else self._folder_driving_awal

    def get_driving_choice(self, emosi: str) -> str:
        """'' = pakai SEMUA video emosi itu; selain itu = path video yang dipilih."""
        if emosi not in self.pilihan_driving:
            return ""
        pilihan = self.pilihan_driving[emosi].get()
        if pilihan in ("Semua", "(tidak ada)", ""):
            return ""
        for path in self.video_driving.get(emosi, []):
            if os.path.basename(path) == pilihan:
                return path
        return ""

    def get_driving_list(self, emosi: str) -> list:
        satu = self.get_driving_choice(emosi)
        return [satu] if satu else list(self.video_driving.get(emosi, []))

    def update_marks_count(self, jumlah: int):
        if self.label_jml_tanda:
            self.label_jml_tanda.configure(text=f"{jumlah} tertanda")

    def update_progress(self, pesan: str, warna: str = "#10b981"):
        if self.label_progres:
            self.label_progres.configure(text=pesan, text_color=warna)

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

    def set_driving_frames(self, frames: list, emosi: str):
        self.frame_driving, self.emosi_driving = frames or [], emosi
        if self.frame_driving:
            self._tampilkan(1, self._viz_jika_aktif(self.frame_driving[0]))
            self.label_pratinjau[1].configure(text=f"{emosi}  ·  frame 1/{len(self.frame_driving)}")
            self.kanvas_pratinjau[1].configure(highlightbackground=LABEL_COLORS.get(emosi, "#6b7280"))
        else:
            kanvas = self.kanvas_pratinjau[1]
            kanvas.delete("all")
            kanvas.create_text(UKURAN_PRATINJAU // 2, UKURAN_PRATINJAU // 2,
                               text="driving\nkosong", fill="#ef4444", font=("Poppins", 9), justify="center")
            self.label_pratinjau[1].configure(text=f"{emosi}  ·  tidak ada")

    def set_result_frames(self, frames: list, emosi: str):
        self.frame_hasil, self.emosi_hasil, self.index_hasil = frames or [], emosi, 0
        self.frame_terpilih.clear()
        self._segarkan_tanda()
        total = len(self.frame_hasil)
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
            nilai = ["Semua"] + nama if nama else ["(tidak ada)"]
            self.menu_driving[emosi].configure(values=nilai)
            if self.pilihan_driving[emosi].get() not in nilai:
                self.pilihan_driving[emosi].set(nilai[0])

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

    def _pindai_folder_driving(self):
        pemetaan = self.app._lp_scan_driving(self.get_driving_folder())
        self.populate_driving_menus(pemetaan)
        total = sum(len(v) for v in pemetaan.values())
        self.update_progress(
            f"Folder driving: {total} video — "
            + "  ".join(f"{l[:4]}:{len(pemetaan.get(l, []))}" for l in LABELS),
            "#10b981" if total else "#f59e0b")

    def _ganti_emosi(self, emosi: str):
        aktif = not self.emosi_aktif[emosi]
        self.emosi_aktif[emosi] = aktif
        tombol, warna = self.tombol_emosi[emosi], LABEL_COLORS[emosi]
        if aktif:
            tombol.configure(fg_color=warna, text_color="#0b0b12", border_color=warna)
        else:
            tombol.configure(fg_color="transparent", text_color=warna, border_color=warna)

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
