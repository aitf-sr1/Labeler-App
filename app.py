"""
app.py

Entry point dan kelas utama VideoLabelerApp.
Bertanggung jawab atas orkestrasi: routing event, state management,
dan koordinasi antar modul.

Modul eksternal:
    core/   -- SigLIP model, face detector, inferensi
    utils/  -- ekstraksi frame, IO file CSV/JSON
    ui/     -- panel kiri, panel kanan, konstanta

Flow utama:
    open_folder()        -- buka dataset, inisialisasi path, load data
    load_video()         -- tampilkan video ke-N, restore state anotasi
    save_current_state() -- simpan label + flag ke disk
    save_and_next()      -- save lalu pindah ke video berikutnya
"""

import os
import glob
import time
import threading
import concurrent.futures
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from dotenv import load_dotenv

load_dotenv()

# Baca konfigurasi LP dari 4-Create/.env (non-override: app .env tetap prioritas)
_lp_env = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "4-Create", ".env")
)
if os.path.exists(_lp_env):
    load_dotenv(_lp_env, override=False)

from ui import LABELS, LABEL_COLORS, LeftPanel, RightPanel, RulesPanel

# Pasangan label yang saling eksklusif: jika satu = 1, pasangannya harus 0
MUTUAL_EXCLUSIVE = {
    "Confusion": "Frustration",
    "Frustration": "Confusion",
    "Boredom": "Engagement",
    "Engagement": "Boredom",
}
from utils.io import (
    load_annotations, save_annotations,
    load_flagged, save_flagged,
    load_frame_annotations, save_frame_annotations,
    load_batch_history, save_batch_history,
    load_batch_meta, update_batch_meta,
    load_skipped, save_skipped,
    load_thresholds, save_thresholds,
)
from core.rules import DEFAULT_RULES, load_rules, save_rules
# Heavy modules (cv2, torch, mediapipe) are imported lazily inside the methods
# that need them — this keeps app startup fast (~1-2s saved).

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VideoLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Labeler App — Pelabelan Emosi")
        self.root.geometry("1280x850")
        self.root.minsize(1100, 750)

        self.font_main = ("Poppins", 11)
        self.font_bold = ("Poppins", 11, "bold")
        self.font_sm   = ("Poppins", 10)

        # Path file output — diisi saat open_folder()
        self.video_files, self.current_index = [], 0
        self.root_folder = ""
        self.path_csv_annotations    = ""
        self.path_csv_flagged        = ""
        self.path_json_frames        = ""
        self.path_dir_cropped        = ""
        self.path_json_batch_history = ""
        self.path_json_skipped       = ""
        self.path_json_thresholds    = ""
        self.path_json_rules         = ""
        self.path_dir_raw_cache      = ""
        self.path_dir_siglip_cache   = ""

        # Rules — parameter landmark & hybrid
        self.rules = DEFAULT_RULES

        self.cap, self.is_playing = None, False
        self.total_frames, self.current_frame, self.after_id = 0, 0, None
        self.fps, self.play_start_time, self.play_start_frame = 30.0, 0.0, 0

        self.flagged_data      = set()
        self.frame_annotations = {}
        self.skipped_videos    = set()
        self.batch_history     = {}

        self.flag_var        = ctk.BooleanVar(value=False)
        self.semi_manual_var = ctk.BooleanVar(value=False)
        self.manual_labels   = {}   # {rel_path: {"0": {lbl: 0|1}, "1": {...}}}
        # Penanda augmentasi LivePortrait (dibaca notebook 4-Create). Lihat _load_augment_marks.
        self.augment_marks   = {"reference_images": [],
                                "driving_images": {l: [] for l in LABELS}}
        self.active_frame_label = ctk.StringVar(value="Boredom")
        self.show_viz   = ctk.BooleanVar(value=False)   # Toggle landmark viz di galeri
        self.viz_images = []   # Cache viz PIL images untuk video aktif

        self.batch_running, self.cancel_batch = False, False
        self._shutting_down = False  # diset True saat window ditutup (stop worker daemon rapi)
        self._viz_regen_requested = False
        self.save_lock = threading.RLock()

        # Gallery background-loading state
        self._gallery_version = 0
        self._gallery_cache: dict = {
            "rel_path": None, "pil_images": [], "viz_images": [],
            "landmark_results": [], "no_face_count": 0, "multi_fc": 0,
        }

        self._rules_panel = None   # backwards compat attribute

        # LP Transform state
        self._lp_cancel_flag   = False
        self._lp_current_source = None   # (vid_idx, frame_idx) frame aktif LP
        self._lp_worker_proc   = None    # subprocess LP persisten (model load sekali)
        self._lp_worker_lock   = threading.Lock()
        self._lp_busy          = False   # cegah proses LP tumpang-tindih (anti pile-up)
        # Cache hasil LP per frame sumber → tampil lagi saat balik ke frame itu.
        # Hanya menyimpan PATH video output + driving (hemat RAM), maks 32 entri.
        self._lp_result_cache  = {}      # (vid_idx, fi) -> {"emo","video","driving"}
        self._lp_result_cache_order = []
        # Satu lock untuk SEMUA baca-ubah-tulis file lp_labels / lp_ai_labels / lp_review
        # (UI thread dan worker batch menulis file yang sama → tanpa lock, update bisa hilang).
        self._lp_json_lock     = threading.RLock()
        # Throttle update UI saat batch (anti-lag): refresh grid tinjau & preview dibatasi
        # frekuensinya — kalau tidak, batch dari cache (instan) membanjiri ratusan refresh.
        self._lp_last_review_ts  = 0.0
        self._lp_last_preview_ts = 0.0

        self._build_ui()
        # Initialize inline rules content in left panel (needs threshold_vars from right panel)
        self.left_panel.init_rules_content(
            rules=self.rules,
            threshold_vars=self.right_panel.threshold_vars,
            on_save=self._save_rules,
            on_recalculate=self._recalculate_all,
            on_rebatch=self._toggle_batch,
        )
        self.left_panel.init_lp_content(self)

    # UI assembly

    def _build_ui(self):
        """Susun layout grid utama dan bind keyboard shortcut."""
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=0, minsize=310)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        self._build_topbar()
        self.left_panel  = LeftPanel(self.root, self)
        self.right_panel = RightPanel(self.root, self)
        self._build_bottombar()

        self.root.bind("<space>", lambda e: self.toggle_play())
        # Panah ◀/▶ sadar-mode: di galeri = pindah video (save & next); di panel LP =
        # pindah gambar di pemeriksa hasil (review ribuan gambar tanpa klik mouse).
        # bind_all (BUKAN root.bind): tangkap di mana pun fokus berada — root.bind hanya
        # menyala bila event sempat menggelembung ke root, sering ditelan widget panel.
        self.root.bind_all("<Right>", lambda e: self._on_arrow(+1))
        self.root.bind_all("<Left>",  lambda e: self._on_arrow(-1))

    def _build_topbar(self):
        """Bangun bar atas: tombol buka folder, label nama video, dan progress counter."""
        bar = ctk.CTkFrame(self.root, fg_color=("#f0f2f5", "#13131d"), corner_radius=0, height=48)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Buka Folder", command=self.open_folder,
            font=self.font_bold, fg_color="#10b981", hover_color="#059669",
            width=110, height=32, corner_radius=8,
        ).pack(side="left", padx=(14, 4), pady=8)

        ctk.CTkButton(
            bar, text="Output", command=self._change_output_folder,
            font=self.font_sm,
            fg_color=("#e5e7eb", "#2a2a3a"), hover_color=("#d1d5db", "#3a3a4a"),
            text_color=("#374151", "#9ca3af"),
            width=80, height=32, corner_radius=8,
        ).pack(side="left", padx=(0, 14), pady=8)

        # Subtle separator
        ctk.CTkFrame(bar, fg_color=("#d1d5db", "#2a2a3a"), width=1, height=24).pack(
            side="left", padx=(0, 10), pady=12
        )

        self.lbl_info = ctk.CTkLabel(
            bar, text="Pilih folder dataset untuk memulai",
            font=self.font_sm, text_color=("#6b7280", "#6b7280"),
        )
        self.lbl_info.pack(side="left")

        self.lbl_fps = ctk.CTkLabel(bar, text="", font=self.font_sm, text_color="#10b981")
        self.lbl_fps.pack(side="left", padx=16)

        self.lbl_progress = ctk.CTkLabel(bar, text="", font=self.font_sm, text_color=("#4b5563", "#9ca3af"))
        self.lbl_progress.pack(side="right", padx=16)

        self.lbl_flag_count = ctk.CTkLabel(
            bar, text="", font=self.font_sm, text_color="#ef4444"
        )
        self.lbl_flag_count.pack(side="right", padx=(0, 4))

        ctk.CTkSwitch(
            bar, text="Viz", variable=self.show_viz,
            font=self.font_sm, progress_color="#6366f1",
            command=self.refresh_frame_gallery,
        ).pack(side="right", padx=(0, 14))

    def _build_bottombar(self):
        """Bangun bar bawah: navigasi prev/skip/next, toggle flag, dan jump to video."""
        bar = ctk.CTkFrame(self.root, fg_color=("#f0f2f5", "#13131d"), corner_radius=0, height=48)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Prev", command=self.prev_video,
            font=self.font_sm, fg_color=("#9ca3af", "#3b3b52"),
            hover_color=("#6b7280", "#374151"), width=90, height=32, corner_radius=8,
        ).pack(side="left", padx=(14, 5), pady=9)

        ctk.CTkButton(
            bar, text="Skip", command=self.skip_video,
            font=self.font_sm, fg_color=("#f59e0b", "#b45309"),
            hover_color=("#d97706", "#92400e"), width=90, height=32, corner_radius=8,
        ).pack(side="left", padx=(0, 16), pady=9)

        self.chk_flag = ctk.CTkSwitch(
            bar, text="Flag / Reject", variable=self.flag_var,
            font=self.font_sm, progress_color="#ef4444",
            command=self._on_flag_toggle,
        )
        self.chk_flag.pack(side="left", padx=(0, 16))

        self.chk_semi_manual = ctk.CTkSwitch(
            bar, text="Label Semi Manual", variable=self.semi_manual_var,
            font=self.font_sm, progress_color="#10b981",
            command=self._on_semi_manual_toggle,
        )
        self.chk_semi_manual.pack(side="left")

        ctk.CTkButton(
            bar, text="Save & Next", command=self.save_and_next,
            font=self.font_bold, fg_color="#3b82f6", hover_color="#2563eb",
            width=130, height=32, corner_radius=8,
        ).pack(side="right", padx=14, pady=9)

        ctk.CTkButton(
            bar, text="Go", command=self.jump_to_video,
            font=self.font_sm, fg_color=("#6b7280", "#374151"),
            hover_color=("#4b5563", "#1f2937"), width=40, height=32, corner_radius=8,
        ).pack(side="right", padx=(0, 4), pady=9)

        self.jump_entry = ctk.CTkEntry(
            bar, width=52, height=32, font=self.font_sm,
            placeholder_text="No.", corner_radius=8,
        )
        self.jump_entry.pack(side="right", pady=9)
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_video())

        ctk.CTkLabel(
            bar, text="Loncat ke:", font=("Poppins", 9), text_color=("#6b7280", "#6b7280")
        ).pack(side="right", padx=(0, 4))

    # Folder & data

    def open_folder(self):
        """
        Buka dialog folder, inisialisasi semua path output, scan video, dan load data.

        Semua file output disimpan di subfolder hasil_label6/ di dalam folder yang dipilih.
        """
        folder = filedialog.askdirectory()
        if not folder: return
        self.root_folder = folder
        
        # Mengambil nama folder output dari .env (default: hasil_label6)
        output_name = os.getenv("OUTPUT_DIR", "hasil_label6")
        # OUTPUT_DIR absolut → pakai langsung (output selalu ke lokasi itu, lepas dari root
        # yang dibuka). OUTPUT_DIR relatif → digabung dengan folder root (perilaku lama).
        base = output_name if os.path.isabs(output_name) else os.path.join(folder, output_name)

        self.path_csv_annotations    = os.path.join(base, "annotations_bener.csv")
        self.path_csv_flagged        = os.path.join(base, "flagged_videos.csv")
        self.path_json_frames        = os.path.join(base, "frame_annotations.json")
        self.path_json_batch_history = os.path.join(base, "batch_history.json")
        self.path_json_skipped       = os.path.join(base, "skipped_videos.json")
        self.path_json_thresholds    = os.path.join(base, "thresholds.json")
        self.path_json_rules         = os.path.join(base, "rules.json")
        # path_json_manual dihapus — pakai _manual_path_for() secara dinamis
        self.path_json_augment       = os.path.join(base, "augment_marks.json")
        self.path_dir_cropped        = os.path.join(base, "cropped_faces")
        self.path_dir_raw_cache      = os.path.join(base, "raw_cache")
        self.path_dir_siglip_cache   = os.path.join(base, "siglip_cache")
        os.makedirs(self.path_dir_cropped, exist_ok=True)
        os.makedirs(self.path_dir_raw_cache, exist_ok=True)
        os.makedirs(self.path_dir_siglip_cache, exist_ok=True)

        self.video_files = sorted(
            glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True)
        )
        if not self.video_files:
            messagebox.showinfo("Info", "Tidak ada file .mp4 ditemukan.")
            return

        self._load_data()
        self.current_index = 0
        self.load_video()
        self._lp_update_save_info()   # tampilkan lokasi simpan LP begitu dataset dibuka

    def _load_data(self):
        """Muat semua file persistensi (CSV + JSON) ke memory state aplikasi."""
        self.flagged_data      = load_flagged(self.path_csv_flagged)
        self.frame_annotations = load_frame_annotations(self.path_json_frames)
        self.batch_history     = load_batch_history(self.path_json_batch_history)
        self._enforce_mutual_exclusion_on_history(self.batch_history)
        self.skipped_videos    = load_skipped(self.path_json_skipped)
        self.manual_labels     = self._load_manual_labels()
        self._sync_manual_missing_from_ai()   # isi video/frame baru + _rejected yang hilang
        self.augment_marks     = self._load_augment_marks()
        self._load_saved_thresholds()
        self._load_rules()
        self._update_flag_count()
        self.right_panel.update_statistics(self.batch_history)
        self.right_panel.update_manual_statistics(self.manual_labels)
        # Refresh dropdown daftar batch file yang tersedia
        self.right_panel.refresh_batch_files(os.path.dirname(self.path_json_batch_history))
        # Kalibrasi per-orang: buang cache modul (anti-stale saat ganti folder) + refresh label
        # supaya tanda netral yang TERSIMPAN di disk langsung tercermin saat folder dibuka.
        try:
            from utils.person_neutral import invalidate_cache
            invalidate_cache()
            n_marked, total, _ = self._neutral_status()
            print(f"[Kalibrasi] person_neutrals.json: {os.path.join(self._dataset_dir(), 'person_neutrals.json')}")
            print(f"[Kalibrasi] {n_marked}/{total} orang sudah punya baseline netral (dimuat dari disk)")
            self._update_neutral_label()
        except Exception as e:
            print(f"[Kalibrasi] gagal muat status netral: {e}")

    def _enforce_mutual_exclusion_on_history(self, history: dict):
        """Perbaiki data batch_history lama: nol-kan prediction dan frame_preds
        untuk label yang kalah dalam pasangan eksklusif (Boredom↔Engagement, Confusion↔Frustration).
        Dipanggil saat load dari disk supaya data lama juga konsisten."""
        for rel_path, vid_data in history.items():
            pl = vid_data.get("per_label", {})
            for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                idx_a = str(LABELS.index(lbl_a))
                idx_b = str(LABELS.index(lbl_b))
                if pl.get(idx_a, {}).get("prediction", 0) == 1 and pl.get(idx_b, {}).get("prediction", 0) == 1:
                    score_a = pl[idx_a].get("avg_score", 0)
                    score_b = pl[idx_b].get("avg_score", 0)
                    loser = idx_b if score_a >= score_b else idx_a
                    pl[loser]["prediction"] = 0
                    n = len(pl[loser].get("frame_preds", []))
                    pl[loser]["frame_preds"] = [0] * n

    def _load_saved_thresholds(self):
        """Muat threshold tersimpan dan terapkan ke slider UI."""
        if not self.path_json_thresholds:
            return
        saved = load_thresholds(self.path_json_thresholds, LABELS)
        if saved:
            for i, val in enumerate(saved):
                self.right_panel.threshold_vars[i].set(val)
                if i < len(self.right_panel.threshold_entry_vars):
                    self.right_panel.threshold_entry_vars[i].set(f"{float(val):.2f}")
            print(f"[Threshold] Dimuat dari disk: {[f'{v:.2f}' for v in saved]}")

    def _save_current_thresholds(self):
        """Simpan threshold saat ini ke disk."""
        if not self.path_json_thresholds:
            return
        thrs = [v.get() for v in self.right_panel.threshold_vars]
        save_thresholds(self.path_json_thresholds, LABELS, thrs)

    def _batch_extra_path(self) -> str | None:
        """
        Jika checkbox 'Buat batch baru' dicentang, kembalikan path file batch baru.
        Jika tidak dicentang, kembalikan None (nimpa mode).
        """
        if not self.right_panel.batch_new_var.get():
            return None
        name = self.right_panel.batch_name_entry.get().strip()
        if not name:
            import datetime
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(self.path_json_batch_history)
        return os.path.join(base_dir, f"batch_history_{name}.json")

    def _siglip_cache_path(self, rel_path: str) -> str:
        """Kembalikan path file siglip cache untuk satu video."""
        if not self.path_dir_siglip_cache:
            return None
        safe = rel_path.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
        safe = os.path.splitext(safe)[0]
        return os.path.join(self.path_dir_siglip_cache, safe + ".json")

    # ── Semi-manual labeling ──────────────────────────────────────────────────

    @staticmethod
    def _manual_path_for(batch_path: str) -> str:
        """Derive path manual_labels dari path batch_history.
        Contoh: .../hasil2/batch_history_hasil2.json → .../hasil2/manual_labels_hasil2.json
                .../hasil/batch_history.json          → .../hasil/manual_labels.json
        """
        base_dir = os.path.dirname(batch_path)
        fname    = os.path.basename(batch_path)          # batch_history_xxx.json
        fname    = fname.replace("batch_history", "manual_labels", 1)
        return os.path.join(base_dir, fname)

    def _load_manual_labels(self) -> dict:
        """Muat manual_labels yang sesuai dengan batch aktif, atau return dict kosong."""
        if not self.path_json_batch_history:
            return {}
        path = self._manual_path_for(self.path_json_batch_history)
        if os.path.exists(path):
            try:
                import json as _json
                with open(path) as f:
                    data = _json.load(f)
                # Bersihkan konflik mutual exclusion dari data lama
                for rel_path, frames in data.items():
                    for f_idx, fdata in frames.items():
                        for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                            if fdata.get(lbl_a, 0) == 1 and fdata.get(lbl_b, 0) == 1:
                                fdata[lbl_b] = 0  # default: pertahankan lbl_a
                return data
            except Exception:
                pass
        return {}

    def _sync_manual_missing_from_ai(self):
        """Isi celah antara manual_labels dan frame_annotations saat load:
        - Video belum ada di manual_labels → copy penuh dari frame_annotations
        - Frame sudah ada tapi field _rejected hilang → copy dari frame_annotations
        - Entri stale (ada di manual_labels tapi tidak di frame_annotations) → hapus
        Label (0/1) dan _rejected yang sudah ada di manual_labels TIDAK ditimpa —
        perubahan user (un-reject manual) tetap dipertahankan setelah reload.
        Override penuh hanya terjadi saat recalculate/proses via _sync_manual_from_ai.
        """
        fa = self.frame_annotations
        # Hapus entri stale
        for k in [k for k in list(self.manual_labels) if k not in fa]:
            del self.manual_labels[k]
        # Isi yang belum ada
        for rel_path, fa_vid in fa.items():
            if rel_path not in self.manual_labels:
                self.manual_labels[rel_path] = {
                    str(fi): {
                        **{lbl: int(fa_vid.get(str(fi), {}).get(lbl, 0)) for lbl in LABELS},
                        "_rejected": fa_vid.get(str(fi), {}).get("_rejected", False),
                    }
                    for fi in range(2)
                }
            else:
                for fi in range(2):
                    fi_str = str(fi)
                    if fi_str not in self.manual_labels[rel_path]:
                        self.manual_labels[rel_path][fi_str] = {
                            **{lbl: int(fa_vid.get(fi_str, {}).get(lbl, 0)) for lbl in LABELS},
                            "_rejected": fa_vid.get(fi_str, {}).get("_rejected", False),
                        }
                    elif "_rejected" not in self.manual_labels[rel_path][fi_str]:
                        # Field belum ada (file lama) → isi default dari AI
                        self.manual_labels[rel_path][fi_str]["_rejected"] = (
                            fa_vid.get(fi_str, {}).get("_rejected", False)
                        )

    def _save_manual_labels(self, extra_batch_path: str | None = None):
        """Simpan manual_labels ke disk (ke file yang sesuai batch aktif).
        Jika extra_batch_path diberikan, simpan salinan ke sana juga."""
        if not self.path_json_batch_history:
            return
        import json as _json
        # Simpan ke path batch utama
        path = self._manual_path_for(self.path_json_batch_history)
        with open(path, "w") as f:
            _json.dump(self.manual_labels, f, indent=2)
        # Simpan salinan ke batch extra jika ada
        if extra_batch_path:
            path_extra = self._manual_path_for(extra_batch_path)
            with open(path_extra, "w") as f:
                _json.dump(self.manual_labels, f, indent=2)
        # Update statistik manual di UI (thread-safe: dipanggil dari main thread setelah save)
        if hasattr(self, "right_panel"):
            self.right_panel.update_manual_statistics(self.manual_labels)

    def _sync_manual_from_ai(self, rel_path: str, save: bool = True):
        """Reset manual_labels untuk satu video dari frame_annotations (label AI terbaru).
        Dipanggil setiap kali AI selesai proses atau label direset.
        save=False: hanya update dict di memori, tidak tulis ke disk (untuk bulk sync).
        """
        fa = self.frame_annotations.get(rel_path, {})
        self.manual_labels[rel_path] = {
            str(fi): {
                **{lbl: int(fa.get(str(fi), {}).get(lbl, 0)) for lbl in LABELS},
                "_rejected": fa.get(str(fi), {}).get("_rejected", False),
            }
            for fi in range(2)
        }
        if save:
            self._save_manual_labels()
        # Refresh checkbox jika mode aktif dan ini video yang sedang tampil
        if self.semi_manual_var.get() and self.video_files:
            cur_rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
            if cur_rel == rel_path:
                for fi in range(2):
                    self.left_panel.update_manual_checkboxes(
                        fi, self.manual_labels[rel_path].get(str(fi), {})
                    )

    def _on_flag_toggle(self):
        """Guard: Flag/Reject hanya bisa diubah saat mode Label Semi Manual aktif."""
        if not self.semi_manual_var.get():
            # Batalkan perubahan — kembalikan ke nilai sebelumnya
            vp = self.video_files[self.current_index] if self.video_files else None
            if vp:
                rel = os.path.relpath(vp, self.root_folder)
                self.flag_var.set(rel in self.flagged_data)
            else:
                self.flag_var.set(False)
            self.right_panel.lbl_batch_status.configure(
                text="Aktifkan 'Label Semi Manual' untuk mengedit",
                text_color="#f59e0b",
            )

    def _on_semi_manual_toggle(self):
        """Dipanggil saat toggle Label Semi Manual diubah.
        ON  → tampilkan checkbox, highlight dari manual_labels
        OFF → sembunyikan checkbox, highlight kembali dari frame_annotations (AI)
        """
        active = self.semi_manual_var.get()
        self.left_panel.show_manual_checkboxes(active)
        if active:
            self._init_manual_for_current()
        # Selalu refresh gallery agar highlight frame sesuai mode (manual / AI)
        self.refresh_frame_gallery()

    def _init_manual_for_current(self):
        """Tampilkan manual_labels untuk video aktif di checkbox.
        Jika belum ada entri, duplikat dulu dari label AI."""
        if not self.video_files:
            return
        vp  = self.video_files[self.current_index]
        rel = os.path.relpath(vp, self.root_folder)
        if rel not in self.manual_labels:
            self._sync_manual_from_ai(rel)
        else:
            for fi in range(2):
                self.left_panel.update_manual_checkboxes(
                    fi, self.manual_labels[rel].get(str(fi), {})
                )

    # ── Penanda augmentasi LivePortrait (untuk notebook 4-Create) ────────────────
    # Model: 1 (atau N) GAMBAR REFERENSI netral per orang + VIDEO driving per emosi.
    # Emosi yang digenerate ditentukan dari video driving yang ada (count>0) per orang.
    @staticmethod
    def _person_of(rel_or_path: str):
        import re as _re
        m = _re.search(r'/([0-9a-f-]{36})/', (rel_or_path or "").replace("\\", "/"))
        return m.group(1) if m else None

    def _default_augment_marks(self) -> dict:
        return {"video_root": self.root_folder,
                "result_root": os.path.dirname(getattr(self, "path_json_augment", "") or ""),
                "reference_images": [],
                "driving_images": {l: [] for l in LABELS},
                "lp_transform_frames": []}

    def _load_augment_marks(self) -> dict:
        """Muat augment_marks.json (gambar referensi per orang + video driving per emosi)."""
        import json as _json
        marks = self._default_augment_marks()
        p = getattr(self, "path_json_augment", None)
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    data = _json.load(f)
                marks["reference_images"] = list(data.get("reference_images", []))
                # Migrasi format lama (source_frames per emosi) → reference_images datar.
                if not marks["reference_images"] and "source_frames" in data:
                    seen = []
                    for l in LABELS:
                        for img in data["source_frames"].get(l, []):
                            if img not in seen:
                                seen.append(img)
                    marks["reference_images"] = seen
                for l in LABELS:
                    marks["driving_images"][l] = list(data.get("driving_images", {}).get(l, []))
                marks["lp_transform_frames"] = list(data.get("lp_transform_frames", []))
            except Exception as e:
                print(f"[Augment] gagal baca {p}: {e}")
        return marks

    def _save_augment_marks(self):
        """Simpan augment_marks.json (auto setiap tandai/lepas)."""
        import json as _json
        p = getattr(self, "path_json_augment", None)
        if not p:
            return
        self.augment_marks["video_root"]  = self.root_folder
        self.augment_marks["result_root"] = os.path.dirname(p)
        try:
            with open(p, "w") as f:
                _json.dump(self.augment_marks, f, indent=2)
        except Exception as e:
            print(f"[Augment] gagal simpan {p}: {e}")

    def _cropped_rel_for_frame(self, frame_idx: int) -> str:
        """Path gambar crop (relatif ke OUTPUT_DIR) untuk frame video aktif."""
        vp   = self.video_files[self.current_index]
        rel  = os.path.relpath(vp, self.root_folder)
        base = os.path.splitext(rel.replace("\\", "/"))[0]
        return os.path.join("cropped_faces", "clean", base, f"frame_{frame_idx:02d}.jpg").replace("\\", "/")

    def _on_reference_mark(self, frame_idx: int, value: bool):
        """Tandai/lepas 1 frame sebagai GAMBAR REFERENSI (netral, sumber augmentasi)."""
        if not self.video_files:
            return
        img_rel = self._cropped_rel_for_frame(frame_idx)
        refs = self.augment_marks.setdefault("reference_images", [])
        if value and img_rel not in refs:
            refs.append(img_rel)
        elif not value and img_rel in refs:
            refs.remove(img_rel)
        self._save_augment_marks()
        self._refresh_augment_widgets(os.path.relpath(self.video_files[self.current_index], self.root_folder))

    def _on_driving_mark(self, frame_idx: int, emotion: str):
        """Tandai GAMBAR (frame) ini sebagai driving untuk SATU emosi ('' = lepas)."""
        if not self.video_files:
            return
        img_rel = self._cropped_rel_for_frame(frame_idx)
        for l in LABELS:                                   # single-select lintas emosi
            if img_rel in self.augment_marks["driving_images"].get(l, []):
                self.augment_marks["driving_images"][l].remove(img_rel)
        if emotion in LABELS:
            self.augment_marks["driving_images"].setdefault(emotion, []).append(img_rel)
        self._save_augment_marks()
        self._refresh_augment_widgets(os.path.relpath(self.video_files[self.current_index], self.root_folder))

    def _person_stats(self, uuid: str) -> dict:
        """Statistik penanda satu orang: gambar referensi + frame LP Transform."""
        refs = [r for r in self.augment_marks.get("reference_images", []) if self._person_of(r) == uuid]
        di   = self.augment_marks.get("driving_images", {})
        drv  = {l: len([v for v in di.get(l, []) if self._person_of(v) == uuid]) for l in LABELS}
        lp_n = len([v for v in self.augment_marks.get("lp_transform_frames", [])
                    if self._person_of(v) == uuid])
        return {"refs": len(refs), "driving": drv, "lp": lp_n}

    def _refresh_augment_widgets(self, rel_path: str):
        """Sinkronkan tombol referensi + segmented driving + statistik orang aktif."""
        if not hasattr(self.left_panel, "set_reference_mark"):
            return
        rel_norm = rel_path.replace("\\", "/")
        base = os.path.splitext(rel_norm)[0]
        refs = set(self.augment_marks.get("reference_images", []))
        di   = self.augment_marks.get("driving_images", {})
        for fi in range(2):
            img_rel = os.path.join("cropped_faces", "clean", base, f"frame_{fi:02d}.jpg").replace("\\", "/")
            self.left_panel.set_reference_mark(fi, img_rel in refs)
            cur = ""
            for l in LABELS:
                if img_rel in di.get(l, []):
                    cur = l
                    break
            self.left_panel.set_driving_mark(fi, cur)
        # LP Transform marks
        lp_frames = set(self.augment_marks.get("lp_transform_frames", []))
        for fi in range(2):
            img_rel = os.path.join("cropped_faces", "clean", base, f"frame_{fi:02d}.jpg").replace("\\", "/")
            if hasattr(self.left_panel, "set_lp_mark"):
                self.left_panel.set_lp_mark(fi, img_rel in lp_frames)
        if hasattr(self.left_panel, "set_lp_total"):
            self.left_panel.set_lp_total(len(lp_frames))

        uuid = self._person_of(rel_norm)
        self.left_panel.update_augment_stats(uuid, self._person_stats(uuid) if uuid else None)
        self.left_panel.set_reference_total(len(self.augment_marks.get("reference_images", [])))
        self.left_panel.set_driving_total(
            sum(len(v) for v in self.augment_marks.get("driving_images", {}).values()))
        # Sinkronkan sumber LP ke video yang sedang dilihat (perbaiki: frame LP tak berubah)
        self._lp_sync_source_for_current()

    def _goto_reference(self, delta: int):
        """Loncat ke video+frame yang ditandai sebagai Gambar Referensi (next/prev)."""
        if not self.video_files:
            return
        import re as _re
        rel_to_idx = self._rel_to_idx()
        targets = []   # (video_index, frame_idx)
        for img in self.augment_marks.get("reference_images", []):
            m = _re.match(r"cropped_faces/clean/(.+)/frame_(\d+)\.jpg$", img.replace("\\", "/"))
            if not m:
                continue
            vid_idx = rel_to_idx.get(m.group(1) + ".mp4")
            if vid_idx is not None:
                targets.append((vid_idx, int(m.group(2))))
        targets = sorted(set(targets))
        if not targets:
            self.right_panel.lbl_batch_status.configure(
                text="Belum ada gambar referensi ditandai", text_color="#fbbf24")
            return
        cur = self.current_index
        if delta > 0:
            nxt = next((t for t in targets if t[0] > cur), targets[0])
        else:
            nxt = next((t for t in reversed(targets) if t[0] < cur), targets[-1])
        self.save_current_state()
        self.current_index = nxt[0]
        self.load_video()
        self.seek_to_frame(nxt[1])

    def _goto_driving(self, delta: int):
        """Loncat ke video+frame yang ditandai sebagai Driving (next/prev), lintas emosi."""
        if not self.video_files:
            return
        import re as _re
        rel_to_idx = self._rel_to_idx()
        targets = []
        for l in LABELS:
            for img in self.augment_marks.get("driving_images", {}).get(l, []):
                m = _re.match(r"cropped_faces/clean/(.+)/frame_(\d+)\.jpg$", img.replace("\\", "/"))
                if not m:
                    continue
                vid_idx = rel_to_idx.get(m.group(1) + ".mp4")
                if vid_idx is not None:
                    targets.append((vid_idx, int(m.group(2))))
        targets = sorted(set(targets))
        if not targets:
            self.right_panel.lbl_batch_status.configure(
                text="Belum ada gambar driving ditandai", text_color="#fbbf24")
            return
        cur = self.current_index
        if delta > 0:
            nxt = next((t for t in targets if t[0] > cur), targets[0])
        else:
            nxt = next((t for t in reversed(targets) if t[0] < cur), targets[-1])
        self.save_current_state()
        self.current_index = nxt[0]
        self.load_video()
        self.seek_to_frame(nxt[1])

    def _on_lp_mark(self, frame_idx: int, value: bool):
        """Tandai/lepas 1 frame sebagai target LP Transform."""
        if not self.video_files:
            return
        img_rel  = self._cropped_rel_for_frame(frame_idx)
        lp_list  = self.augment_marks.setdefault("lp_transform_frames", [])
        if value and img_rel not in lp_list:
            lp_list.append(img_rel)
        elif not value and img_rel in lp_list:
            lp_list.remove(img_rel)
        self._save_augment_marks()
        self._refresh_augment_widgets(
            os.path.relpath(self.video_files[self.current_index], self.root_folder)
        )

    def _rel_to_idx(self) -> dict:
        """Peta path-relatif video (pemisah '/') → index di self.video_files.
        Dipakai bersama oleh navigasi referensi, driving, dan LP."""
        return {os.path.relpath(v, self.root_folder).replace("\\", "/"): i
                for i, v in enumerate(self.video_files)}

    def _lp_targets(self):
        """Daftar (vid_idx, frame_idx) untuk semua frame bertanda LP, terurut."""
        import re as _re
        rel_to_idx = self._rel_to_idx()
        targets = []
        for img in self.augment_marks.get("lp_transform_frames", []):
            m = _re.match(r"cropped_faces/clean/(.+)/frame_(\d+)\.jpg$",
                          img.replace("\\", "/"))
            if not m:
                continue
            vid_idx = rel_to_idx.get(m.group(1) + ".mp4")
            if vid_idx is not None:
                targets.append((vid_idx, int(m.group(2))))
        return sorted(set(targets))

    def _goto_lp_mark(self, delta: int):
        """Loncat ke video+frame yang ditandai LP Transform (next/prev)."""
        if not self.video_files:
            return
        targets = self._lp_targets()
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not targets:
            if lp:
                lp.update_progress("Belum ada frame bertanda LP Transform", "#fbbf24")
            return
        cur = (self.current_index,
               self._lp_current_source[1] if self._lp_current_source else -1)
        if delta > 0:
            nxt = next((t for t in targets if t > cur), targets[0])
        else:
            nxt = next((t for t in reversed(targets) if t < cur), targets[-1])
        self._lp_current_source = nxt
        self.save_current_state()
        self.current_index = nxt[0]
        self.load_video()
        self.seek_to_frame(nxt[1])
        self._lp_set_source_preview()

    def _open_lp_panel(self):
        """Buka LP Transform panel di area galeri kiri."""
        if hasattr(self, "left_panel"):
            self.left_panel.show_lp_mode()
            # Pilih frame LP pertama bila belum ada sumber aktif
            if self._lp_current_source is None:
                t = self._lp_targets()
                if t:
                    self._lp_current_source = t[0]
            self._lp_set_source_preview()

    def _lp_set_source_preview(self):
        """Tampilkan gambar source (frame bertanda) di panel LP, dengan UUID + frame."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not lp._built or self._lp_current_source is None:
            return
        vid_idx, fi = self._lp_current_source
        if vid_idx >= len(self.video_files):
            return
        rel  = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        uuid = self._person_of(rel)
        result_dir = os.path.dirname(self.path_json_augment) if self.path_json_augment else ""
        sp = os.path.join(result_dir, "cropped_faces", "clean", base, f"frame_{fi:02d}.jpg")
        if os.path.exists(sp):
            import cv2 as _cv2
            bgr = _cv2.imread(sp)
            if bgr is not None:
                lp.set_source_frame(bgr, uuid or "", fi)
            else:
                lp.set_source_label(uuid or "", fi)
        else:
            lp.set_source_label(uuid or "", fi)
        self._lp_update_save_info()   # lokasi simpan ikut UUID frame sumber ini
        # Tampilkan lagi hasil yang sudah pernah diproses untuk frame ini (dari cache)
        self._lp_restore_cached_result((vid_idx, fi))

    def _lp_update_save_info(self):
        """Perbarui indikator 'frame disimpan ke mana' di panel LP. Dipanggil saat folder
        dataset dibuka, saat emosi/sumber berganti, dan setelah menyimpan — supaya user
        selalu tahu lokasi penyimpanan (menjawab kebingungan mekanisme simpan)."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not getattr(lp, "_built", False):
            return
        if not getattr(self, "path_json_augment", None):
            lp.set_save_info(
                "Lokasi simpan: buka folder dataset dulu (tombol 'Buka Folder' di atas).",
                "#9ca3af")
            return
        result_dir = os.path.dirname(self.path_json_augment)
        base = os.path.basename(result_dir.rstrip("/\\")) or result_dir
        emos = lp.get_selected_emotions()
        emo = emos[0] if emos else "{emosi}"
        uuid = "{uuid}"
        if self._lp_current_source and self._lp_current_source[0] < len(self.video_files):
            rel = os.path.relpath(self.video_files[self._lp_current_source[0]],
                                  self.root_folder).replace("\\", "/")
            uuid = (self._person_of(rel) or "unknown")[:8]
        lp.set_save_info(
            f"Lokasi simpan: {base}/augmented/liveportrait_app/{uuid}/{emo}/  "
            f"→ langsung muncul di 'Tinjau & Label'.", "#2563eb")

    def _lp_cache_result(self, key, emo, video_path, driving_path):
        """Catat hasil LP per frame sumber. Yang disimpan hanya PATH video output +
        driving (RAM nyaris nol) — frame di-decode ulang dari disk saat dibutuhkan."""
        self._lp_result_cache[key] = {"emo": emo, "video": video_path, "driving": driving_path}
        if key in self._lp_result_cache_order:
            self._lp_result_cache_order.remove(key)
        self._lp_result_cache_order.append(key)
        while len(self._lp_result_cache_order) > 32:
            tua = self._lp_result_cache_order.pop(0)
            self._lp_result_cache.pop(tua, None)

    # ── Cache hasil PERSISTEN (lintas-sesi). Kunci = identitas file (ukuran+mtime),
    #    BUKAN nama — file di-rename tetap dikenali; video baru pasti beda kunci. ──
    @staticmethod
    def _lp_sig(path: str) -> str:
        """Identitas isi file dari ukuran + waktu modifikasi. Rename TIDAK mengubahnya;
        file baru / di-encode ulang pasti berubah."""
        try:
            st = os.stat(path)
            return f"{st.st_size}-{int(st.st_mtime)}"
        except OSError:
            return ""

    def _lp_pcache_path(self) -> str | None:
        p = getattr(self, "path_json_augment", None)   # baru ada setelah open_folder
        return os.path.join(os.path.dirname(p), "lp_result_cache.json") if p else None

    def _lp_pcache_key(self, src: str, drv: str, emo: str) -> str:
        return f"{self._lp_sig(src)}|{self._lp_sig(drv)}|{emo}"

    def _lp_pcache_get(self, src: str, drv: str, emo: str) -> str | None:
        """Path video hasil LP yang SUDAH pernah dibuat untuk kombinasi (sumber, driving,
        emosi) yang identik — proses ulang jadi instan, walau file driving di-rename."""
        import json as _json
        p = self._lp_pcache_path()
        k = self._lp_pcache_key(src, drv, emo)
        if not p or not os.path.exists(p) or k.startswith("|") or "||" in k:
            return None
        try:
            with self._lp_json_lock, open(p) as f:
                ent = _json.load(f).get(k)
        except Exception:
            return None
        if ent and os.path.exists(ent.get("video", "")):
            return ent["video"]
        return None

    def _lp_pcache_put(self, src: str, drv: str, emo: str, out_vid: str):
        import json as _json
        p = self._lp_pcache_path()
        k = self._lp_pcache_key(src, drv, emo)
        if not p or not out_vid or k.startswith("|") or "||" in k:
            return
        with self._lp_json_lock:
            data = {}
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        data = _json.load(f)
                except Exception:
                    data = {}
            data[k] = {"video": out_vid, "driving": drv, "emo": emo, "src": src}
            try:
                with open(p, "w") as f:
                    _json.dump(data, f, indent=1)
            except Exception as e:
                print(f"[LP cache] gagal menyimpan lp_result_cache.json: {e}")

    def _lp_pcache_lookup_any_for_source(self, key):
        """Render TERAKHIR yang pernah dibuat untuk frame sumber `key` — LEPAS dari emosi/
        driving yang sedang dipilih. Supaya hasil 'Proses Frame Ini' / Batch muncul lagi
        otomatis saat frame itu dibuka kembali, bahkan setelah aplikasi ditutup-buka dan
        tanpa harus memilih ulang emosi. Dicocokkan via identitas file (sig), bukan nama."""
        import json as _json
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not getattr(lp, "_built", False) or not getattr(self, "path_json_augment", None):
            return None
        vid_idx, fi = key
        if vid_idx >= len(self.video_files):
            return None
        rel  = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        result_dir = os.path.dirname(self.path_json_augment)
        src = os.path.join(result_dir, "cropped_faces", "clean", base, f"frame_{fi:02d}.jpg")
        sig = self._lp_sig(src)
        p = self._lp_pcache_path()
        if not sig or not p or not os.path.exists(p):
            return None
        try:
            with self._lp_json_lock, open(p) as f:
                data = _json.load(f)
        except Exception:
            return None
        # dict menjaga urutan sisip → entri belakangan = paling baru
        best = None
        for kk, ent in data.items():
            if not kk.startswith(sig + "|"):
                continue
            if ent.get("video") and os.path.exists(ent["video"]):
                emo = ent.get("emo") or (kk.split("|")[2] if kk.count("|") >= 2 else "")
                best = {"emo": emo, "video": ent["video"], "driving": ent.get("driving", "")}
        return best

    def _lp_pcache_lookup_for_source(self, key):
        """Cari hasil tersimpan (cache persisten) untuk frame sumber `key`, memakai
        emosi + driving yang sedang dipilih di panel. Untuk restore lintas-sesi."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not getattr(lp, "_built", False) or not getattr(self, "path_json_augment", None):
            return None
        emos = lp.get_selected_emotions()
        if not emos:
            return None
        emo = emos[0]
        drv = lp.get_driving_choice(emo)
        vid_idx, fi = key
        if not drv or vid_idx >= len(self.video_files):
            return None
        rel  = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        result_dir = os.path.dirname(self.path_json_augment)
        src = os.path.join(result_dir, "cropped_faces", "clean", base, f"frame_{fi:02d}.jpg")
        out = self._lp_pcache_get(src, drv, emo)
        return {"emo": emo, "video": out, "driving": drv} if out else None

    def _lp_restore_cached_result(self, key):
        """Tampilkan kembali hasil LP frame ini bila pernah diproses — dari cache sesi,
        atau dari cache persisten (hasil sesi lalu / file di-rename tetap dikenali).
        Decode video di THREAD LATAR supaya pindah-pindah frame tetap mulus."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not getattr(lp, "_built", False):
            return
        data = self._lp_result_cache.get(key)
        if not data or not os.path.exists(data.get("video", "")):
            # cache sesi miss → cache persisten (emosi terpilih dulu, lalu lepas-emosi)
            data = (self._lp_pcache_lookup_for_source(key)
                    or self._lp_pcache_lookup_any_for_source(key))
        if not data or not os.path.exists(data.get("video", "")):
            return
        lp.start_loading("Memuat hasil yang sudah pernah diproses")

        def _worker(d=data, k=key):
            from ui.lp_panel import _decode_video
            res_frames = _decode_video(d["video"])
            drv = d.get("driving", "")
            drv_frames = _decode_video(drv) if drv and os.path.exists(drv) else []

            def _terapkan():
                if self._lp_current_source != k:
                    lp.stop_loading("")          # user sudah pindah frame — jangan timpa
                    return
                if drv_frames:
                    lp.set_driving_frames(drv_frames, d["emo"], path=drv)
                lp.set_result_frames(res_frames, d["emo"])
                lp.stop_loading(
                    f"Hasil sebelumnya ditampilkan ({d['emo']}, {len(res_frames)} frame).",
                    "#10b981")
            self.root.after(0, _terapkan)

        threading.Thread(target=_worker, daemon=True).start()

    def _lp_clear_all_marks(self):
        """Hapus SEMUA frame yang ditandai LP Transform (setelah konfirmasi)."""
        n = len(self.augment_marks.get("lp_transform_frames", []))
        if n == 0:
            lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
            if lp:
                lp.update_progress("Belum ada tanda LP untuk dihapus", "#fbbf24")
            return
        if not messagebox.askyesno("Hapus Semua Tanda LP",
                                   f"Hapus semua {n} tanda 'LP Transform'? (gambar hasil TIDAK terhapus)"):
            return
        self.augment_marks["lp_transform_frames"] = []
        self._lp_current_source = None
        self._save_augment_marks()
        if self.video_files:
            self._refresh_augment_widgets(
                os.path.relpath(self.video_files[self.current_index], self.root_folder))
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp:
            lp.update_progress(f"{n} tanda LP dihapus.", "#10b981")

    # ── Driving folder ───────────────────────────────────────────────────────
    _LP_EMO_ALIASES = {
        "Boredom":     ["bored", "bosan", "boring"],
        "Engagement":  ["engag", "antusias", "attentive", "fokus"],
        "Confusion":   ["confus", "confuse", "bingung", "puzzl"],
        "Frustration": ["frustrat", "frustrasi", "marah", "anger", "angry"],
    }

    def _lp_scan_driving(self, folder: str) -> dict:
        """Pindai folder driving; kelompokkan video per emosi dari nama file,
        urut berdasarkan angka di nama (boredom1, boredom2, …)."""
        import glob as _glob, re as _re
        mapping = {l: [] for l in LABELS}
        if not folder or not os.path.isdir(folder):
            return mapping
        vids = []
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4"):
            vids += _glob.glob(os.path.join(folder, ext))
        for p in sorted(set(vids)):
            name = os.path.basename(p).lower()
            for emo, keys in self._LP_EMO_ALIASES.items():
                if any(k in name for k in keys):
                    mapping[emo].append(p)
                    break

        def _key(p):
            m = _re.search(r'(\d+)', os.path.basename(p))
            return (int(m.group(1)) if m else 0, os.path.basename(p))
        for l in LABELS:
            mapping[l] = sorted(mapping[l], key=_key)
        return mapping

    # ── Dataset wajah baru (tiap foto = 1 orang baru, untuk menambah ragam orang) ──
    def _lp_scan_faces(self, folder: str) -> list:
        """Daftar path semua gambar wajah di folder (jpg/png/…), terurut."""
        import glob as _glob
        imgs = []
        if folder and os.path.isdir(folder):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                        "*.JPG", "*.JPEG", "*.PNG"):
                imgs += _glob.glob(os.path.join(folder, ext))
        return sorted(set(imgs))

    def _lp_refresh_faces(self):
        """Decode thumbnail wajah di thread latar (anti-lag), lalu render ke grid panel."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not lp._built:
            return
        paths = self._lp_scan_faces(lp.get_face_folder())
        lp.update_progress(f"Memuat {len(paths)} foto wajah…", "#6b7280")

        def _worker():
            import cv2 as _cv2
            from ui.lp_panel import _letterbox
            items = []
            for p in paths[:300]:
                bgr = _cv2.imread(p)
                if bgr is None:
                    continue
                # letterbox (rasio dijaga) — foto wajah dari luar bisa non-persegi
                items.append((p, False, _letterbox(bgr, 120, 120)))
            self.root.after(0, lambda it=items, semua=paths: (
                lp.render_faces(it, semua_path=semua),
                lp.update_progress(
                    f"{len(it)} thumbnail ditampilkan dari {len(semua)} foto — "
                    f"'Pilih Semua' memilih SEMUA foto di folder.", "#10b981")))

        threading.Thread(target=_worker, daemon=True).start()

    def _lp_prepare_face_source(self, face_path: str, result_dir: str) -> str:
        """Crop foto wajah baru dengan pipeline & RASIO SAMA seperti frame video:
        crop_face() (square + padding 0.20) lalu resize 224×224 — persis seperti
        `cropped_faces/clean/*.jpg`. Tanpa ini, foto luar diumpankan mentah ke LP
        sehingga hasilnya beda bingkai/zoom (gepeng) dengan crop dataset video.
        Hasil crop di-cache supaya tidak mengulang deteksi wajah tiap proses."""
        import cv2 as _cv2
        cache_dir = os.path.join(result_dir, "augmented", "_wajah_crop")
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, f"{self._lp_face_uuid(face_path)}.jpg")
        # cache valid bila crop sudah ada DAN tidak lebih tua dari foto sumber
        if os.path.exists(dst):
            try:
                if os.path.getmtime(dst) >= os.path.getmtime(face_path):
                    return dst
            except OSError:
                return dst
        bgr = _cv2.imread(face_path)
        if bgr is None:
            return face_path                      # fallback: pakai foto asli
        try:
            from core.face_detector import crop_face
            cropped, _found, _n, _bb = crop_face(bgr)
        except Exception as e:
            print(f"[LP Face] crop_face gagal ({face_path}): {e}")
            cropped = bgr
        if cropped is None or cropped.size == 0:
            cropped = bgr
        sq = _cv2.resize(cropped, (224, 224), interpolation=_cv2.INTER_AREA)
        _cv2.imwrite(dst, sq)
        return dst

    @staticmethod
    def _lp_face_uuid(face_path: str) -> str:
        """UUID unik per foto wajah baru (prefix 'newface-' agar bisa difilter saat merge).
        Diberi hash pendek dari path supaya 'a b.jpg' vs 'a-b.jpg' tidak dianggap orang sama."""
        import re as _re, hashlib as _hl
        stem = os.path.splitext(os.path.basename(face_path))[0]
        bersih = _re.sub(r'[^0-9a-zA-Z]+', '-', stem).strip('-').lower()[:28] or "x"
        sidik = _hl.md5(face_path.encode("utf-8")).hexdigest()[:6]
        return f"newface-{bersih}-{sidik}"

    def _lp_process_faces(self):
        """Proses foto wajah terpilih: tiap foto = ORANG BARU. Pakai emosi + driving +
        frame index (picked/N) yang sama seperti alur LP biasa."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp:
            return
        if not lp._built:
            lp.build()
        faces = lp.get_selected_faces()
        if not faces:
            lp.update_progress("Centang dulu foto wajah yang ingin diproses", "#f59e0b")
            return
        emos = lp.get_selected_emotions()
        if not emos:
            lp.update_progress("Pilih emosi (pill) dulu", "#f59e0b")
            return
        result_dir = os.path.dirname(self.path_json_augment) if self.path_json_augment else ""
        if not result_dir:
            lp.update_progress("Buka folder dataset terlebih dahulu", "#ef4444")
            return
        fracs  = lp.get_picked_fractions()
        n_even = lp.get_target_n()
        plan = []
        for face in faces:
            for emo in emos:
                for drv in lp.get_driving_list(emo):
                    plan.append((face, emo, drv))
        if not plan:
            lp.update_progress("Tidak ada video driving untuk emosi terpilih (Pindai folder driving)",
                               "#ef4444")
            return
        if not self._lp_begin(lp):
            return
        lp.start_loading(f"Proses {len(plan)} job wajah baru")

        def _worker(jobs=plan, frac=fracs, ndump=n_even):
            import cv2 as _cv2
            from ui.lp_panel import _decode_video
            done = 0
            drv_terkirim = set()
            for k, (face, emo, drv) in enumerate(jobs, 1):
                if self._lp_cancel_flag:
                    break
                uuid = self._lp_face_uuid(face)
                drv_stem = os.path.splitext(os.path.basename(drv))[0]
                out_d = os.path.join(result_dir, "augmented", "liveportrait_app",
                                     uuid, emo, drv_stem)
                stem = os.path.splitext(os.path.basename(face))[0][:14]
                self.root.after(0, lambda m=f"Wajah {k}/{len(jobs)} · {stem} · {emo}":
                                lp.start_loading(m))
                try:
                    # Samakan rasio crop dengan frame video DULU (crop_face 224 square)
                    face_src = self._lp_prepare_face_source(face, result_dir)
                    # Cache persisten: foto(crop)+driving+emosi yang sama tidak diproses ulang
                    out_vid = self._lp_pcache_get(face_src, drv, emo)
                    if out_vid is None:
                        out_vid = self._lp_infer(face_src, drv, out_d)
                        if out_vid:
                            self._lp_pcache_put(face_src, drv, emo, out_vid)
                    if not out_vid:
                        continue
                    frames = _decode_video(out_vid)
                    if not frames:
                        continue
                    total = len(frames)
                    idxs = self._lp_extract_indices(frac, ndump, total)
                    sel = [(i, frames[i]) for i in idxs if i < total]
                    self._lp_write_frames(sel, emo, "", 0, drv_stem,
                                          uuid_override=uuid, stem_override=uuid)
                    done += 1
                    # REALTIME (di-throttle anti-lag, sama seperti batch)
                    import time as _t
                    now = _t.time()
                    tampil_preview = (now - self._lp_last_preview_ts >= 1.2)
                    df = None
                    if tampil_preview:
                        self._lp_last_preview_ts = now
                        if drv not in drv_terkirim:
                            drv_terkirim.add(drv)
                            df = _decode_video(drv)
                    src_bgr = _cv2.imread(face_src) if tampil_preview else None
                    def _tampil(rf=frames, dfx=df, e2=emo, dpp=drv, sb=src_bgr, u=uuid,
                                show=tampil_preview):
                        if show:
                            if sb is not None:
                                lp.set_source_frame(sb, u, 0)
                            if dfx is not None:
                                lp.set_driving_frames(dfx, e2, path=dpp)
                            lp.set_result_frames(rf, e2)
                        self._lp_refresh_review_throttled()
                    self.root.after(0, _tampil)
                except Exception as ex:
                    print(f"[LP Face] {face} {emo} {drv_stem}: {ex}")

            def _fin(d=done, t=len(jobs)):
                self._lp_busy = False
                lp.stop_loading(f"Wajah baru selesai: {d}/{t} job. Cek di 'Tinjau & Label'.", "#10b981")
                self._lp_refresh_review()
            self.root.after(0, _fin)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Worker LP persisten (model di-load SEKALI → tak ada lag reload tiap frame) ──
    def _lp_base_flags(self) -> list:
        return os.getenv("LP_EXTRA_FLAGS", "").split()

    def _lp_pipe_readline(self, pipe, timeout: float):
        """readline dengan timeout (anti-hang), dicek per detik supaya tombol Batal
        tetap berfungsi bahkan saat menunggu model load. None bila timeout/EOF/batal."""
        import select, time as _t
        batas = _t.time() + timeout
        while _t.time() < batas:
            if self._lp_cancel_flag:
                return None
            try:
                r, _, _ = select.select([pipe], [], [], 1.0)
            except Exception:
                # select gagal (mis. pipe Windows) → blocking readline biasa
                return pipe.readline() or None
            if r:
                return pipe.readline() or None
        return None

    def _lp_ensure_worker(self) -> bool:
        """Pastikan worker LP hidup (model sudah di-load). Return True bila siap."""
        import subprocess, json as _json
        if self._lp_worker_proc and self._lp_worker_proc.poll() is None:
            return True
        lp_python = os.getenv("LIVEPORTRAIT_PYTHON", "")
        project_root = os.getenv("PROJECT_ROOT", "")
        if not lp_python or not os.path.exists(lp_python) or not project_root:
            return False
        lp_dir = os.path.join(project_root, "4-Create", "LivePortrait")
        worker = os.path.join(project_root, "4-Create", "lp_worker.py")
        if not os.path.exists(worker) or not os.path.isdir(lp_dir):
            return False
        log_path = os.path.join(project_root, "4-Create", "lp_worker.log")
        try:
            log_fh = open(log_path, "w")
        except Exception:
            log_fh = subprocess.DEVNULL
        self._lp_worker_proc = subprocess.Popen(
            [lp_python, worker] + self._lp_base_flags(),
            cwd=lp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_fh, text=True, bufsize=1,
        )
        # tunggu model load (timeout longgar 300s; kalau lewat → anggap gagal, jangan hang)
        line = self._lp_pipe_readline(self._lp_worker_proc.stdout, 300)
        if not line:
            self._lp_kill_worker()
            return False
        try:
            msg = _json.loads(line)
        except Exception:
            self._lp_kill_worker()
            return False
        if not msg.get("ready"):
            self._lp_kill_worker()
            return False
        print("[LP worker] siap (model ter-load sekali, dipakai ulang).")
        return True

    def _lp_worker_infer(self, src: str, drv: str, out: str) -> str | None:
        """Kirim satu job ke worker, kembalikan path video output. Lempar jika gagal."""
        import json as _json
        os.makedirs(out, exist_ok=True)
        p = self._lp_worker_proc
        p.stdin.write(_json.dumps({"source": src, "driving": drv, "output_dir": out}) + "\n")
        p.stdin.flush()
        line = self._lp_pipe_readline(p.stdout, 240)   # 1 job LP < ~beberapa detik; 240s anti-hang
        if not line:
            raise RuntimeError("worker LP timeout/tidak merespons")
        msg = _json.loads(line)
        if not msg.get("ok"):
            raise RuntimeError(msg.get("error", "tak diketahui"))
        return msg.get("output")

    def _lp_kill_worker(self):
        p = self._lp_worker_proc
        self._lp_worker_proc = None
        if not p:
            return
        try:
            if p.poll() is None:
                try:
                    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
                except Exception:
                    pass
                p.terminate()
        except Exception:
            pass

    def _lp_infer(self, src: str, drv: str, out: str) -> str | None:
        """Inferensi LP: pakai worker persisten (cepat); fallback subprocess sekali jalan.
        Menghormati tombol Batal: berhenti tanpa fallback bila user membatalkan."""
        with self._lp_worker_lock:
            try:
                if self._lp_ensure_worker():
                    return self._lp_worker_infer(src, drv, out)
            except Exception as e:
                print(f"[LP worker] error, fallback ke subprocess sekali jalan: {e}")
                self._lp_kill_worker()
        if self._lp_cancel_flag:
            raise RuntimeError("dibatalkan oleh user")
        return self._lp_run_subprocess(src, drv, out)

    def _lp_sync_source_for_current(self):
        """Sinkronkan sumber LP ke video yang SEDANG dilihat (perbaiki bug: frame LP
        tak berubah saat pindah video). Jika video ini punya frame bertanda LP, jadikan
        itu sumber; jika tidak, tampilkan pesan."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not getattr(lp, "_built", False) or not self.video_files:
            return
        rel  = os.path.relpath(self.video_files[self.current_index], self.root_folder).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        lp_frames = self.augment_marks.get("lp_transform_frames", [])
        marked = [fi for fi in range(2)
                  if f"cropped_faces/clean/{base}/frame_{fi:02d}.jpg" in lp_frames]
        if marked:
            cur = self._lp_current_source
            if not (cur and cur[0] == self.current_index and cur[1] in marked):
                self._lp_current_source = (self.current_index, marked[0])
            self._lp_set_source_preview()
        else:
            lp.clear_source("Video ini belum ditandai 'LP Transform' (tandai di galeri)")

    def _lp_run_subprocess(self, src_path: str, drv_path: str, out_dir: str) -> str | None:
        """Jalankan LivePortrait inference.py sekali jalan (fallback); video output (tanpa _concat)."""
        import subprocess, glob as _glob

        lp_python = os.getenv("LIVEPORTRAIT_PYTHON", "")
        if not lp_python or not os.path.exists(lp_python):
            raise RuntimeError(f"LIVEPORTRAIT_PYTHON tidak ditemukan: '{lp_python}'")
        project_root = os.getenv("PROJECT_ROOT", "")
        if not project_root:
            raise RuntimeError("PROJECT_ROOT belum diset di .env")
        lp_dir = os.path.join(project_root, "4-Create", "LivePortrait")
        lp_inference = os.path.join(lp_dir, "inference.py")
        if not os.path.exists(lp_inference):
            raise RuntimeError(f"inference.py tidak ditemukan: {lp_inference}")

        lp_flags = os.getenv("LP_EXTRA_FLAGS", "").split()
        os.makedirs(out_dir, exist_ok=True)
        cmd = [lp_python, lp_inference,
               "--source", src_path, "--driving", drv_path,
               "--output-dir", out_dir] + lp_flags
        result = subprocess.run(cmd, cwd=lp_dir, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"LP inference gagal (rc={result.returncode}):\n{result.stderr[-800:]}")
        vids = sorted([f for f in _glob.glob(os.path.join(out_dir, "*.mp4"))
                       if "_concat" not in os.path.basename(f)], key=os.path.getmtime)
        return vids[-1] if vids else None

    def _lp_resolve_source(self, lp):
        """Pastikan _lp_current_source terisi; return (vid_idx, fi) atau None."""
        if self._lp_current_source is not None:
            return self._lp_current_source
        t = self._lp_targets()
        if not t:
            lp.update_progress(
                "Tandai frame dengan tombol 'LP Transform' di galeri dulu", "#f59e0b")
            return None
        self._lp_current_source = t[0]
        return self._lp_current_source

    @staticmethod
    def _lp_extract_indices(fractions: list, n_even: int, total: int) -> list:
        """Index frame yang diekstrak dari satu video hasil berlength `total`.
        Pakai `fractions` (posisi proporsional 0-1, jumlah tetap) bila ada — sehingga
        ganti driving = jumlah frame sama, posisi menyesuaikan. Selain itu: N merata."""
        if total <= 0:
            return []
        if fractions:
            return sorted({min(total - 1, max(0, int(round(f * (total - 1))))) for f in fractions})
        return sorted({int(round(i * (total - 1) / max(n_even - 1, 1))) for i in range(n_even)})

    def _lp_begin(self, lp) -> bool:
        """Guard anti tumpang-tindih: cegah dua proses LP jalan bersamaan (anti-lag/pile-up)."""
        if self._lp_busy:
            lp.update_progress("Masih ada proses berjalan — tunggu selesai atau klik Batal", "#f59e0b")
            return False
        self._lp_busy = True
        self._lp_cancel_flag = False
        self._lp_last_review_ts = self._lp_last_preview_ts = 0.0   # job pertama tampil langsung
        return True

    def _lp_process_current(self):
        """Proses 1 frame sumber aktif → satu video hasil untuk dipratinjau & dipilih frame-nya."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp:
            return
        if not lp._built:
            lp.build()
        emos = lp.get_selected_emotions()
        if not emos:
            lp.update_progress("Pilih emosi (pill) dulu", "#f59e0b")
            return
        src = self._lp_resolve_source(lp)
        if src is None:
            return
        vid_idx, fi = src
        if vid_idx >= len(self.video_files):
            lp.update_progress("Indeks video tidak valid", "#ef4444")
            return
        rel  = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
        base = os.path.splitext(rel)[0]
        result_dir = os.path.dirname(self.path_json_augment) if self.path_json_augment else ""
        if not result_dir:
            lp.update_progress("Buka folder dataset terlebih dahulu", "#ef4444")
            return
        src_path = os.path.join(result_dir, "cropped_faces", "clean", base, f"frame_{fi:02d}.jpg")
        if not os.path.exists(src_path):
            lp.update_progress(f"Gambar crop tidak ditemukan: {src_path}", "#ef4444")
            return
        emo = emos[0]
        drv_list = lp.get_driving_list(emo)
        if not drv_list:
            lp.update_progress(
                f"Tidak ada video driving '{emo}' di folder — taruh {emo.lower()}1.mp4 lalu Pindai",
                "#ef4444")
            return
        drv_path = drv_list[0]
        uuid = self._person_of(rel)
        out_dir = os.path.join(result_dir, "augmented", "liveportrait_app",
                               uuid or "unknown", emo,
                               os.path.splitext(os.path.basename(drv_path))[0])
        self._lp_set_source_preview()
        if not self._lp_begin(lp):
            return
        lp.start_loading(f"Memproses {emo} (driving {os.path.basename(drv_path)}) — "
                         f"model di-load sekali, mohon tunggu")

        def _worker(sp=src_path, dp=drv_path, od=out_dir, e=emo, src=(vid_idx, fi)):
            from ui.lp_panel import _decode_video
            try:
                # Cache persisten: kombinasi (sumber, driving, emosi) identik → hasil lama
                # langsung dipakai, tanpa menjalankan LP lagi (dikenali dari isi file,
                # bukan nama — di-rename pun tetap kena).
                out_vid = self._lp_pcache_get(sp, dp, e)
                dari_cache = out_vid is not None
                if not dari_cache:
                    out_vid = self._lp_infer(sp, dp, od)
                if not out_vid:
                    self.root.after(0, lambda: lp.stop_loading(
                        "LP tidak menghasilkan video output", "#ef4444"))
                    return
                if not dari_cache:
                    self._lp_pcache_put(sp, dp, e, out_vid)
                res_frames = _decode_video(out_vid)        # full-res (untuk disimpan)
                drv_frames = _decode_video(dp)             # preview driving
                def on_done(rf=res_frames, df=drv_frames, emo2=e, key=src, ov=out_vid,
                            dpp=dp, cache=dari_cache):
                    self._lp_cache_result(key, emo2, ov, dpp)   # simpan PATH saja (hemat RAM)
                    lp.set_driving_frames(df, emo2, path=dpp)
                    lp.set_result_frames(rf, emo2)
                    asal = " (hasil lama dipakai — sumber & driving sama)" if cache else ""
                    lp.stop_loading(
                        f"Selesai · {emo2} · {len(rf)} frame{asal}. Tandai frame lalu Simpan.",
                        "#10b981")
                self.root.after(0, on_done)
            except Exception as ex:
                err = str(ex)[:160]
                self.root.after(0, lambda: lp.stop_loading(f"Error: {err}", "#ef4444"))
            finally:
                self._lp_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _lp_process_batch(self):
        """Batch: SATU video driving (dari dropdown) → proses SEMUA frame bertanda LP →
        ekstrak frame index yang DITANDAI di scrubber (ekspresi sama karena driving sama).
        Ganti driving → batch lagi."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp:
            return
        if not lp._built:
            lp.build()
        emos = lp.get_selected_emotions()
        if not emos:
            lp.update_progress("Pilih emosi (pill) dulu", "#f59e0b")
            return
        targets = self._lp_targets()
        if not targets:
            lp.update_progress("Belum ada frame bertanda LP Transform di galeri", "#f59e0b")
            return
        result_dir = os.path.dirname(self.path_json_augment) if self.path_json_augment else ""
        if not result_dir:
            lp.update_progress("Buka folder dataset terlebih dahulu", "#ef4444")
            return

        # Fraksi posisi (0-1) sebagai template: tiap video driving dipetakan ke index-nya
        # sendiri → jumlah frame sama, posisi proporsional walau panjang driving beda.
        fracs  = lp.get_picked_fractions()
        n_even = lp.get_target_n()                 # fallback bila belum ada pilihan
        # Rencana job: (vid_idx, fi, emo, drv). Honor dropdown (1 video bila spesifik).
        plan = []
        for vid_idx, fi in targets:
            for emo in emos:
                for drv in lp.get_driving_list(emo):
                    plan.append((vid_idx, fi, emo, drv))
        if not plan:
            lp.update_progress("Tidak ada video driving untuk emosi terpilih (Pindai folder dulu)", "#ef4444")
            return
        mode = (f"{len(fracs)} frame (posisi proporsional)" if fracs else f"{n_even} frame merata")
        if not self._lp_begin(lp):
            return
        lp.start_loading(f"Batch {len(plan)} job · ekstrak {mode}")

        def _worker(jobs=plan, frac=fracs, ndump=n_even):
            import cv2 as _cv2
            from ui.lp_panel import _decode_video
            done = 0
            drv_terkirim = set()    # driving yang preview-nya sudah dikirim ke panel (decode 1x)
            for k, (vid_idx, fi, emo, drv) in enumerate(jobs, 1):
                if self._lp_cancel_flag:
                    break
                rel  = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
                base = os.path.splitext(rel)[0]
                uuid = self._person_of(rel)
                sp   = os.path.join(result_dir, "cropped_faces", "clean", base, f"frame_{fi:02d}.jpg")
                if not os.path.exists(sp):
                    continue
                drv_stem = os.path.splitext(os.path.basename(drv))[0]
                out_d = os.path.join(result_dir, "augmented", "liveportrait_app",
                                     uuid or "unknown", emo, drv_stem)
                self.root.after(0, lambda m=f"Batch {k}/{len(jobs)} · {(uuid or '?')[:6]} · {emo}":
                                lp.start_loading(m))
                try:
                    # Cache persisten: job yang persis sama (sumber+driving+emosi, dikenali
                    # dari isi file bukan nama) tidak diproses ulang — batch ulang jadi cepat.
                    out_vid = self._lp_pcache_get(sp, drv, emo)
                    if out_vid is None:
                        out_vid = self._lp_infer(sp, drv, out_d)
                        if out_vid:
                            self._lp_pcache_put(sp, drv, emo, out_vid)
                    if not out_vid:
                        continue
                    frames = _decode_video(out_vid)
                    if not frames:
                        continue
                    total = len(frames)
                    idxs = self._lp_extract_indices(frac, ndump, total)
                    sel = [(i, frames[i]) for i in idxs if i < total]
                    self._lp_write_frames(sel, emo, rel, fi, drv_stem)
                    done += 1
                    # REALTIME (di-throttle anti-lag): tampilkan job terbaru di pratinjau &
                    # isi grid tinjau, TAPI dibatasi frekuensinya — preview ~1.2s, grid ~3s.
                    # Tanpa throttle, batch (apalagi dari cache instan) membanjiri UI → lag.
                    import time as _t
                    now = _t.time()
                    tampil_preview = (now - self._lp_last_preview_ts >= 1.2)
                    df = None
                    if tampil_preview:
                        self._lp_last_preview_ts = now
                        if drv not in drv_terkirim:
                            drv_terkirim.add(drv)
                            df = _decode_video(drv)
                    src_bgr = _cv2.imread(sp) if tampil_preview else None
                    def _tampil(rf=frames, dfx=df, e2=emo, dpp=drv, sb=src_bgr,
                                u=(uuid or ""), f2=fi, kk=(vid_idx, fi), ov=out_vid,
                                show=tampil_preview):
                        self._lp_cache_result(kk, e2, ov, dpp)
                        if show:
                            if sb is not None:
                                lp.set_source_frame(sb, u, f2)
                            if dfx is not None:
                                lp.set_driving_frames(dfx, e2, path=dpp)
                            lp.set_result_frames(rf, e2)
                        self._lp_refresh_review_throttled()
                    self.root.after(0, _tampil)
                except Exception as ex:
                    print(f"[LP Batch] {rel} f{fi} {emo} {drv_stem}: {ex}")

            def _fin(d=done, t=len(jobs)):
                self._lp_busy = False
                lp.stop_loading(f"Batch selesai: {d}/{t} job. Cek di 'Tinjau & Label'.", "#10b981")
                self._lp_refresh_review()
            self.root.after(0, _fin)

        threading.Thread(target=_worker, daemon=True).start()

    def _lp_cancel(self):
        """Minta proses LP berhenti. PENTING: _lp_busy TIDAK dibersihkan di sini —
        worker yang sedang jalan yang membersihkannya saat benar-benar berhenti,
        supaya tidak ada dua proses tumpang-tindih setelah klik Batal."""
        self._lp_cancel_flag = True
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp:
            if self._lp_busy:
                lp.start_loading("Dibatalkan — menyelesaikan job yang sedang berjalan")
            else:
                lp.stop_loading("Tidak ada proses yang berjalan.", "#9ca3af")

    def _lp_write_frames(self, frames: list, emo: str, rel: str, fi: int, drv_stem: str = "",
                         uuid_override: str = "", stem_override: str = "") -> int:
        """Tulis daftar (idx, bgr) ke augmented/liveportrait_app/{uuid}/{emo}/ +
        catat label default (emo target=1) di lp_labels.json untuk dicek sebelum merge.

        uuid_override/stem_override dipakai untuk sumber 'dataset wajah baru' (tiap foto =
        orang baru, tidak punya rel video)."""
        import cv2 as _cv2
        if not self.path_json_augment:
            return 0
        uuid = uuid_override or self._person_of(rel)
        if stem_override:
            stem = stem_override
        else:
            stem = os.path.splitext(rel)[0].replace("/", "__") + f"_f{fi:02d}"
        result_dir = os.path.dirname(self.path_json_augment)
        gen_dir = self._lp_generated_dir() or ""
        out_dir = os.path.join(result_dir, "augmented", "liveportrait_app",
                               uuid or "unknown", emo)
        os.makedirs(out_dir, exist_ok=True)
        tag = ("_" + drv_stem) if drv_stem else ""
        # Tulis gambar dulu (di luar lock — IO berat), baru update file label sekaligus.
        rel_baru = []
        for idx, fr in frames:
            fp = os.path.join(out_dir, f"{stem}{tag}_lp{idx:03d}.jpg")
            _cv2.imwrite(fp, fr)
            rel_baru.append(os.path.relpath(fp, gen_dir).replace("\\", "/") if gen_dir else fp)
        with self._lp_json_lock:
            labels = self._lp_load_labels()
            for rel_gen in rel_baru:
                labels[rel_gen] = self._lp_default_label(emo)
            self._lp_save_labels(labels)
        return len(rel_baru)

    def _lp_save_frames(self, frames: list, emo: str):
        """Simpan frame hasil (interaktif) memakai sumber aktif. Nama file diberi tag
        video DRIVING-nya — simpan dari driving berbeda TIDAK saling menimpa, sedangkan
        simpan ulang dari driving yang sama menimpa file lama (otomatis bebas duplikat)."""
        src = self._lp_current_source
        if not src or not self.path_json_augment:
            return
        vid_idx, fi = src
        rel = os.path.relpath(self.video_files[vid_idx], self.root_folder).replace("\\", "/")
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        drv_stem = ""
        if lp and getattr(lp, "driving_aktif", ""):
            drv_stem = os.path.splitext(os.path.basename(lp.driving_aktif))[0]
        saved = self._lp_write_frames(frames, emo, rel, fi, drv_stem)
        if lp:
            uuid = (self._person_of(rel) or "unknown")[:8]
            base = os.path.basename(os.path.dirname(self.path_json_augment).rstrip("/\\"))
            lp.update_progress(
                f"Tersimpan {saved} frame ({emo}) → {base}/augmented/liveportrait_app/"
                f"{uuid}/{emo}/  ·  lihat di 'Tinjau & Label' di bawah.", "#10b981")
            self._lp_update_save_info()
            self._lp_refresh_review()

    # ── Tinjau hasil generate (terima/tolak) ─────────────────────────────────
    def _lp_review_path(self) -> str | None:
        if not self.path_json_augment:
            return None
        return os.path.join(os.path.dirname(self.path_json_augment), "lp_review.json")

    def _lp_load_review(self) -> set:
        import json as _json
        p = self._lp_review_path()
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    return set(_json.load(f).get("rejected", []))
            except Exception:
                pass
        return set()

    def _lp_save_review(self, rejected: set):
        import json as _json
        p = self._lp_review_path()
        if not p:
            return
        try:
            with open(p, "w") as f:
                _json.dump({"rejected": sorted(rejected)}, f, indent=2)
        except Exception as e:
            self._lp_report_error(f"GAGAL menyimpan status tolak/terima: {e}")

    def _lp_generated_dir(self) -> str | None:
        if not self.path_json_augment:
            return None
        return os.path.join(os.path.dirname(self.path_json_augment),
                            "augmented", "liveportrait_app")

    def _lp_list_generated(self) -> list:
        """List (abs_path, emo) semua jpg hasil generate, terurut.
        Folder _trash/ (gambar yang dibuang) tidak ikut."""
        import glob as _glob
        d = self._lp_generated_dir()
        if not d or not os.path.isdir(d):
            return []
        out = []
        for p in sorted(_glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)):
            parts = p.replace("\\", "/").split("/")
            if "_trash" in parts:
                continue
            emo = next((x for x in parts if x in LABELS), "?")
            out.append((p, emo))
        return out

    def _lp_refresh_review_throttled(self, jeda: float = 3.0):
        """Refresh grid tinjau TAPI maksimal sekali per `jeda` detik. Dipakai saat batch
        supaya banjir job (apalagi dari cache yang instan) tidak memicu ratusan rebuild
        grid + glob rekursif → ini biang lag. Refresh penuh tetap dijalankan di akhir batch."""
        import time as _t
        now = _t.time()
        if now - self._lp_last_review_ts >= jeda:
            self._lp_last_review_ts = now
            self._lp_refresh_review()

    def _lp_refresh_review(self):
        """Kirim SELURUH daftar hasil + state ke panel (tanpa decode thumbnail di muka →
        cepat walau ribuan; thumbnail di-decode per-halaman oleh panel saat dibutuhkan)."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp or not lp._built or not getattr(self, "path_json_augment", None):
            return
        gen_dir = self._lp_generated_dir() or ""
        gen = self._lp_list_generated()
        with self._lp_json_lock:
            rejected  = self._lp_load_review()
            labels    = self._lp_load_labels()
            ai_labels = self._lp_load_ai_labels()
        items = [(path, emo, os.path.relpath(path, gen_dir).replace("\\", "/") if gen_dir else path)
                 for path, emo in gen]
        dibuang = self._lp_list_trashed()
        lp.set_review_data(items, labels, ai_labels, rejected, dibuang)
        ket = f"{len(items)} hasil siap ditinjau — pakai tombol panah ◀/▶ keyboard atau Loncat."
        if dibuang:
            ket += f"  ·  {len(dibuang)} di _trash (lihat filter 'Dibuang (_trash)')."
        lp.update_progress(ket, "#10b981" if items else "#6b7280")

    def _lp_toggle_reject(self, path: str):
        """Tolak/terima satu gambar hasil. Aman lintas-thread (lock). Update panel di tempat
        (tanpa reload seluruh daftar) supaya posisi navigasi tidak hilang.
        Untuk gambar yang sudah DIBUANG ke _trash: dobel-klik = pulihkan."""
        gen_dir = self._lp_generated_dir() or ""
        rel = os.path.relpath(path, gen_dir).replace("\\", "/")
        if rel.split("/")[0] == "_trash":
            self._lp_restore_one(path)
            return
        with self._lp_json_lock:
            rejected = self._lp_load_review()
            baru = rel not in rejected
            if baru:
                rejected.add(rel)
            else:
                rejected.discard(rel)
            self._lp_save_review(rejected)
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp and getattr(lp, "_built", False):
            lp.terapkan_state(rel, ditolak=baru)

    # ── Label hasil LP (untuk QA sebelum merge) ──────────────────────────────
    def _lp_labels_path(self) -> str | None:
        if not self.path_json_augment:
            return None
        return os.path.join(os.path.dirname(self.path_json_augment), "lp_labels.json")

    def _lp_load_labels(self) -> dict:
        import json as _json
        p = self._lp_labels_path()
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    return _json.load(f)
            except Exception:
                pass
        return {}

    def _lp_report_error(self, pesan: str):
        """Tampilkan error ke label status panel LP — jangan diam-diam (hanya print)."""
        print(f"[LP] {pesan}")
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp and getattr(lp, "_built", False):
            self.root.after(0, lambda: lp.update_progress(pesan[:160], "#ef4444"))

    def _lp_save_labels(self, labels: dict):
        import json as _json
        p = self._lp_labels_path()
        if not p:
            return
        try:
            with open(p, "w") as f:
                _json.dump(labels, f, indent=2)
        except Exception as e:
            self._lp_report_error(f"GAGAL menyimpan label: {e}")

    @staticmethod
    def _lp_default_label(emo: str) -> dict:
        return {l: (1 if l == emo else 0) for l in LABELS}

    def _lp_ai_labels_path(self) -> str | None:
        """File terpisah untuk hasil DETEKSI AI (dibandingkan dgn label manual final)."""
        if not self.path_json_augment:
            return None
        return os.path.join(os.path.dirname(self.path_json_augment), "lp_ai_labels.json")

    def _lp_load_ai_labels(self) -> dict:
        import json as _json
        p = self._lp_ai_labels_path()
        if p and os.path.exists(p):
            try:
                with open(p) as f:
                    return _json.load(f)
            except Exception:
                pass
        return {}

    def _lp_save_ai_labels(self, labels: dict):
        import json as _json
        p = self._lp_ai_labels_path()
        if not p:
            return
        try:
            with open(p, "w") as f:
                _json.dump(labels, f, indent=2)
        except Exception as e:
            self._lp_report_error(f"GAGAL menyimpan hasil deteksi AI: {e}")

    def _lp_set_label(self, gen_rel: str, label: str, value: int):
        """Toggle satu label manual untuk satu gambar hasil (dengan mutual exclusion).
        Update panel di tempat (tanpa reload daftar) agar posisi navigasi tetap."""
        with self._lp_json_lock:
            labels = self._lp_load_labels()
            cur = labels.get(gen_rel) or {l: 0 for l in LABELS}
            cur[label] = int(value)
            if int(value) == 1 and label in MUTUAL_EXCLUSIVE:
                cur[MUTUAL_EXCLUSIVE[label]] = 0
            labels[gen_rel] = cur
            self._lp_save_labels(labels)
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp and getattr(lp, "_built", False):
            lp.terapkan_state(gen_rel, label=dict(cur))

    def _lp_label_all_ai(self):
        """Deteksi emosi SEMUA gambar hasil pakai SigLIP+MediaPipe (penggaris yang sama
        dengan menu utama), di thread latar. Hasil DISIMPAN TERPISAH (lp_ai_labels.json)
        sebagai pembanding label manual final — user bisa cek bila ada emosi tak diinginkan."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if not lp:
            return
        if self._lp_busy:
            lp.update_progress("Masih ada proses berjalan — tunggu selesai", "#f59e0b")
            return
        gen = self._lp_list_generated()
        if not gen:
            lp.update_progress("Belum ada gambar hasil untuk dilabel", "#f59e0b")
            return
        gen_dir = self._lp_generated_dir() or ""
        # INKREMENTAL: hanya label gambar yang BELUM punya hasil AI → run ulang cepat.
        sudah = self._lp_load_ai_labels()
        antrian = [(p, e) for p, e in gen
                   if os.path.relpath(p, gen_dir).replace("\\", "/") not in sudah]
        dilewati = len(gen) - len(antrian)
        if not antrian:
            lp.update_progress(
                f"Semua {len(gen)} hasil sudah berlabel AI. Hapus lp_ai_labels.json untuk ulang.",
                "#10b981")
            self._lp_refresh_review()
            return
        prompts, ths = self.right_panel.get_prompts_and_thresholds()
        self._lp_busy = True
        lp.start_loading(f"Deteksi AI {len(antrian)} hasil baru"
                         + (f" ({dilewati} sudah berlabel, dilewati)" if dilewati else ""))

        def _worker():
            import cv2 as _cv2
            from concurrent.futures import ThreadPoolExecutor
            from PIL import Image as _Image
            from core.inference import run_siglip_on_frames
            from core.landmark_analyzer import analyze_frame
            ai_labels = self._lp_load_ai_labels()
            done = 0
            BATCH = 16   # batch GPU lebih besar = lebih sedikit forward-pass = lebih cepat

            def _baca(item):
                # imread melepas GIL → aman & efektif diparalelkan (IO-bound)
                return item[0], _cv2.imread(item[0])

            try:
                pembaca = ThreadPoolExecutor(max_workers=4)
                for s in range(0, len(antrian), BATCH):
                    if self._lp_cancel_flag:
                        break
                    chunk = antrian[s:s + BATCH]
                    terbaca = list(pembaca.map(_baca, chunk))
                    pil_imgs, lms, rels = [], [], []
                    for path, bgr in terbaca:
                        if bgr is None:
                            continue
                        rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                        pil_imgs.append(_Image.fromarray(rgb))
                        lms.append(analyze_frame(bgr))   # MediaPipe: sengaja sekuensial (1 instans)
                        rels.append(os.path.relpath(path, gen_dir).replace("\\", "/"))
                    if not pil_imgs:
                        continue
                    res = run_siglip_on_frames(pil_imgs, prompts, ths,
                                               landmark_results=lms, cfg=self.rules)
                    per_label = res.get("per_label", {})
                    for fidx, rel in enumerate(rels):
                        lab = {}
                        for li, l in enumerate(LABELS):
                            preds = per_label.get(li, {}).get("frame_preds", [])
                            lab[l] = int(preds[fidx]) if fidx < len(preds) else 0
                        for a, b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                            if lab[a] == 1 and lab[b] == 1:
                                lab[b] = 0
                        ai_labels[rel] = lab
                        done += 1
                    # Simpan TIAP batch (bukan hanya di akhir): bila dibatalkan/crash di
                    # tengah, progres tidak hilang — run berikutnya lanjut (inkremental).
                    with self._lp_json_lock:
                        self._lp_save_ai_labels(ai_labels)
                    self.root.after(0, lambda d=done, t=len(antrian):
                                    lp.start_loading(f"Deteksi AI {d}/{t}"))
                pembaca.shutdown(wait=False)
                def _fin(d=done):
                    self._lp_busy = False
                    lp.stop_loading(f"Deteksi AI selesai: {d} gambar. Bandingkan dengan label "
                                    f"manual di pemeriksa, lalu Merge.", "#10b981")
                    self._lp_refresh_review()
                self.root.after(0, _fin)
            except Exception as ex:
                err = str(ex)[:160]
                self._lp_busy = False
                self.root.after(0, lambda: lp.stop_loading(f"Error deteksi AI: {err}", "#ef4444"))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Merge hasil LP ke dataset Label2d (non-destruktif + bisa undo) ────────
    def _lp_label2d_base(self) -> str:
        return os.path.join(os.path.dirname(self.path_json_frames), "Label2d")

    def _lp_merge_dir(self, mode: str = "lp") -> str:
        """Folder output per komposisi: base / lp / lp_new (disimpan terpisah)."""
        return os.path.join(os.path.dirname(self.path_json_frames), f"Label2d_merged_{mode}")

    @staticmethod
    def _lp_path_is_newface(path: str) -> bool:
        """True bila gambar hasil berasal dari 'dataset wajah baru' (uuid prefix newface-)."""
        parts = path.replace("\\", "/").split("/")
        if "liveportrait_app" in parts:
            i = parts.index("liveportrait_app")
            if i + 1 < len(parts):
                return parts[i + 1].startswith("newface-")
        return False

    _LP_NAMA_MODE = {"base": "Tanpa LP", "lp": "Dengan LP",
                     "lp_new": "Dengan LP + Dataset Baru"}

    def _lp_merge_into_label2d(self):
        """(UI) Buat dataset gabungan sesuai komposisi radio. Validasi cepat di sini,
        kerja berat (glob ribuan file + tulis CSV) di THREAD LATAR supaya UI tak beku."""
        if not self.path_json_frames or not self.path_json_augment:
            messagebox.showinfo("Buat Dataset", "Buka folder dataset terlebih dahulu.")
            return
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        mode = lp.get_merge_mode() if lp else "lp"
        # Split dasar dibuat di UI thread (sekali, kasus pertama saja) karena
        # _split_dataset_2d membaca widget tkinter — tidak aman dari thread lain.
        base = self._lp_label2d_base()
        if not os.path.exists(os.path.join(base, "train.csv")):
            self._split_dataset_2d(silent=True, force_default=True)
        if lp:
            lp.start_loading(f"Menyusun dataset '{self._LP_NAMA_MODE[mode]}'")

        def _worker(m=mode):
            try:
                hasil = self._lp_build_merged_dataset(m)
            except ValueError as ex:           # kondisi user (bukan bug) → info biasa
                pesan = str(ex)
                self.root.after(0, lambda p=pesan: (
                    lp.stop_loading(p, "#f59e0b") if lp else None,
                    messagebox.showinfo("Buat Dataset", p)))
                return
            except Exception as ex:
                pesan = f"Gagal membuat dataset: {str(ex)[:200]}"
                self.root.after(0, lambda p=pesan: (
                    lp.stop_loading(p, "#ef4444") if lp else None,
                    messagebox.showerror("Buat Dataset", p)))
                return

            def _selesai(h=hasil):
                if lp:
                    lp.stop_loading(
                        f"Dataset '{h['nama_mode']}' dibuat: {os.path.basename(h['out'])}",
                        "#10b981")
                messagebox.showinfo("Dataset Dibuat", h["ringkasan"])
            self.root.after(0, _selesai)

        threading.Thread(target=_worker, daemon=True).start()

    def _lp_build_merged_dataset(self, mode: str) -> dict:
        """(Logika murni, TANPA UI — bisa diuji terpisah) Susun Label2d_merged_{mode}:
          - base   : dataset asli saja
          - lp     : asli + hasil LP dari frame video dataset
          - lp_new : asli + hasil LP video + hasil LP dataset wajah baru
        Non-destruktif: val/test disalin, kolom 'synthetic' ditambah, Label2d asli utuh.
        Raise ValueError dengan pesan ramah bila prasyarat tak terpenuhi."""
        import csv as _csv, json as _json
        base = self._lp_label2d_base()
        if not os.path.exists(os.path.join(base, "train.csv")):
            raise ValueError("Label2d dasar belum ada. Buat split 2D dulu (panel kanan).")

        aug_rows = []
        if mode in ("lp", "lp_new"):
            with self._lp_json_lock:
                rejected = self._lp_load_review()
                labels   = self._lp_load_labels()
            gen_dir    = self._lp_generated_dir() or ""
            result_dir = os.path.dirname(self.path_json_augment)
            for path, emo in self._lp_list_generated():
                if mode == "lp" and self._lp_path_is_newface(path):
                    continue   # mode 'lp' tidak ikutkan dataset wajah baru
                rel_gen = os.path.relpath(path, gen_dir).replace("\\", "/")
                if rel_gen in rejected:
                    continue
                lab = labels.get(rel_gen) or self._lp_default_label(emo)
                if sum(lab.get(l, 0) for l in LABELS) == 0:
                    continue   # tak ada label positif → lewati
                rel_ds = os.path.relpath(path, result_dir).replace("\\", "/")
                aug_rows.append((rel_ds, lab))
            if not aug_rows:
                raise ValueError(
                    "Tidak ada gambar hasil (diterima + berlabel) untuk komposisi ini. "
                    "Proses LP / dataset wajah dulu, atau pilih komposisi 'Tanpa LP'.")

        out = self._lp_merge_dir(mode)
        os.makedirs(out, exist_ok=True)

        def _salin_dengan_kolom_synthetic(split):
            src = os.path.join(base, f"{split}.csv")
            if not os.path.exists(src):
                return 0
            with open(src, newline="") as f:
                rows = list(_csv.reader(f))
            with open(os.path.join(out, f"{split}.csv"), "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(rows[0] + ["synthetic"])
                for r in rows[1:]:
                    w.writerow(r + ["0"])
            return len(rows) - 1

        n_val  = _salin_dengan_kolom_synthetic("val")
        n_test = _salin_dengan_kolom_synthetic("test")
        with open(os.path.join(base, "train.csv"), newline="") as f:
            rows = list(_csv.reader(f))
        with open(os.path.join(out, "train.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(rows[0] + ["synthetic"])
            for r in rows[1:]:
                w.writerow(r + ["0"])
            for rel_ds, lab in aug_rows:
                w.writerow([rel_ds] + [lab[l] for l in LABELS] + ["1"])
        n_train_base = len(rows) - 1

        with open(os.path.join(out, "lp_merge_manifest.json"), "w") as f:
            _json.dump({"mode": mode, "base": base,
                        "added_train": [r for r, _ in aug_rows], "n_aug": len(aug_rows)},
                       f, indent=2)

        nama_mode = self._LP_NAMA_MODE[mode]
        ringkasan = (
            f"Komposisi: {nama_mode}\nOutput: {out}\n\n"
            f"train: {n_train_base} asli + {len(aug_rows)} sintetik = {n_train_base + len(aug_rows)}\n"
            f"val: {n_val}   test: {n_test}\n\n"
            f"Kolom 'synthetic' = 1 untuk hasil LP. Klik 'Undo' untuk hapus folder ini.")
        print(f"[LP Merge] mode={mode} {len(aug_rows)} sintetik → {out}")
        return {"out": out, "nama_mode": nama_mode, "ringkasan": ringkasan}

    def _lp_undo_merge(self):
        """Batalkan/ hapus folder dataset gabungan (Label2d_merged_*). Label2d asli aman."""
        import glob as _glob, shutil
        cari = os.path.join(os.path.dirname(self.path_json_frames), "Label2d_merged_*")
        folders = sorted(_glob.glob(cari))
        if not folders:
            messagebox.showinfo("Undo", "Belum ada folder dataset gabungan (Label2d_merged_*).")
            return
        daftar = "\n".join("  • " + os.path.basename(f) for f in folders)
        if not messagebox.askyesno("Undo / Hapus Dataset Gabungan",
                f"Hapus folder dataset gabungan berikut?\n\n{daftar}\n\n(Label2d asli TIDAK terhapus.)"):
            return
        for f in folders:
            shutil.rmtree(f, ignore_errors=True)
        messagebox.showinfo("Undo", f"{len(folders)} folder dataset gabungan dihapus.")
        print(f"[LP Merge] undo: {len(folders)} folder Label2d_merged_* dihapus.")

    def _lp_show_stats(self):
        """Statistik dataset: distribusi label per emosi — asli (Label2d train) vs hasil LP
        (diterima/ditolak, berapa dari dataset wajah baru). Untuk melihat ketimpangan kelas
        dan menilai apakah augmentasi sudah cukup."""
        if not self.path_json_augment:
            messagebox.showinfo("Statistik Dataset", "Buka folder dataset terlebih dahulu.")
            return
        import csv as _csv
        gen_dir = self._lp_generated_dir() or ""
        with self._lp_json_lock:
            rejected = self._lp_load_review()
            labels   = self._lp_load_labels()

        # Hitung hasil LP per emosi (pakai label final manual; default = emosi target)
        st = {l: {"asli": 0, "lp": 0, "wajah_baru": 0, "ditolak": 0} for l in LABELS}
        for path, emo in self._lp_list_generated():
            rel = os.path.relpath(path, gen_dir).replace("\\", "/")
            lab = labels.get(rel) or self._lp_default_label(emo)
            tolak = rel in rejected
            for l in LABELS:
                if lab.get(l, 0) != 1:
                    continue
                if tolak:
                    st[l]["ditolak"] += 1
                else:
                    st[l]["lp"] += 1
                    if self._lp_path_is_newface(path):
                        st[l]["wajah_baru"] += 1

        # Label asli dari Label2d/train.csv (bila sudah pernah split)
        train_csv = os.path.join(self._lp_label2d_base(), "train.csv")
        ada_base = os.path.exists(train_csv)
        if ada_base:
            with open(train_csv, newline="") as f:
                rows = list(_csv.reader(f))
            kol = {l: rows[0].index(l) for l in LABELS if l in rows[0]}
            for r in rows[1:]:
                for l, i in kol.items():
                    if i < len(r) and r[i] == "1":
                        st[l]["asli"] += 1

        baris = ["Distribusi label kelas (train):", ""]
        for l in LABELS:
            d = st[l]
            total = d["asli"] + d["lp"]
            baris.append(
                f"{l}:\n"
                f"   asli {d['asli']}  +  LP diterima {d['lp']} "
                f"(dari wajah baru: {d['wajah_baru']})  =  {total}"
                + (f"   |   ditolak {d['ditolak']}" if d["ditolak"] else ""))
        if not ada_base:
            baris.append("\n(Label2d/train.csv belum ada — kolom 'asli' = 0. "
                         "Buat split 2D dulu untuk angka asli.)")
        n_trash = len(self._lp_list_trashed())
        if n_trash:
            baris.append(f"\nDi _trash (dibuang, bisa dipulihkan): {n_trash} gambar.")
        baris.append("\n'LP diterima' = yang akan masuk saat Buat Dataset. "
                     "Pakai angka ini untuk menyeimbangkan kelas minoritas.")
        messagebox.showinfo("Statistik Dataset", "\n".join(baris))

    def _lp_auto_reject_mismatch(self):
        """QA cepat untuk RIBUAN hasil: tandai TOLAK semua gambar yang menurut deteksi AI
        TIDAK mengandung emosi targetnya. Butuh 'Deteksi AI Semua' dijalankan dulu.
        Bisa dibatalkan per-gambar (Batal Tolak) — tidak menghapus file apa pun."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        gen_dir = self._lp_generated_dir() or ""
        with self._lp_json_lock:
            rejected = self._lp_load_review()
            ai       = self._lp_load_ai_labels()
        kandidat, tanpa_ai = [], 0
        for path, emo in self._lp_list_generated():
            rel = os.path.relpath(path, gen_dir).replace("\\", "/")
            if rel in rejected:
                continue
            lab = ai.get(rel)
            if not lab:
                tanpa_ai += 1
                continue
            if lab.get(emo, 0) == 0:      # AI tidak melihat emosi target di gambar ini
                kandidat.append(rel)
        if not kandidat:
            pesan = "Tidak ada mismatch AI vs target."
            if tanpa_ai:
                pesan += (f"\n\n{tanpa_ai} gambar belum punya hasil deteksi AI — "
                          "jalankan 'Deteksi AI Semua' dulu.")
            messagebox.showinfo("Auto-Tolak", pesan)
            return
        if not messagebox.askyesno(
                "Auto-Tolak (AI != target)",
                f"Tandai TOLAK {len(kandidat)} gambar yang menurut AI tidak mengandung "
                f"emosi targetnya?\n\n"
                + (f"({tanpa_ai} gambar belum dideteksi AI — dilewati.)\n" if tanpa_ai else "")
                + "Tidak ada file dihapus; bisa dibatalkan per-gambar (Batal Tolak)."):
            return
        with self._lp_json_lock:
            rejected = self._lp_load_review()
            rejected |= set(kandidat)
            self._lp_save_review(rejected)
        self._lp_refresh_review()
        if lp:
            lp.update_progress(
                f"Auto-Tolak: {len(kandidat)} gambar ditandai tolak (filter 'Ditolak' untuk meninjau).",
                "#10b981")

    def _lp_delete_rejected(self):
        """Buang gambar yang ditolak ke folder _trash/ (BUKAN hapus permanen) —
        user bisa memulihkan manual bila berubah pikiran (prinsip undo/recovery)."""
        with self._lp_json_lock:
            rejected = self._lp_load_review()
        if not rejected:
            messagebox.showinfo("Tinjau Hasil", "Tidak ada gambar yang ditolak.")
            return
        gen_dir = self._lp_generated_dir() or ""
        trash_dir = os.path.join(gen_dir, "_trash")
        if not messagebox.askyesno(
                "Buang yang Ditolak",
                f"Pindahkan {len(rejected)} gambar ditolak ke folder _trash?\n\n"
                f"{trash_dir}\n\n(Tidak dihapus permanen — bisa dipulihkan manual.)"):
            return
        import shutil
        n = 0
        for rel in list(rejected):
            sumber = os.path.join(gen_dir, rel.replace("/", os.sep))
            tujuan = os.path.join(trash_dir, rel.replace("/", os.sep))
            try:
                if os.path.exists(sumber):
                    os.makedirs(os.path.dirname(tujuan), exist_ok=True)
                    shutil.move(sumber, tujuan)
                n += 1
            except Exception as e:
                self._lp_report_error(f"Gagal memindah {rel}: {e}")
        with self._lp_json_lock:
            self._lp_save_review(set())
        self._lp_refresh_review()
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp:
            lp.update_progress(f"{n} gambar ditolak dipindah ke _trash — masih bisa dilihat "
                               f"lewat filter 'Dibuang (_trash)' dan dipulihkan.", "#10b981")

    def _lp_trash_dir(self) -> str | None:
        d = self._lp_generated_dir()
        return os.path.join(d, "_trash") if d else None

    def _lp_list_trashed(self) -> list:
        """List (abs_path, emo, rel_asli) gambar yang pernah dibuang ke _trash —
        supaya tetap BISA DILIHAT di panel (filter 'Dibuang') dan dipulihkan."""
        import glob as _glob
        t = self._lp_trash_dir()
        if not t or not os.path.isdir(t):
            return []
        out = []
        for p in sorted(_glob.glob(os.path.join(t, "**", "*.jpg"), recursive=True)):
            rel = os.path.relpath(p, t).replace("\\", "/")
            emo = next((x for x in rel.split("/") if x in LABELS), "?")
            out.append((p, emo, rel))
        return out

    def _lp_restore_trash(self):
        """Pulihkan SEMUA gambar dari _trash ke tempat asalnya. Gambar yang dipulihkan
        diberi status DITOLAK lagi supaya keputusannya bisa ditinjau ulang (bukan
        langsung ikut dataset)."""
        import shutil
        trashed = self._lp_list_trashed()
        if not trashed:
            messagebox.showinfo("Pulihkan dari _trash", "_trash kosong — tidak ada gambar untuk dipulihkan.")
            return
        gen_dir = self._lp_generated_dir() or ""
        if not messagebox.askyesno(
                "Pulihkan dari _trash",
                f"Kembalikan {len(trashed)} gambar dari _trash ke tempat asal?\n\n"
                "(Statusnya tetap DITOLAK — batalkan tolaknya satu per satu bila ingin dipakai.)"):
            return
        n, pulih = 0, set()
        for p, _emo, rel in trashed:
            tujuan = os.path.join(gen_dir, rel.replace("/", os.sep))
            try:
                os.makedirs(os.path.dirname(tujuan), exist_ok=True)
                shutil.move(p, tujuan)
                pulih.add(rel)
                n += 1
            except Exception as e:
                self._lp_report_error(f"Gagal memulihkan {rel}: {e}")
        with self._lp_json_lock:
            self._lp_save_review(self._lp_load_review() | pulih)
        self._lp_refresh_review()
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp:
            lp.update_progress(f"{n} gambar dipulihkan dari _trash (status: ditolak — "
                               f"batal-tolak yang ingin dipakai).", "#10b981")

    def _lp_restore_one(self, trash_path: str):
        """Pulihkan SATU gambar dari _trash (dobel-klik gambar di filter 'Dibuang')."""
        import shutil
        t = self._lp_trash_dir() or ""
        gen_dir = self._lp_generated_dir() or ""
        rel = os.path.relpath(trash_path, t).replace("\\", "/")
        tujuan = os.path.join(gen_dir, rel.replace("/", os.sep))
        try:
            os.makedirs(os.path.dirname(tujuan), exist_ok=True)
            shutil.move(trash_path, tujuan)
        except Exception as e:
            self._lp_report_error(f"Gagal memulihkan {rel}: {e}")
            return
        with self._lp_json_lock:
            self._lp_save_review(self._lp_load_review() | {rel})
        self._lp_refresh_review()
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        if lp:
            lp.update_progress(f"Dipulihkan dari _trash: {rel} (status: ditolak).", "#10b981")

    def _on_manual_check(self, frame_idx: int, label: str, value: bool):
        """Dipanggil saat user centang/uncentang checkbox manual label."""
        if not self.video_files:
            return
        vp  = self.video_files[self.current_index]
        rel = os.path.relpath(vp, self.root_folder)
        if rel not in self.manual_labels:
            self._init_manual_for_current()
        self.manual_labels[rel][str(frame_idx)][label] = int(value)

        # Mutual exclusion: jika label diaktifkan, matikan pasangannya & update checkbox-nya
        if int(value) == 1 and label in MUTUAL_EXCLUSIVE:
            pair = MUTUAL_EXCLUSIVE[label]
            self.manual_labels[rel][str(frame_idx)][pair] = 0
            self.left_panel.update_manual_checkboxes(
                frame_idx, self.manual_labels[rel][str(frame_idx)]
            )

        self._save_manual_labels()
        # Refresh tampilan pill label frame ini dari state otoritatif (clicked + pasangan mutex)
        self.left_panel.update_manual_checkboxes(frame_idx, self.manual_labels[rel][str(frame_idx)])
        # Update highlight frame yang berubah sesuai label aktif
        active_lbl = self.active_frame_label.get()
        if label == active_lbl:
            # Rebuild frame_data dari manual_labels untuk frame ini
            fa  = self.frame_annotations.get(rel, {})
            mfr = dict(self.manual_labels[rel].get(str(frame_idx), {}))
            mfr["_rejected"] = fa.get(str(frame_idx), {}).get("_rejected", False)
            # render_frames tidak dipanggil supaya tidak flicker — hanya update satu frame
            self.left_panel._current_frame_annotations[str(frame_idx)] = mfr
            self.left_panel.update_single_frame_highlight(frame_idx, active_lbl, int(value))

    # ── Rules ─────────────────────────────────────────────────────────────────

    def _load_rules(self):
        """Muat rules dari disk ke self.rules."""
        if self.path_json_rules:
            self.rules = load_rules(self.path_json_rules)
        # Sync inline rules content in left panel
        lp_rc = getattr(getattr(self, "left_panel", None), "rules_content", None)
        if lp_rc is not None:
            lp_rc._load_from_rules(self.rules)

    def _save_rules(self, rules: dict):
        """Simpan rules ke disk dan update self.rules."""
        self.rules = rules
        if self.path_json_rules:
            save_rules(self.path_json_rules, rules)
        print(f"[Rules] Tersimpan ke {self.path_json_rules}")

    def _open_rules_panel(self):
        """Buka rules editor di area galeri kiri."""
        if hasattr(self, "left_panel"):
            self.left_panel.show_rules_mode()
        # self._rules_panel stays None for backwards compat

    def _recalculate_all(self, rules: dict, extra_path_sentinel: str | None = None):
        """
        Hitung ulang batch_history dari raw cache + siglip cache dengan rules baru.
        Berjalan di background thread.

        extra_path_sentinel: None = timpa saat ini, "__batch_name__<name>" = simpan ke file baru.
        """
        if not self.batch_history:
            messagebox.showinfo("Recalculate", "Belum ada riwayat batch AI. Proses batch dulu.")
            return
        if not self.path_dir_raw_cache or not self.path_dir_siglip_cache:
            messagebox.showinfo("Recalculate", "Buka folder dataset terlebih dahulu.")
            return

        # Resolve extra_path dari sentinel yang dikirim RulesPanel
        extra: str | None = None
        if extra_path_sentinel and extra_path_sentinel.startswith("__batch_name__"):
            name = extra_path_sentinel[len("__batch_name__"):]
            base_dir = os.path.dirname(self.path_json_batch_history)
            extra = os.path.join(base_dir, f"batch_history_{name}.json")

        thresholds = [v.get() for v in self.right_panel.threshold_vars]
        self.right_panel.lbl_batch_status.configure(
            text="Menghitung ulang…", text_color="#fbbf24"
        )

        def worker():
            from core.recalculate import recalculate_batch
            try:
                new_history, new_fa, skipped = recalculate_batch(
                    batch_history=self.batch_history,
                    rules=rules,
                    thresholds=thresholds,
                    raw_cache_dir=self.path_dir_raw_cache,
                    siglip_cache_dir=self.path_dir_siglip_cache,
                    frame_annotations=self.frame_annotations,
                )
                with self.save_lock:
                    self.batch_history     = new_history
                    self.frame_annotations = new_fa
                    save_batch_history(self.path_json_batch_history, self.batch_history)
                    if extra:
                        save_batch_history(extra, self.batch_history)
                    save_frame_annotations(self.path_json_frames, self.frame_annotations)

                def on_done(extra_path=extra, _rules=rules):
                    for rel in self.frame_annotations:
                        self._sync_manual_from_ai(rel, save=False)
                    # Hapus entri stale di manual_labels yang tidak ada di frame_annotations
                    stale = [k for k in list(self.manual_labels) if k not in self.frame_annotations]
                    for k in stale:
                        del self.manual_labels[k]
                    self._save_manual_labels(extra_batch_path=extra_path if extra_path else None)

                    total  = len(self.batch_history)
                    done   = total - skipped
                    msg    = f"Selesai: {done}/{total} video dihitung ulang"
                    if skipped:
                        msg += f" ({skipped} tanpa cache)"
                    if extra_path:
                        msg += f" → {os.path.basename(extra_path)}"
                    self.right_panel.lbl_batch_status.configure(
                        text=msg, text_color="#10b981"
                    )
                    self.right_panel.update_statistics(self.batch_history)
                    # Simpan rules + threshold ke __meta__ di kedua file batch
                    thrs = [v.get() for v in self.right_panel.threshold_vars]
                    update_batch_meta(self.path_json_batch_history, _rules, thrs)
                    if extra_path:
                        update_batch_meta(extra_path, _rules, thrs)
                    # Regenerate viz overlay untuk video aktif dengan rules baru
                    self._regenerate_viz_for_current(_rules)
                    # Refresh gallery — fast path karena rel_path cache masih valid
                    self.refresh_frame_gallery()
                    # Auto-parse: update Label2d CSV dari hasil recalculate terbaru
                    self._split_dataset_2d(silent=True, force_default=True)
                    # Refresh dropdown batch history (mungkin ada file baru)
                    self.right_panel.refresh_batch_files()
                    print(f"[Recalc] {msg}")

                self.root.after(0, on_done)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self.right_panel.lbl_batch_status.configure(
                    text=f"Error: {err[:60]}", text_color="#ef4444"
                ))
                import traceback; traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def _change_output_folder(self):
        """Buka dialog untuk memilih folder output yang berbeda."""
        if not self.root_folder:
            messagebox.showinfo("Info", "Buka folder dataset terlebih dahulu.")
            return
        folder = filedialog.askdirectory(
            title="Pilih Folder Output",
            initialdir=os.path.dirname(self.path_csv_annotations) if self.path_csv_annotations else self.root_folder,
        )
        if not folder:
            return

        self.path_csv_annotations    = os.path.join(folder, "annotations_bener.csv")
        self.path_csv_flagged        = os.path.join(folder, "flagged_videos.csv")
        self.path_json_frames        = os.path.join(folder, "frame_annotations.json")
        self.path_json_batch_history = os.path.join(folder, "batch_history.json")
        self.path_json_skipped       = os.path.join(folder, "skipped_videos.json")
        self.path_json_thresholds    = os.path.join(folder, "thresholds.json")
        self.path_json_rules         = os.path.join(folder, "rules.json")
        self.path_json_augment       = os.path.join(folder, "augment_marks.json")
        self.path_dir_cropped        = os.path.join(folder, "cropped_faces")
        self.path_dir_raw_cache      = os.path.join(folder, "raw_cache")
        self.path_dir_siglip_cache   = os.path.join(folder, "siglip_cache")
        os.makedirs(self.path_dir_cropped, exist_ok=True)
        os.makedirs(self.path_dir_raw_cache, exist_ok=True)
        os.makedirs(self.path_dir_siglip_cache, exist_ok=True)

        self._load_data()
        self.load_video()
        messagebox.showinfo("Output", f"Folder output diubah ke:\n{folder}")

    def _restart_batch(self):
        """
        Reset semua label, batch_history, frame_annotations, dan manual_labels ke 0.
        Semua video diproses ulang dari awal.
        """
        if not self.path_json_batch_history:
            return
        if not messagebox.askyesno(
            "Restart Batch",
            "Reset semua label ke 0?\n"
            "Batch history, frame annotations, dan label manual akan dihapus semua."
        ):
            return
        self.batch_history     = {}
        self.frame_annotations = {}
        self.manual_labels     = {}
        save_batch_history(self.path_json_batch_history, self.batch_history)
        save_frame_annotations(self.path_json_frames, self.frame_annotations)
        self._save_manual_labels()
        self.right_panel.update_statistics(self.batch_history)
        self.right_panel.update_manual_statistics(self.manual_labels)
        self.right_panel.lbl_batch_status.configure(
            text="Semua label direset ke 0", text_color="#10b981"
        )
        print("[Batch] Semua label direset ke 0.")

    def _update_flag_count(self):
        """Perbarui label counter flag di topbar."""
        n = len(self.flagged_data)
        if n > 0:
            self.lbl_flag_count.configure(text=f"{n} flagged")
        else:
            self.lbl_flag_count.configure(text="")

    def save_current_state(self):
        """
        Simpan state video saat ini ke disk.

        Jika video di-flag: hapus dari frame_annotations, tambahkan ke flagged_data.
        Selalu tulis ulang flagged_videos.csv, dan frame_annotations.json.
        """
        if not self.video_files: return
        vp  = self.video_files[self.current_index]
        rel = os.path.relpath(vp, self.root_folder)

        if self.flag_var.get():
            self.flagged_data.add(rel)
            self.frame_annotations.pop(rel, None)
        else:
            self.flagged_data.discard(rel)

        save_flagged(self.path_csv_flagged, self.flagged_data)
        save_frame_annotations(self.path_json_frames, self.frame_annotations)
        self._update_flag_count()

    # Video playback

    def load_video(self):
        """
        Muat dan tampilkan video pada self.current_index.

        Restore label dan status flag dari data yang sudah tersimpan.
        Langsung mulai playback setelah video dimuat.
        """
        self.is_playing = False
        if self.cap: self.cap.release()
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        import cv2  # lazy — cached after first call
        vp = self.video_files[self.current_index]
        self.cap = cv2.VideoCapture(vp)
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_val <= 0 or fps_val > 60:
            # Beberapa codec melaporkan FPS salah (mis. 1000). Estimasi dari timestamp.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames - 1)
            self.cap.read()
            end_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps_val = (self.total_frames - 1) / (end_ms / 1000.0) if end_ms > 0 else 30.0
        self.fps = max(1.0, fps_val)
        self.left_panel.slider.configure(to=self.total_frames)
        self.left_panel.slider.set(0)
        self.current_frame = self.play_start_time = self.play_start_frame = 0

        rel = os.path.relpath(vp, self.root_folder)
        self.lbl_info.configure(text=os.path.basename(rel))
        self.lbl_fps.configure(text=f"FPS: {self.fps:.2f}")
        self.lbl_progress.configure(text=f"{self.current_index + 1} / {len(self.video_files)}")

        self.flag_var.set(rel in self.flagged_data)

        # Refresh checkbox semi-manual jika mode aktif
        if self.semi_manual_var.get():
            self._init_manual_for_current()

        self.refresh_frame_gallery()
        self.toggle_play()

    def update_frame(self):
        """
        Loop playback berbasis waktu nyata — dipanggil rekursif via root.after().

        Menghitung frame target dari waktu yang sudah berlalu sejak play dimulai,
        sehingga playback tidak bergantung pada kecepatan CPU.
        """
        import cv2  # lazy — already cached after load_video() runs, dict lookup only
        if not self.is_playing or not self.cap: return
        t_frame = self.play_start_frame + int(
            (time.time() - self.play_start_time) * self.fps
        )
        if t_frame >= self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.play_start_time = time.time()
            self.play_start_frame = 0
            t_frame = 0

        frames_behind = t_frame - self.current_frame
        ret, frame = False, None
        
        if frames_behind > 0:
            # Jika ketinggalan jauh (> 5 frame), terpaksa pakai set() yang berat
            if frames_behind > 5:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame)
                ret, frame = self.cap.read()
                self.current_frame = t_frame
            else:
                # Jika cuma 1-5 frame, panggil read() berulang lebih ringan daripada set()
                for _ in range(frames_behind):
                    ret, frame = self.cap.read()
                    self.current_frame += 1
            
            if ret and frame is not None:
                self.left_panel.slider.set(self.current_frame)
                self.left_panel.show_video_frame(frame)

        self.after_id = self.root.after(15, self.update_frame)

    def toggle_play(self):
        """Toggle antara play dan pause. Jika mulai play, inisialisasi timer playback."""
        if not self.cap: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_start_time  = time.time()
            self.play_start_frame = self.current_frame
            self.update_frame()

    def on_slider_move(self, val):
        """Sync posisi video ke slider. Dipanggil oleh callback slider saat digeser."""
        import cv2
        if not self.cap: return
        self.current_frame    = int(float(val))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.play_start_time  = time.time()
        self.play_start_frame = self.current_frame

    def seek_to_frame(self, frame_idx: int):
        """
        Seek video ke POSISI ASLI thumbnail ke-frame_idx (0 atau 1).

        Thumbnail diekstrak pada 25% dan 75% durasi video — rumus (2i+1)/(2N) di
        utils/video.py — jadi seek harus memakai rumus yang SAMA supaya gambar yang
        tampil di player cocok dengan thumbnail yang diklik.
        """
        import cv2
        if not self.cap or not self.video_files: return
        target = int(self.total_frames * (2 * frame_idx + 1) / 4)
        self.current_frame = self.play_start_frame = target
        self.play_start_time = time.time()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.left_panel.slider.set(target)
        ret, frame = self.cap.read()
        if ret:
            self.left_panel.show_video_frame(frame)

    # Navigation

    def _arrow_target_lp(self):
        """Panel LP bila sedang TAMPIL (panah keyboard dialihkan ke pemeriksa hasil)."""
        lp = getattr(getattr(self, "left_panel", None), "lp_panel_content", None)
        try:
            tampil = bool(self.left_panel._lp_container.winfo_ismapped())
        except Exception:
            tampil = False
        return lp if (tampil and lp and getattr(lp, "_built", False)) else None

    def _on_arrow(self, delta: int):
        """Tombol panah kiri/kanan, sadar-mode:
        - galeri  : pindah video (kanan = save & next, kiri = sebelumnya) — perilaku lama;
        - panel LP: (1) pindah gambar di pemeriksa 'Tinjau & Label' bila daftar tinjau ada;
                    (2) kalau belum, scrub video HASIL/DRIVING yang sedang tampil;
                    (3) kalau itu pun belum ada, pindah antar frame sumber bertanda.
        Saat kursor sedang di kotak input, panah dibiarkan menggerakkan kursor teks."""
        fokus = self.root.focus_get()
        if isinstance(fokus, tk.Entry):
            return                      # sedang mengetik → jangan rebut panah
        lp = self._arrow_target_lp()
        if lp is not None:
            if lp.review_items:
                lp._tinjau_geser(delta)
            elif lp.geser_frame_relatif(delta):
                pass                    # scrub video hasil/driving yang sedang tampil
            else:
                self._goto_lp_mark(delta)
            return
        if delta > 0:
            self.save_and_next()
        else:
            self.prev_video()

    def save_and_next(self):
        """Simpan state video saat ini lalu pindah ke video berikutnya."""
        if not self.video_files: return
        self.save_current_state()
        if self.current_index < len(self.video_files) - 1:
            self.current_index += 1
            self.load_video()

    def prev_video(self):
        """Simpan state video saat ini lalu kembali ke video sebelumnya."""
        if self.current_index > 0:
            self.save_current_state()
            self.current_index -= 1
            self.load_video()

    def skip_video(self):
        """
        Lewati video saat ini tanpa menyimpan label.

        Video ditambahkan ke skipped_videos dan langsung disimpan ke disk.
        """
        if not self.video_files: return
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        self.skipped_videos.add(rel)
        save_skipped(self.path_json_skipped, self.skipped_videos)
        print(f"skip: {rel}")
        if self.current_index < len(self.video_files) - 1:
            self.current_index += 1
            self.load_video()
        else:
            messagebox.showinfo("Selesai", "Semua video sudah diproses atau di-skip.")

    def jump_to_video(self):
        """
        Loncat langsung ke nomor video yang diketik di kolom jump.

        Input berupa nomor 1-based. Simpan state video saat ini sebelum berpindah.
        """
        if not self.video_files: return
        try:
            n = int(self.jump_entry.get().strip())
        except ValueError:
            return
        idx = max(0, min(n - 1, len(self.video_files) - 1))
        self.jump_entry.delete(0, "end")
        self.root.focus_set()         # lepas fokus dari kotak → panah ◀/▶ aktif lagi
        if idx == self.current_index: return
        self.save_current_state()
        self.current_index = idx
        self.load_video()

    # Frame gallery & label helpers

    def _set_active_tab(self, label: str):
        """Set label aktif di tab frame gallery dan refresh tampilan galeri."""
        self.active_frame_label.set(label)
        self.left_panel.set_active_tab_highlight(label)
        self.refresh_frame_gallery()

    def _active_vid_data(self, rel_path: str) -> dict:
        """Kembalikan data label frame untuk video rel_path sesuai mode aktif.
        Mode manual: ambil dari manual_labels (jika ada), fallback ke frame_annotations.
        Mode AI: selalu ambil dari frame_annotations."""
        if self.semi_manual_var.get():
            ml = self.manual_labels.get(rel_path)
            if ml:
                # Manual mode: _rejected dan label semuanya dari manual_labels
                # (frame_annotations tidak disentuh agar AI result tetap bersih)
                return {fi_str: dict(fdata) for fi_str, fdata in ml.items()}
        return self.frame_annotations.get(rel_path, {})

    def refresh_frame_gallery(self, *args):
        """
        Render ulang galeri frame.

        Fast path: jika cache masih untuk video yang sama (tab switch / viz toggle),
                   langsung re-render tanpa memanggil prepare_cropped_frames lagi.
        Slow path: video baru — tampilkan "memuat…", mulai background thread,
                   terapkan hasilnya ke UI via _apply_gallery_result.
        """
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)
        active_lbl = self.active_frame_label.get()
        try:
            self._update_neutral_label()
            # Index frame acuan netral (untuk marker '★ NETRAL' di galeri), -1 bila bukan video netral
            from utils.person_neutral import get_person_neutral_frame
            nf = get_person_neutral_frame(self._dataset_dir(), rel_path)
            self.left_panel._neutral_frame_idx = nf if nf is not None else -1
        except Exception:
            self.left_panel._neutral_frame_idx = -1

        self._refresh_augment_widgets(rel_path)   # sinkronkan tombol augmentasi

        # ── Fast path: cache hit ─────────────────────────────────────────────
        if self._gallery_cache["rel_path"] == rel_path:
            pil_images = self._gallery_cache["pil_images"]
            viz_images = self._gallery_cache["viz_images"]
            self.viz_images = viz_images
            n = len(pil_images) or 2
            if pil_images:
                vid_data = self._active_vid_data(rel_path)
                display  = viz_images if (self.show_viz.get() and viz_images) else pil_images
                self.left_panel.render_frames(display, vid_data, active_lbl)
            self.left_panel.update_frame_quality(
                self._gallery_cache["no_face_count"],
                self._gallery_cache["multi_fc"],
                self._count_rejected(rel_path, n),
                n_frames=n,
            )
            hist = self.batch_history.get(rel_path)
            if hist:
                for i, lbl in enumerate(LABELS):
                    pl  = hist["per_label"].get(str(i), {})
                    fsc = pl.get("frame_scores", [])
                    thr = pl.get("threshold")
                    if fsc:
                        self.right_panel.update_ai_score_bar(lbl, fsc, thr)
            return

        # ── Slow path: new video ─────────────────────────────────────────────
        self._gallery_version += 1
        my_version = self._gallery_version

        self.left_panel.show_loading()

        # Tampilkan skor AI dari history langsung (sebelum background selesai)
        hist = self.batch_history.get(rel_path)
        if hist:
            for i, lbl in enumerate(LABELS):
                pl  = hist["per_label"].get(str(i), {})
                fsc = pl.get("frame_scores", [])
                thr = pl.get("threshold")
                if fsc:
                    self.right_panel.update_ai_score_bar(lbl, fsc, thr)

        def worker():
            try:
                from utils.video import prepare_cropped_frames
                result = prepare_cropped_frames(
                    vp, self.root_folder, self.path_dir_cropped,
                    raw_cache_dir=self.path_dir_raw_cache if hasattr(self, "path_dir_raw_cache") else None,
                    cfg=self.rules if hasattr(self, "rules") else None,
                )
                if getattr(self, "_shutting_down", False) or self._gallery_version != my_version:
                    return  # app menutup / hasil kedaluwarsa — buang
                self.root.after(0, lambda: self._apply_gallery_result(rel_path, result, my_version))
            except RuntimeError as e:
                # Race saat app ditutup (MediaPipe/Tk shutdown) — abaikan diam-diam
                if "interpreter shutdown" in str(e) or "main thread is not in main loop" in str(e):
                    return
                print(f"[Gallery] worker error: {e}")
            except Exception as e:
                if not getattr(self, "_shutting_down", False):
                    print(f"[Gallery] worker error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _apply_gallery_result(self, rel_path: str, result: tuple, version: int):
        """Terapkan hasil prepare_cropped_frames ke UI (harus dipanggil dari main thread)."""
        if self._gallery_version != version:
            return
        pil_images, no_face_count, multi_fc, landmark_results, viz_images = result

        self._gallery_cache = {
            "rel_path":        rel_path,
            "pil_images":      pil_images,
            "viz_images":      viz_images,
            "landmark_results": landmark_results,
            "no_face_count":   no_face_count,
            "multi_fc":        multi_fc,
        }
        self.viz_images = viz_images

        # Auto-reject frame tanpa wajah (hanya jika user belum pernah menyentuh flag-nya)
        if landmark_results:
            changed = False
            for i, lr in enumerate(landmark_results):
                if not lr.face_found:
                    if rel_path not in self.frame_annotations:
                        self.frame_annotations[rel_path] = {}
                    if str(i) not in self.frame_annotations[rel_path]:
                        self.frame_annotations[rel_path][str(i)] = {l: 0 for l in LABELS}
                    frame_data = self.frame_annotations[rel_path][str(i)]
                    if "_rejected" not in frame_data:
                        frame_data["_rejected"] = True
                        changed = True
            if changed and self.path_json_frames:
                save_frame_annotations(self.path_json_frames, self.frame_annotations)

        n = len(pil_images) or 2
        self.left_panel.update_frame_quality(
            no_face_count, multi_fc,
            self._count_rejected(rel_path, n),
            n_frames=n,
        )

        if not pil_images:
            for cv_widget in self.left_panel.frame_canvases:
                cv_widget.delete("all")
                cv_widget.configure(highlightbackground="#333")
            return

        active_lbl = self.active_frame_label.get()
        vid_data   = self._active_vid_data(rel_path)
        display    = viz_images if (self.show_viz.get() and viz_images) else pil_images
        self.left_panel.render_frames(display, vid_data, active_lbl)

        hist = self.batch_history.get(rel_path)
        if hist:
            for i, lbl in enumerate(LABELS):
                pl  = hist["per_label"].get(str(i), {})
                fsc = pl.get("frame_scores", [])
                thr = pl.get("threshold")
                if fsc:
                    self.right_panel.update_ai_score_bar(lbl, fsc, thr)

        # Regenerasi viz dengan rules aktif jika diminta (misal setelah batch switch)
        if self._viz_regen_requested and landmark_results:
            self._viz_regen_requested = False
            self._regenerate_viz_for_current(self.rules)
            if self.show_viz.get():
                new_viz = self._gallery_cache.get("viz_images", [])
                if new_viz:
                    self.left_panel.render_frames(
                        new_viz,
                        self._active_vid_data(rel_path),
                        self.active_frame_label.get(),
                    )

    def _regenerate_viz_for_current(self, rules: dict):
        """










































        
        Regenerate viz overlay untuk video yang sedang ditampilkan menggunakan rules baru.
        Menggunakan landmark_results yang tersimpan di gallery cache — tidak re-run MediaPipe.
        Update gallery cache, self.viz_images, dan file viz di disk.
        """
        import cv2, numpy as np
        from PIL import Image
        from core.landmark_analyzer import compute_emotion_scores, draw_landmark_viz

        rel_path         = self._gallery_cache.get("rel_path")
        pil_images       = self._gallery_cache.get("pil_images", [])
        landmark_results = self._gallery_cache.get("landmark_results", [])

        if not pil_images or not landmark_results or rel_path is None:
            return

        base_name    = os.path.splitext(rel_path)[0]
        viz_crop_dir = os.path.join(self.path_dir_cropped, "viz", base_name)
        os.makedirs(viz_crop_dir, exist_ok=True)

        new_viz = []
        for i, (pil_img, lr) in enumerate(zip(pil_images, landmark_results)):
            bgr       = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            scores    = compute_emotion_scores(lr, rules)
            viz_bgr   = draw_landmark_viz(bgr, lr, scores)
            viz_path  = os.path.join(viz_crop_dir, f"frame_{i:02d}_viz.jpg")
            cv2.imwrite(viz_path, viz_bgr)
            new_viz.append(Image.fromarray(cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)))

        self._gallery_cache["viz_images"] = new_viz
        self.viz_images = new_viz

    def _count_rejected(self, rel_path: str, n_frames: int = 4) -> int:
        if self.semi_manual_var.get():
            src = self.manual_labels.get(rel_path, {})
        else:
            src = self.frame_annotations.get(rel_path, {})
        return sum(1 for i in range(n_frames) if src.get(str(i), {}).get("_rejected", False))

    def toggle_frame_reject(self, frame_idx: int):
        """
        Double-klik frame → toggle status rejected.

        Frame rejected ditampilkan dengan overlay merah dan tidak dihitung
        dalam voting prediksi AI saat proses inferensi.
        Hanya aktif saat mode Label Semi Manual ON.
        """
        if not self.video_files: return
        if not self.semi_manual_var.get():
            self.right_panel.lbl_batch_status.configure(
                text="Aktifkan 'Label Semi Manual' untuk mengedit",
                text_color="#f59e0b",
            )
            return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)

        # Pastikan manual_labels untuk video ini sudah ada
        if rel_path not in self.manual_labels:
            self._init_manual_for_current()
        if str(frame_idx) not in self.manual_labels[rel_path]:
            self.manual_labels[rel_path][str(frame_idx)] = {l: 0 for l in LABELS}

        # Tulis ke manual_labels, BUKAN frame_annotations — agar AI result tidak ikut berubah
        current = self.manual_labels[rel_path][str(frame_idx)].get("_rejected", False)
        self.manual_labels[rel_path][str(frame_idx)]["_rejected"] = not current

        self.refresh_frame_gallery()  # fast path — cache masih valid
        self._save_manual_labels()

    def toggle_single_frame(self, frame_idx: int):
        """
        Toggle label aktif pada satu frame — hanya aktif saat mode Label Semi Manual ON.
        Edit manual_labels (bukan frame_annotations).
        """
        if not self.video_files: return
        if not self.semi_manual_var.get():
            # Beri tahu user bahwa editing perlu mode semi manual
            self.right_panel.lbl_batch_status.configure(
                text="Aktifkan 'Label Semi Manual' untuk mengedit label",
                text_color="#f59e0b",
            )
            return

        vp         = self.video_files[self.current_index]
        rel_path   = os.path.relpath(vp, self.root_folder)
        active_lbl = self.active_frame_label.get()

        if rel_path not in self.manual_labels:
            self._init_manual_for_current()
        if str(frame_idx) not in self.manual_labels[rel_path]:
            self.manual_labels[rel_path][str(frame_idx)] = {l: 0 for l in LABELS}

        current_val = self.manual_labels[rel_path][str(frame_idx)].get(active_lbl, 0)
        new_val     = 1 if current_val == 0 else 0
        self.manual_labels[rel_path][str(frame_idx)][active_lbl] = new_val

        # Mutual exclusion: jika label diaktifkan, matikan pasangannya
        if new_val == 1 and active_lbl in MUTUAL_EXCLUSIVE:
            pair = MUTUAL_EXCLUSIVE[active_lbl]
            self.manual_labels[rel_path][str(frame_idx)][pair] = 0

        self._save_manual_labels()

        # Update checkbox di panel kiri
        self.left_panel.update_manual_checkboxes(
            frame_idx, self.manual_labels[rel_path][str(frame_idx)]
        )
        # Update highlight canvas
        self.left_panel.update_single_frame_highlight(frame_idx, active_lbl, new_val)
        fa  = self.frame_annotations.get(rel_path, {})
        mfr = dict(self.manual_labels[rel_path].get(str(frame_idx), {}))
        mfr["_rejected"] = fa.get(str(frame_idx), {}).get("_rejected", False)
        self.left_panel._current_frame_annotations[str(frame_idx)] = mfr

    def toggle_frame_label_by_dot(self, frame_idx: int, y: int):
        """
        Toggle label spesifik dengan klik pada dot strip — hanya saat mode Semi Manual ON.
        y: koordinat klik pada dot_cv (tinggi 360px).
        Label j ada di cy = start_y + j * (2*dot_r + gap) + dot_r.
        """
        if not self.video_files: return
        if not self.semi_manual_var.get():
            self.right_panel.lbl_batch_status.configure(
                text="Aktifkan 'Label Semi Manual' untuk mengedit label",
                text_color="#f59e0b",
            )
            return

        # Hitung index label dari posisi y (harus sync dengan _draw_frame_dots)
        dot_r   = 7
        gap     = 22
        n       = len(LABELS)
        total_h = n * (2 * dot_r) + (n - 1) * gap
        start_y = (360 - total_h) // 2

        j = (y - start_y) // (2 * dot_r + gap)
        if j < 0 or j >= n:
            return  # klik di luar area dot

        label = LABELS[j]
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)

        if rel_path not in self.manual_labels:
            self._init_manual_for_current()
        if str(frame_idx) not in self.manual_labels[rel_path]:
            self.manual_labels[rel_path][str(frame_idx)] = {l: 0 for l in LABELS}

        current_val = self.manual_labels[rel_path][str(frame_idx)].get(label, 0)
        new_val     = 1 if current_val == 0 else 0
        self.manual_labels[rel_path][str(frame_idx)][label] = new_val

        # Mutual exclusion: jika label diaktifkan, matikan pasangannya
        if new_val == 1 and label in MUTUAL_EXCLUSIVE:
            pair = MUTUAL_EXCLUSIVE[label]
            self.manual_labels[rel_path][str(frame_idx)][pair] = 0

        self._save_manual_labels()

        # Update checkbox
        self.left_panel.update_manual_checkboxes(
            frame_idx, self.manual_labels[rel_path][str(frame_idx)]
        )
        # Update highlight jika label yang diklik adalah label aktif
        active_lbl = self.active_frame_label.get()
        if label == active_lbl:
            self.left_panel.update_single_frame_highlight(frame_idx, label, new_val)
        fa  = self.frame_annotations.get(rel_path, {})
        mfr = dict(self.manual_labels[rel_path].get(str(frame_idx), {}))
        mfr["_rejected"] = fa.get(str(frame_idx), {}).get("_rejected", False)
        self.left_panel._current_frame_annotations[str(frame_idx)] = mfr
        # Gambar ulang dots untuk frame ini
        self.left_panel._draw_frame_dots(self.left_panel.frame_dot_canvases[frame_idx], mfr)

    # SigLIP inference

    def _apply_siglip_result(self, rel_path: str, res: dict):
        """
        Tulis hasil inferensi SigLIP ke frame_annotations dan batch_history.

        Frame yang di-reject oleh user di-zero-out dari voting sehingga
        tidak mempengaruhi prediksi akhir.
        """
        if rel_path not in self.frame_annotations:
            self.frame_annotations[rel_path] = {}

        # Catat frame yang di-reject SEBELUM ditimpa AI predictions
        n_res = res.get("n_frames", 2)
        rejected_set = {
            i for i in range(n_res)
            if self.frame_annotations.get(rel_path, {}).get(str(i), {}).get("_rejected", False)
        }

        for f_idx in range(n_res):
            if str(f_idx) not in self.frame_annotations[rel_path]:
                self.frame_annotations[rel_path][str(f_idx)] = {}
            for i, lbl in enumerate(LABELS):
                self.frame_annotations[rel_path][str(f_idx)][lbl] = \
                    res["per_label"][i]["frame_preds"][f_idx]
            # Enforce mutual exclusion per frame: jika keduanya 1, nol-kan yang lebih lemah
            for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                fa_frame = self.frame_annotations[rel_path][str(f_idx)]
                if fa_frame.get(lbl_a, 0) == 1 and fa_frame.get(lbl_b, 0) == 1:
                    idx_a = LABELS.index(lbl_a)
                    idx_b = LABELS.index(lbl_b)
                    score_a = res["per_label"][idx_a]["frame_scores"][f_idx]
                    score_b = res["per_label"][idx_b]["frame_scores"][f_idx]
                    if score_a >= score_b:
                        fa_frame[lbl_b] = 0
                    else:
                        fa_frame[lbl_a] = 0

        # Build batch_history dengan koreksi rejected frames
        per_label_history = {}
        for i in range(len(LABELS)):
            r           = res["per_label"][i]
            frame_preds = list(r["frame_preds"])

            for ri in rejected_set:
                if ri < len(frame_preds):
                    frame_preds[ri] = 0

            valid_preds  = [p for j, p in enumerate(frame_preds) if j not in rejected_set]
            valid_scores = [s for j, s in enumerate(r["frame_scores"]) if j not in rejected_set]
            n_valid      = len(valid_preds)
            vote_pos     = sum(valid_preds)
            valid_avg    = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0
            thr_i        = res["thresholds"][i]
            prediction   = 1 if (n_valid > 0 and valid_avg >= thr_i) else 0

            per_label_history[str(i)] = {
                "prediction":   prediction,
                "vote_pos":     vote_pos,
                "vote_neg":     n_valid - vote_pos,
                "skipped":      len(rejected_set),
                "avg_score":    valid_avg,
                "siglip_avg":   r.get("siglip_avg"),
                "landmark_avg": r.get("landmark_avg"),
                "threshold":    res["thresholds"][i],
                "frame_scores": list(r["frame_scores"]),
                "frame_preds":  frame_preds,
            }

        # Enforce mutual exclusion pada prediksi FINAL: jika kedua label dalam pasangan
        # eksklusif sama-sama 1, nol-kan prediction DAN frame_preds yang avg_score-nya lebih rendah.
        for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
            idx_a = str(LABELS.index(lbl_a))
            idx_b = str(LABELS.index(lbl_b))
            if per_label_history[idx_a]["prediction"] == 1 and per_label_history[idx_b]["prediction"] == 1:
                if per_label_history[idx_a]["avg_score"] >= per_label_history[idx_b]["avg_score"]:
                    loser = idx_b
                else:
                    loser = idx_a
                per_label_history[loser]["prediction"] = 0
                per_label_history[loser]["frame_preds"] = [0] * len(per_label_history[loser]["frame_preds"])

        # Auto-reject per FRAME: frame yang tidak punya label sama sekali (semua emosi < threshold)
        # ditolak otomatis. Frame yang punya minimal 1 label tetap dipertahankan.
        rejected_count = 0
        for f_idx in range(n_res):
            frame_has_label = any(
                per_label_history[str(i)]["frame_preds"][f_idx] == 1
                for i in range(len(LABELS))
            )
            if not frame_has_label:
                self.frame_annotations[rel_path][str(f_idx)]["_rejected"] = True
                rejected_count += 1
        if rejected_count > 0:
            print(f"  [NO_LABEL] {rel_path} → {rejected_count}/{n_res} frame ditolak (tidak ada label)")

        # no_label = True jika SEMUA frame tidak punya label
        no_label = rejected_count == n_res

        self.batch_history[rel_path] = {
            "per_label":  per_label_history,
            "thresholds": res["thresholds"],
            "no_label":   no_label,
        }
        with self.save_lock:
            save_batch_history(self.path_json_batch_history, self.batch_history)
            extra = self._batch_extra_path()
            if extra:
                save_batch_history(extra, self.batch_history)

        self._sync_manual_from_ai(rel_path)
        self.right_panel.update_statistics(self.batch_history)

    def reset_current_labels(self):
        """Reset semua label, rejected flag, dan riwayat AI untuk video aktif."""
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)

        # Timpa dengan entry bersih (tidak ada guard — selalu reset)
        self.frame_annotations[rel_path] = {
            str(i): {l: 0 for l in LABELS} for i in range(2)
        }
        self.batch_history.pop(rel_path, None)
        self.save_current_state()
        self._sync_manual_from_ai(rel_path)

        self.refresh_frame_gallery()  # fast path: cache masih valid
        self.right_panel.update_statistics(self.batch_history)
        for lbl in LABELS:
            self.right_panel.update_ai_score_bar(lbl, [0.0, 0.0])
        self.right_panel.lbl_batch_status.configure(
            text="Semua label direset ke 0", text_color="#10b981"
        )

    def _split_dataset_2d(self, silent: bool = False, force_default: bool = False, source: str = "ai"):
        """
        Split frame_annotations per UUID (siswa) → Label2d/train|val|test.csv.
        Hanya frame yang tidak di-reject yang masuk.
        Folder output: Label2d/ (nimpa) atau Label2d_{name}/ (buat baru).

        source="ai"     : gunakan frame_annotations (label AI).
        source="manual" : gunakan manual_labels (label semi-manual user).
        silent=True: skip messagebox, hanya print ke console.
        """
        import csv as _csv
        import random
        from collections import defaultdict

        if not self.frame_annotations:
            if not silent:
                messagebox.showinfo("Split", "Belum ada frame_annotations. Proses batch dulu.")
            return
        if not self.path_json_frames:
            if not silent:
                messagebox.showinfo("Split", "Buka folder dataset terlebih dahulu.")
            return

        # Pilih sumber frame_annotations:
        # Jika source="manual", gunakan manual_labels langsung (tanpa override batch dropdown).
        # Jika source="ai", cek dropdown batch; jika bukan default, derive dari batch itu.
        selected_batch = getattr(self.right_panel, "_batch_file_var", None)
        selected_batch = selected_batch.get() if selected_batch else "batch_history.json"
        default_batch  = os.path.basename(self.path_json_batch_history)

        if source == "manual":
            if not self.manual_labels:
                if not silent:
                    messagebox.showinfo("Split Manual", "Belum ada label manual. Aktifkan mode Label Semi Manual terlebih dahulu.")
                return
            # manual_labels hanya menyimpan label (0|1), tanpa _rejected.
            # Gabungkan dengan _rejected dari frame_annotations.
            fa_source = {}
            for rp, frames_m in self.manual_labels.items():
                fa_ai = self.frame_annotations.get(rp, {})
                vid_fa = {}
                for fi_str, lbl_dict in frames_m.items():
                    fd = dict(lbl_dict)
                    fd["_rejected"] = fa_ai.get(fi_str, {}).get("_rejected", False)
                    vid_fa[fi_str] = fd
                fa_source[rp] = vid_fa
            if not silent:
                print(f"[Split] Menggunakan label: MANUAL")
        elif not force_default and selected_batch and selected_batch != default_batch:
            # Load batch history yang dipilih dan derive frame_annotations dari frame_preds
            from utils import load_batch_history
            from ui.constants import LABELS as _LABELS
            bh_path   = os.path.join(os.path.dirname(self.path_json_batch_history), selected_batch)
            alt_batch = load_batch_history(bh_path)
            fa_source = {}
            for rp, entry in alt_batch.items():
                per_label = entry.get("per_label", {})
                n_fr = len(per_label.get("0", {}).get("frame_preds", []))
                existing = self.frame_annotations.get(rp, {})
                vid_fa = {}
                for fi in range(n_fr):
                    fd = dict(existing.get(str(fi), {}))
                    for li, lbl in enumerate(_LABELS):
                        preds = per_label.get(str(li), {}).get("frame_preds", [])
                        fd[lbl] = preds[fi] if fi < len(preds) else 0
                    vid_fa[str(fi)] = fd
                fa_source[rp] = vid_fa
            if not silent:
                print(f"[Split] Menggunakan batch: {selected_batch}")
        else:
            fa_source = self.frame_annotations

        # Tentukan output dir — nama folder split = nama batch
        create_new = self.right_panel.split_new_var.get()
        split_name = self.right_panel.batch_name_entry.get().strip()
        base_out   = os.path.dirname(self.path_json_frames)

        if create_new:
            if not split_name:
                import datetime
                split_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"{split_name}_manual" if source == "manual" else split_name
            out_dir = os.path.join(base_out, f"Label2d_{suffix}")
        else:
            out_dir = os.path.join(base_out, "Label2d_manual" if source == "manual" else "Label2d")

        # UUID depth dari entry
        try:
            uuid_depth = int(self.right_panel.split_uuid_depth_entry.get().strip())
        except (ValueError, AttributeError):
            uuid_depth = 2

        MAJORITY = 1
        N_FRAMES = 2

        # Grouping by UUID
        videos_by_uuid = defaultdict(list)
        skipped = 0
        no_label_count = 0
        for rel_path, frames in fa_source.items():
            if rel_path in self.flagged_data:
                skipped += 1
                continue

            # Skip video tanpa label (dari batch_history no_label flag)
            bh_entry = self.batch_history.get(rel_path, {})
            if bh_entry.get("no_label", False):
                no_label_count += 1
                skipped += 1
                continue

            parts = rel_path.replace("\\", "/").split("/")
            uuid  = parts[uuid_depth] if len(parts) > uuid_depth else parts[-2] if len(parts) > 1 else rel_path

            vote = {lbl: 0 for lbl in LABELS}
            valid = 0
            for idx_str, fdata in frames.items():
                if idx_str.startswith("_") or fdata.get("_rejected", False):
                    continue
                valid += 1
                for lbl in LABELS:
                    vote[lbl] += fdata.get(lbl, 0)

            if valid == 0:
                skipped += 1
                continue

            # Skip juga jika voting menghasilkan 0 di semua label
            all_zero = all(vote[lbl] < MAJORITY for lbl in LABELS)
            if all_zero:
                no_label_count += 1
                skipped += 1
                continue

            video_labels = {lbl: 1 if vote[lbl] >= MAJORITY else 0 for lbl in LABELS}
            # Mutual exclusion pada video_labels: jika keduanya 1, pertahankan yang lebih banyak vote
            for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                if video_labels[lbl_a] == 1 and video_labels[lbl_b] == 1:
                    if vote[lbl_a] >= vote[lbl_b]:
                        video_labels[lbl_b] = 0
                    else:
                        video_labels[lbl_a] = 0
            videos_by_uuid[uuid].append({
                "file_path": rel_path,
                "frames":    frames,
                "valid":     valid,
                **{lbl.lower(): str(video_labels[lbl]) for lbl in LABELS},
            })

        unique_uuids = list(videos_by_uuid.keys())
        if not unique_uuids:
            if not silent:
                messagebox.showwarning("Split", "Tidak ada video valid untuk di-split.")
            return

        random.seed(42)
        random.shuffle(unique_uuids)
        n       = len(unique_uuids)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        train_uuids = set(unique_uuids[:n_train])
        val_uuids   = set(unique_uuids[n_train:n_train + n_val])

        train_vids, val_vids, test_vids = [], [], []
        for uuid, vids in videos_by_uuid.items():
            if uuid in train_uuids:   train_vids.extend(vids)
            elif uuid in val_uuids:   val_vids.extend(vids)
            else:                     test_vids.extend(vids)

        os.makedirs(out_dir, exist_ok=True)

        def write_2d_csv(filename, video_list):
            path    = os.path.join(out_dir, filename)
            written = 0
            with open(path, "w", newline="") as fp:
                w = _csv.writer(fp)
                w.writerow(["frame_path"] + LABELS)
                for v in video_list:
                    rel  = v["file_path"]
                    base = os.path.splitext(rel.replace("\\", "/"))[0]
                    for i in range(N_FRAMES):
                        fdata = v["frames"].get(str(i), {})
                        if fdata.get("_rejected", False):
                            continue
                        frame_path = os.path.join(
                            "cropped_faces", "clean", base, f"frame_{i:02d}.jpg"
                        )
                        row = {lbl: fdata.get(lbl, int(v[lbl.lower()])) for lbl in LABELS}
                        for lbl_a, lbl_b in [("Boredom", "Engagement"), ("Confusion", "Frustration")]:
                            if row[lbl_a] == 1 and row[lbl_b] == 1:
                                row[lbl_b] = 0
                        w.writerow([frame_path] + [row[lbl] for lbl in LABELS])
                        written += 1
            return written

        counts = {}
        for split_name_s, vids in [("train", train_vids), ("val", val_vids), ("test", test_vids)]:
            counts[split_name_s] = write_2d_csv(f"{split_name_s}.csv", vids)

        total_frames = sum(counts.values())
        total_vids   = len(train_vids) + len(val_vids) + len(test_vids)

        src_label = "MANUAL" if source == "manual" else f"AI ({selected_batch})"
        summary = (
            f"Sumber: {src_label}\n"
            f"Output: {out_dir}\n\n"
            f"UUID/siswa: {n} → train={len(train_uuids)} / val={len(val_uuids)} / test={n - len(train_uuids) - len(val_uuids)}\n"
            f"Video: {total_vids} ({skipped} diskip, {no_label_count} tanpa label)\n"
            f"Frame: {total_frames} total\n"
            f"  train.csv : {counts['train']}\n"
            f"  val.csv   : {counts['val']}\n"
            f"  test.csv  : {counts['test']}"
        )
        if not silent:
            messagebox.showinfo("Split Label 2D Selesai", summary)
        print(f"[Split 2D] Selesai → {out_dir} | "
              f"train={counts['train']} val={counts['val']} test={counts['test']} frame")

    def _compare_ai_vs_manual(self):
        """
        Bandingkan label AI (frame_annotations) vs label Manual (manual_labels).
        Tampilkan jumlah video berbeda, jumlah frame berbeda, persentase, dan breakdown per label.
        """
        if not self.manual_labels:
            messagebox.showinfo("Perbandingan", "Belum ada label manual. Aktifkan mode Label Semi Manual terlebih dahulu.")
            return
        if not self.frame_annotations:
            messagebox.showinfo("Perbandingan", "Belum ada label AI. Proses batch terlebih dahulu.")
            return

        total_videos  = 0
        diff_videos   = 0
        total_frames  = 0
        diff_frames   = 0
        per_label_diff = {lbl: 0 for lbl in LABELS}
        per_label_total = {lbl: 0 for lbl in LABELS}

        common_videos = set(self.manual_labels.keys()) & set(self.frame_annotations.keys())
        total_videos  = len(common_videos)

        for rel in common_videos:
            man_vid = self.manual_labels[rel]
            ai_vid  = self.frame_annotations[rel]
            vid_has_diff = False
            for fi in range(2):
                fi_str = str(fi)
                man_fr = man_vid.get(fi_str, {})
                ai_fr  = ai_vid.get(fi_str, {})
                if ai_fr.get("_rejected", False):
                    continue
                total_frames += 1
                frame_has_diff = False
                for lbl in LABELS:
                    m_val = int(man_fr.get(lbl, 0))
                    a_val = int(ai_fr.get(lbl, 0))
                    per_label_total[lbl] += 1
                    if m_val != a_val:
                        per_label_diff[lbl] += 1
                        frame_has_diff = True
                if frame_has_diff:
                    diff_frames += 1
                    vid_has_diff = True
            if vid_has_diff:
                diff_videos += 1

        if total_videos == 0:
            messagebox.showinfo("Perbandingan", "Tidak ada video yang memiliki kedua label AI dan Manual.")
            return

        pct_vid   = 100 * diff_videos  / total_videos  if total_videos  else 0
        pct_frame = 100 * diff_frames  / total_frames  if total_frames  else 0

        lines = [
            f"Video dengan label berbeda : {diff_videos}/{total_videos} ({pct_vid:.1f}%)",
            f"Frame dengan label berbeda : {diff_frames}/{total_frames} ({pct_frame:.1f}%)",
            "",
            "Breakdown per label (frame):",
        ]
        for lbl in LABELS:
            tot = per_label_total[lbl]
            dif = per_label_diff[lbl]
            pct = 100 * dif / tot if tot else 0
            lines.append(f"  {lbl:<12}: {dif}/{tot} frame berbeda ({pct:.1f}%)")

        lines += [
            "",
            f"(Hanya video yang memiliki kedua label AI & Manual dihitung.)",
            f"Total video AI: {len(self.frame_annotations)} | Total video Manual: {len(self.manual_labels)}",
        ]

        self._show_compare_dialog(lines)

    def _show_compare_dialog(self, lines: list):
        """Tampilkan hasil perbandingan AI vs Manual dalam dialog CTkToplevel yang bersih."""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Perbandingan AI vs Manual")
        dialog.geometry("420x380")
        dialog.minsize(360, 300)
        dialog.lift()
        dialog.focus_force()
        dialog.after(100, dialog.grab_set)

        dialog.rowconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=0)
        dialog.columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))

        for line in lines:
            stripped = line.strip()
            if not stripped:
                ctk.CTkFrame(scroll, fg_color="transparent", height=6).pack()
                continue
            # Section header lines
            if stripped.endswith(":") or stripped.startswith("("):
                ctk.CTkLabel(
                    scroll, text=stripped,
                    font=("Poppins", 9, "bold"), text_color=("#4b5563", "#9ca3af"),
                    anchor="w",
                ).pack(fill="x", pady=(2, 0))
            elif stripped.startswith("Video") or stripped.startswith("Frame"):
                ctk.CTkLabel(
                    scroll, text=stripped,
                    font=("Poppins", 10, "bold"), text_color=("#1f2937", "#e5e7eb"),
                    anchor="w",
                ).pack(fill="x", pady=(0, 2))
            else:
                ctk.CTkLabel(
                    scroll, text=stripped,
                    font=("Poppins", 9), text_color=("#374151", "#9ca3af"),
                    anchor="w",
                ).pack(fill="x", pady=(0, 1))

        btn_bar = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 12))
        ctk.CTkButton(
            btn_bar, text="Tutup", command=dialog.destroy,
            fg_color="#3b82f6", hover_color="#2563eb",
            font=("Poppins", 10, "bold"), height=32, corner_radius=8, width=100,
        ).pack(side="right")

    def _proses_satu(self):
        """
        Jalankan inferensi SigLIP pada video yang sedang ditampilkan.

        Berjalan di background thread agar UI tidak freeze.
        Hasil ditulis ke frame_annotations dan batch_history, lalu UI diupdate.
        """
        if not self.video_files: return
        # GATE kalibrasi: video ini hanya boleh diproses bila ORANG-nya sudah punya baseline netral,
        # kecuali checkbox "pakai default" dicentang (pakai baseline populasi).
        if not self._calib_ok_single():
            messagebox.showwarning("Kalibrasi netral",
                "Orang di video ini BELUM ditandai frame netralnya.\n\n"
                "Tandai dulu via 'Tandai Frame Netral Orang Ini', ATAU centang "
                "'Pakai baseline default' untuk memproses dengan baseline populasi.")
            return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)
        prompts, ths = self.right_panel.get_prompts_and_thresholds()
        self.right_panel.lbl_batch_status.configure(text="Memproses…", text_color="#fbbf24")

        def worker():
            from utils.video import prepare_cropped_frames
            from core.inference import run_siglip_on_frames
            imgs, no_face_count, multi_face_count, landmark_results, viz_imgs = prepare_cropped_frames(
                vp, self.root_folder, self.path_dir_cropped,
                raw_cache_dir=self.path_dir_raw_cache, cfg=self.rules,
            )
            if not imgs:
                self.root.after(0, lambda: self.right_panel.lbl_batch_status.configure(
                    text="Gagal: tidak ada frame", text_color="#ef4444"
                ))
                return

            # Auto-flag jika frame kurang dari yang diharapkan (video terlalu pendek)
            if len(imgs) < 2:
                self.flagged_data.add(rel_path)
                save_flagged(self.path_csv_flagged, self.flagged_data)
                self.root.after(0, lambda nf=len(imgs): (
                    self.flag_var.set(True),
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Auto-flag: hanya {nf}/2 frame (video terlalu pendek)",
                        text_color="#ef4444"
                    ),
                    self._update_flag_count()
                ))
                return

            # Auto-flag jika >50% frame tidak terdeteksi wajah oleh MediaPipe
            if no_face_count == len(imgs):
                self.flagged_data.add(rel_path)
                save_flagged(self.path_csv_flagged, self.flagged_data)
                self.root.after(0, lambda: (
                    self.flag_var.set(True),
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Auto-flag: semua {len(imgs)} frame gagal deteksi wajah",
                        text_color="#ef4444"
                    ),
                    self._update_flag_count()
                ))
                return

            # Auto-flag jika terlalu banyak frame dengan >1 wajah
            mf_thr = int(os.getenv("MULTI_FACE_FRAMES_THRESHOLD", "1"))
            if multi_face_count > mf_thr:
                self.flagged_data.add(rel_path)
                save_flagged(self.path_csv_flagged, self.flagged_data)
                self.root.after(0, lambda mfc=multi_face_count: (
                    self.flag_var.set(True),
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Auto-flag: {mfc} frame berisi >1 wajah",
                        text_color="#ef4444"
                    ),
                    self._update_flag_count()
                ))
                return

            siglip_cache_path = self._siglip_cache_path(rel_path)
            res = run_siglip_on_frames(imgs, prompts, ths,
                                        landmark_results=landmark_results,
                                        cfg=self.rules,
                                        siglip_cache_path=siglip_cache_path,
                                        rel_path=rel_path)
            self._apply_siglip_result(rel_path, res)

            # ── DEBUG DUMP ────────────────────────────────────────────────────
            from core.landmark_analyzer import compute_emotion_scores as _ces
            _n   = res.get("n_frames", 2)
            _ths = res["thresholds"]
            _fa  = self.frame_annotations.get(rel_path, {})
            _lines = [
                "",
                "=" * 70,
                f"  VIDEO : {rel_path}",
                "=" * 70,
            ]
            for _fi in range(_n):
                _rej     = _fa.get(str(_fi), {}).get("_rejected", False)
                _rej_tag = "  ⚠ REJECTED" if _rej else ""
                _lines.append(f"\nFrame {_fi}{_rej_tag}")

                # ── Raw landmark ──
                _lr = landmark_results[_fi] if (landmark_results and _fi < len(landmark_results)) else None
                if _lr and _lr.face_found:
                    _lines.append(
                        f"  pose   yaw={_lr.yaw:+.1f}°  pitch={_lr.pitch:+.1f}°  roll={_lr.roll:+.1f}°"
                    )
                    _lines.append(
                        f"  iris   eye=(x={_lr.iris_x:+.3f} y={_lr.iris_y:+.3f})  "
                        f"img=(x={_lr.iris_img_x:+.3f} y={_lr.iris_img_y:+.3f})"
                    )
                    _lines.append(
                        f"  hand   1-hand={_lr.hand_one:.3f}  2-hands={_lr.hand_two:.3f}"
                    )
                    # ── AU dihitung dari blendshape MediaPipe (mediapipe-only, TANPA py-feat) ──
                    # Tampilkan SUMBER blendshape agar bisa diverifikasi: 'mediapipe' (bawaan
                    # FaceLandmarker) atau 'mp_blendshapes' (model py-feat, jika env BLENDSHAPE_SOURCE diset).
                    from core.blendshape_features import compute_action_units as _cau
                    try:
                        from core.landmark_analyzer import _BLENDSHAPE_SOURCE as _bsrc
                    except Exception:
                        _bsrc = "mediapipe"
                    _au_mp = _cau(_lr.blendshapes, self.rules,
                                  person_neutral=getattr(_lr, "person_neutral", None))
                    _au_str = "  ".join(
                        f"{_k}={_au_mp.get(_k, 0.0):.3f}"
                        for _k in ["AU1","AU2","AU4","AU7","AU12","AU14","AU25","AU26","AU43"]
                    )
                    _lines.append(f"  AU[blendshape={_bsrc}]  {_au_str}")

                    # ── Landmark emotion scores ──
                    _lsc = _ces(_lr, self.rules)
                    _lsc_str = "  ".join(
                        f"{LABELS[_ii][0]}={list(_lsc.values())[_ii] if isinstance(_lsc, dict) else _lsc[_ii]:.4f}"
                        for _ii in range(len(LABELS))
                    )
                    _lines.append(f"  land   {_lsc_str}")
                else:
                    _lines.append("  pose   [no face]")

                # ── SigLIP, hybrid per frame ──
                _sig_row  = "  ".join(
                    f"{LABELS[_ii][0]}={res['per_label'][_ii]['siglip_scores'][_fi]:.4f}"
                    if res['per_label'][_ii].get('siglip_scores') and _fi < len(res['per_label'][_ii]['siglip_scores'])
                    else f"{LABELS[_ii][0]}=—"
                    for _ii in range(len(LABELS))
                )
                _hyb_row  = "  ".join(
                    f"{LABELS[_ii][0]}={res['per_label'][_ii]['frame_scores'][_fi]:.4f}"
                    if _fi < len(res['per_label'][_ii]['frame_scores'])
                    else f"{LABELS[_ii][0]}=—"
                    for _ii in range(len(LABELS))
                )
                _pred_row = "  ".join(
                    f"{LABELS[_ii][0]}={'1✓' if res['per_label'][_ii]['frame_preds'][_fi] else '0✗'}"
                    if _fi < len(res['per_label'][_ii]['frame_preds'])
                    else f"{LABELS[_ii][0]}=—"
                    for _ii in range(len(LABELS))
                )
                _lines.append(f"  siglip {_sig_row}")
                _lines.append(f"  hybrid {_hyb_row}  (thr: {' '.join(f'{LABELS[_ii][0]}={_ths[_ii]:.2f}' for _ii in range(len(LABELS)))})")
                _lines.append(f"  pred   {_pred_row}")

            _lines.append("\nFINAL")
            for _ii, _lbl in enumerate(LABELS):
                _r    = res["per_label"][_ii]
                _ph   = self.batch_history.get(rel_path, {}).get("per_label", {}).get(str(_ii), {})
                _fin  = _ph.get("prediction", 0)
                _avg  = _r.get("avg_score", 0.0)
                _sig  = _r.get("siglip_avg", 0.0)
                _land = _r.get("landmark_avg")
                _ls   = f"  land_avg={_land:.4f}" if _land is not None else ""
                _mark = "✓" if _fin == 1 else "✗"
                _lines.append(
                    f"  {_mark} {_lbl:<12} siglip_avg={_sig:.4f}{_ls}  "
                    f"hybrid_avg={_avg:.4f}  thr={_ths[_ii]:.2f}  → {_fin}"
                )
            _lines += ["", "=" * 70, ""]
            print("\n".join(_lines))
            # ─────────────────────────────────────────────────────────────────

            def update_ui():
                for i, lbl in enumerate(LABELS):
                    r = res["per_label"][i]
                    self.right_panel.update_ai_score_bar(lbl, list(r["frame_scores"]), r.get("threshold"))
                self.save_current_state()
                thrs = [v.get() for v in self.right_panel.threshold_vars]
                update_batch_meta(self.path_json_batch_history, self.rules, thrs)
                # Invalidasi cache agar viz images terbaru & frame highlights terupdate
                self._gallery_cache["rel_path"] = None
                self.refresh_frame_gallery()
                self.right_panel.lbl_batch_status.configure(text="Selesai", text_color="#10b981")

            self.root.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    # ── Kalibrasi netral PER-ORANG (Bosch 2023 + FACS) ────────────────────────
    def _dataset_dir(self) -> str:
        """Folder dataset tempat person_neutrals.json disimpan (induk raw_cache)."""
        return os.path.dirname(self.path_dir_raw_cache) if self.path_dir_raw_cache else ""

    def _enumerate_persons(self) -> list:
        """List UUID orang unik (urut) dari video_files."""
        from utils.person_neutral import person_id_from_relpath
        seen = []
        for vp in self.video_files:
            rel = os.path.relpath(vp, self.root_folder)
            pid = person_id_from_relpath(rel)
            if pid and pid not in seen:
                seen.append(pid)
        return seen

    def _current_person_id(self) -> str | None:
        from utils.person_neutral import person_id_from_relpath
        if not self.video_files:
            return None
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        return person_id_from_relpath(rel)

    def _neutral_status(self) -> tuple:
        """(jumlah_ditandai, total_orang, indeks_orang_sekarang_1based)."""
        from utils.person_neutral import load_person_neutrals
        persons = self._enumerate_persons()
        marked  = load_person_neutrals(self._dataset_dir())
        n_marked = sum(1 for p in persons if p in marked)
        cur = self._current_person_id()
        cur_idx = (persons.index(cur) + 1) if (cur in persons) else 0
        return n_marked, len(persons), cur_idx

    def _goto_person(self, delta: int):
        """Loncat ke video PERTAMA orang berikutnya (+1) / sebelumnya (-1) untuk kalibrasi."""
        if not self.video_files:
            return
        from utils.person_neutral import person_id_from_relpath
        persons = self._enumerate_persons()
        if not persons:
            return
        cur = self._current_person_id()
        cur_idx = persons.index(cur) if cur in persons else 0
        target_idx = max(0, min(cur_idx + delta, len(persons) - 1))
        if target_idx == cur_idx and cur in persons:
            self.right_panel.lbl_batch_status.configure(
                text=("Sudah orang terakhir" if delta > 0 else "Sudah orang pertama"),
                text_color="#fbbf24")
            return
        target_person = persons[target_idx]
        # Kumpulkan semua video index milik target_person
        idxs = [i for i, vp in enumerate(self.video_files)
                if person_id_from_relpath(os.path.relpath(vp, self.root_folder)) == target_person]
        if not idxs:
            return
        # Kalau sudah ada video netral yang ditandai → loncat ke situ (biar marker kelihatan);
        # else ke video pertama orang itu.
        from utils.person_neutral import load_person_neutrals
        marked = load_person_neutrals(self._dataset_dir()).get(target_person, {})
        neutral_vid = marked.get("_video")
        target_i = idxs[0]
        if neutral_vid:
            for i in idxs:
                if os.path.relpath(self.video_files[i], self.root_folder) == neutral_vid:
                    target_i = i
                    break
        self.save_current_state()
        self.current_index = target_i
        self.load_video()

    def _next_person(self):
        self._goto_person(+1)

    def _prev_person(self):
        self._goto_person(-1)

    def _use_default_baseline(self) -> bool:
        """True jika checkbox 'pakai baseline default (populasi)' dicentang."""
        var = getattr(self.right_panel, "use_default_baseline_var", None)
        return bool(var.get()) if var is not None else False

    def _calib_ok_single(self) -> bool:
        """Boleh proses video saat ini? (orangnya sudah netral, ATAU pakai default)."""
        if self._use_default_baseline():
            return True
        if not self.video_files:
            return False
        from utils.person_neutral import get_person_neutral
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        return get_person_neutral(self._dataset_dir(), rel) is not None

    def _update_neutral_label(self):
        """Perbarui label status kalibrasi + state tombol proses (gate kalibrasi)."""
        if not hasattr(self.right_panel, "lbl_neutral_status"):
            return
        if not self.video_files:
            self.right_panel.lbl_neutral_status.configure(text="Kalibrasi: buka folder dulu")
            return
        from utils.person_neutral import get_person_neutral
        n_marked, total, cur_idx = self._neutral_status()
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        cur_done = get_person_neutral(self._dataset_dir(), rel) is not None
        tick = " ✓ sudah" if cur_done else " — BELUM"
        color = "#10b981" if n_marked == total and total > 0 else "#fbbf24"
        self.right_panel.lbl_neutral_status.configure(
            text=f"Orang ke-{cur_idx}/{total}{tick}  ·  {n_marked}/{total} netral ditandai",
            text_color=color,
        )
        # ── Gate tombol proses (jangan ganggu saat batch sedang jalan) ──
        if getattr(self, "batch_running", False):
            return
        use_def = self._use_default_baseline()
        # Proses Video Ini: aktif bila orang ini sudah netral ATAU pakai default
        ok_single = use_def or cur_done
        self.right_panel.btn_proses_satu.configure(state=("normal" if ok_single else "disabled"))
        # Batch Semua: aktif bila SEMUA orang netral ATAU pakai default
        ok_batch = use_def or (total > 0 and n_marked >= total)
        self.right_panel.btn_proses_semua.configure(state=("normal" if ok_batch else "disabled"))

    def _on_default_toggle(self):
        """Dipanggil saat checkbox 'pakai default' diubah → refresh gate tombol."""
        self._update_neutral_label()

    def _mark_neutral_current(self):
        """Tandai 1 frame video saat ini sebagai netral orang ini → person_neutrals.json."""
        from utils.person_neutral import set_person_neutral, person_id_from_relpath
        if not self.video_files:
            return
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        pid = person_id_from_relpath(rel)
        if not pid:
            messagebox.showwarning("Kalibrasi", "Tidak bisa kenali identitas orang dari path video ini.")
            return
        dd = self._dataset_dir()
        if not dd:
            messagebox.showwarning("Kalibrasi",
                "Folder dataset belum siap (path_dir_raw_cache kosong). Buka folder dataset dulu.")
            return
        lrs = self._gallery_cache.get("landmark_results") or []
        # Gunakan raw blendshape MediaPipe (bukan py-feat) → AU raw values sebagai neutral anchor.
        # _raw_blendshape_signals() mengembalikan nilai blendshape mentah yg dipetakan ke AU keys
        # (AU1, AU2, AU4, ...) — sama persis dengan `val` dalam normalisasi compute_blendshape_features.
        from core.blendshape_features import _raw_blendshape_signals as _rbs
        def _lr_to_au_raw(lr):
            raw = _rbs(getattr(lr, "blendshapes", {}) or {})
            return {k: v for k, v in raw.items() if not k.startswith("_")}

        valid = [(i, lr, _lr_to_au_raw(lr)) for i, lr in enumerate(lrs)
                 if getattr(lr, "face_found", False) and getattr(lr, "blendshapes", None)]
        print(f"[Kalibrasi] klik Tandai: {len(lrs)} frame di galeri, {len(valid)} valid (face+blendshape) | dataset={dd}")
        if not valid:
            messagebox.showwarning("Kalibrasi",
                "Wajah belum terdeteksi di frame ini.\n\n"
                "Pastikan video sudah dimuat dan wajah terlihat di kamera.")
            return

        def _save(au_values, frame_idx):
            set_person_neutral(dd, pid, au_values, meta={"_video": rel, "_frame": frame_idx})
            self._update_neutral_label()
            self.refresh_frame_gallery()   # gambar marker '★ NETRAL' di frame terpilih
            self.right_panel.lbl_batch_status.configure(
                text=f"✓ Netral disimpan (frame {frame_idx}) — {pid[:10]}…", text_color="#10b981")

        if len(valid) == 1:
            _save(valid[0][2], valid[0][0])
            return

        # >1 frame valid → biar annotator pilih yang paling netral
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Pilih frame paling NETRAL")
        dlg.geometry("520x230")
        dlg.transient(self.root)
        # grab_set HARUS setelah window viewable (CTkToplevel render tertunda) → defer.
        def _grab():
            try:
                dlg.grab_set()
            except Exception:
                pass
        dlg.after(50, _grab)
        ctk.CTkLabel(dlg, text=f"Orang: {pid[:24]}…  — pilih frame wajah paling RILEKS/netral",
                     font=self.app_font_or_default()).pack(pady=(12, 6))
        row = ctk.CTkFrame(dlg, fg_color="transparent"); row.pack(fill="both", expand=True, padx=12)
        for i, lr, au in valid:
            summary = (f"Frame {i}\nAU4(alis↓)={au.get('AU4',0):.3f}  AU1(alis↑)={au.get('AU1',0):.3f}\n"
                       f"AU7={au.get('AU7',0):.3f}  AU43(mata)={au.get('AU43',0):.3f}")
            cell = ctk.CTkFrame(row); cell.pack(side="left", fill="both", expand=True, padx=6, pady=6)
            ctk.CTkLabel(cell, text=summary, justify="left", font=("Poppins", 11)).pack(pady=(8, 4))
            ctk.CTkButton(cell, text=f"Pilih frame {i} ini",
                          command=lambda a=au, fi=i: (_save(a, fi), dlg.destroy()),
                          fg_color="#10b981", hover_color="#059669").pack(pady=(4, 8), padx=8, fill="x")

    def app_font_or_default(self):
        return getattr(self, "font_sm", ("Poppins", 11))

    def _toggle_batch(self):
        """
        Toggle antara memulai dan menghentikan proses batch.

        Jika batch sedang berjalan, set flag cancel_batch dan tunggu worker selesai sendiri.
        Jika tidak berjalan, mulai batch baru via _proses_semua().
        """
        if self.batch_running:
            self.cancel_batch = True
            self.right_panel.btn_proses_semua.configure(
                text="Menghentikan…", fg_color="#6b7280", hover_color="#4b5563"
            )
        else:
            self._proses_semua()

    def _proses_semua(self):
        """
        Jalankan inferensi SigLIP pada semua video dalam dataset secara berurutan.

        Berjalan di background thread. Video yang sudah ada di batch_history di-skip.
        Batch dapat dihentikan sewaktu-waktu via _toggle_batch().
        Hasil tiap video langsung disimpan ke disk.
        """
        if not self.video_files: return

        # GATE: SEMUA orang harus punya baseline netral dulu (Bosch 2023 per-person calibration),
        # KECUALI checkbox "pakai default" dicentang. Tanpa centang → BLOK (tidak bisa batch).
        n_marked, total, _ = self._neutral_status()
        use_default = self._use_default_baseline()
        if total > 0 and n_marked < total and not use_default:
            messagebox.showwarning(
                "Batch terkunci — kalibrasi belum lengkap",
                f"Baru {n_marked}/{total} orang yang ditandai frame netralnya.\n\n"
                f"Selesaikan kalibrasi semua orang dulu, ATAU centang "
                f"'Pakai baseline default' untuk batch dengan baseline populasi.",
            )
            return

        self.batch_running = True
        self.cancel_batch  = False
        # Matikan debug log saat batch — print per-frame sangat memperlambat
        import core.landmark_analyzer as _lm
        _lm._DBG_LAND = False
        self.right_panel.btn_proses_semua.configure(
            text="Hentikan", fg_color="#ef4444", hover_color="#dc2626"
        )
        self.right_panel.btn_proses_satu.configure(state="disabled")

        prompts, ths = self.right_panel.get_prompts_and_thresholds()
        def worker():
            import queue as _q, time as _time
            from utils.video import prepare_cropped_frames
            from core.inference import run_siglip_batch
            total             = len(self.video_files)
            already_done      = set(self.batch_history.keys())
            preprocess_workers = int(os.getenv("PREPROCESS_WORKERS", "4"))
            siglip_batch_size  = int(os.getenv("SIGLIP_BATCH_VIDEOS", "8"))
            mf_thr             = int(os.getenv("MULTI_FACE_FRAMES_THRESHOLD", "1"))

            # Filter video yang perlu diproses
            todo = [(idx, vp) for idx, vp in enumerate(self.video_files)
                    if os.path.relpath(vp, self.root_folder) not in already_done]
            n_todo = len(todo)

            batch_start_time = _time.monotonic()
            n_done_so_far    = [0]  # mutable untuk closure

            def _fmt_eta(done, total_left, elapsed):
                if done == 0 or elapsed < 2:
                    return ""
                rate = done / elapsed          # vid/s
                remaining = (total_left - done) / rate
                if remaining < 60:
                    return f" · sisa ~{int(remaining)}d"
                elif remaining < 3600:
                    return f" · sisa ~{int(remaining/60)}m"
                else:
                    h = int(remaining / 3600)
                    m = int((remaining % 3600) / 60)
                    return f" · sisa ~{h}j{m:02d}m"

            if n_todo == 0:
                return

            # Update status skip untuk video yang sudah selesai
            skip_count = total - n_todo
            if skip_count > 0:
                def upd_skip_count(sc=skip_count):
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Skip {sc} video (sudah ada), proses {n_todo} sisanya…",
                        text_color="#6b7280",
                    )
                self.root.after(0, upd_skip_count)

            # Queue dibatasi agar CPU workers tidak jauh mendahului GPU.
            # Maxsize = preprocess_workers * 2 + siglip_batch_size cukup untuk menjaga
            # GPU selalu ada pekerjaan tanpa RAM meledak dari PIL images yang menumpuk.
            # Saat cancel: GPU consumer break lebih dulu, CPU workers yang masih
            # berjalan akan timeout di put() dan kembali — tidak deadlock.
            _queue_max = preprocess_workers * 2 + siglip_batch_size
            prep_queue = _q.Queue(maxsize=_queue_max)

            def cpu_preprocess(idx, vp):
                """CPU worker: ekstrak frame + MediaPipe, taruh di queue."""
                if self.cancel_batch:
                    # GPU consumer sudah break — tidak perlu put apapun.
                    return
                rel_path = os.path.relpath(vp, self.root_folder)
                def upd_status(i=idx, r=rel_path):
                    done    = n_done_so_far[0]
                    elapsed = _time.monotonic() - batch_start_time
                    eta     = _fmt_eta(done, n_todo, elapsed)
                    self.right_panel.lbl_batch_status.configure(
                        text=f"{done}/{n_todo}{eta}  ·  {os.path.basename(r)}",
                        text_color="#fbbf24",
                    )
                self.root.after(0, upd_status)

                try:
                    imgs, no_face_count, multi_face_count, landmark_results, _ = prepare_cropped_frames(
                        vp, self.root_folder, self.path_dir_cropped,
                        raw_cache_dir=self.path_dir_raw_cache, cfg=self.rules,
                    )
                    if not imgs:
                        try:
                            prep_queue.put({"type": "skip"}, timeout=5.0)
                        except _q.Full:
                            pass
                        return

                    flag_reason = None
                    if len(imgs) < 2:
                        flag_reason = f"hanya {len(imgs)}/2 frame"
                    elif no_face_count == len(imgs):
                        flag_reason = f"semua {len(imgs)} frame gagal deteksi wajah"
                    elif multi_face_count > mf_thr:
                        flag_reason = f"{multi_face_count} frame multi-wajah"

                    if flag_reason:
                        with self.save_lock:
                            self.flagged_data.add(rel_path)
                            save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_flag(r=rel_path, reason=flag_reason):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} ({reason})",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_flag)
                        try:
                            prep_queue.put({"type": "flagged"}, timeout=5.0)
                        except _q.Full:
                            pass
                        return

                    # Debug ringkas (mediapipe-only): sumber blendshape + pose + AU43 per frame
                    from core.blendshape_features import compute_action_units as _cau2
                    try:
                        from core.landmark_analyzer import _BLENDSHAPE_SOURCE as _bsrc2
                    except Exception:
                        _bsrc2 = "mediapipe"
                    _frame_parts = []
                    for _i, _lr in enumerate(landmark_results):
                        if not _lr.face_found:
                            _frame_parts.append(f"[{_i}] no_face")
                        else:
                            _au43 = _cau2(_lr.blendshapes, self.rules,
                                          person_neutral=getattr(_lr, "person_neutral", None)).get("AU43", 0.0)
                            _frame_parts.append(
                                f"[{_i}] yaw={_lr.yaw:+.1f}° pitch={_lr.pitch:+.1f}° AU43={_au43:.3f}"
                            )
                    print(f"[Batch] {os.path.basename(rel_path):<38} blendshape={_bsrc2}  "
                          + "  |  ".join(_frame_parts))

                    item_ready = {
                        "type":             "ready",
                        "idx":              idx,
                        "rel_path":         rel_path,
                        "pil_images":       imgs,
                        "landmark_results": landmark_results,
                        "siglip_cache_path": self._siglip_cache_path(rel_path),
                    }
                    # Loop put dengan timeout pendek agar cancel langsung responsif
                    # (tidak perlu nunggu 10 detik bila GPU consumer sudah keluar)
                    while not self.cancel_batch:
                        try:
                            prep_queue.put(item_ready, timeout=0.5)
                            break
                        except _q.Full:
                            continue  # coba lagi setelah 0.5d, cek cancel di atas
                except Exception as e:
                    print(f"[Error] MediaPipe failed for {os.path.relpath(vp, self.root_folder)}: {e}")
                    try:
                        prep_queue.put({"type": "error"}, timeout=5.0)
                    except _q.Full:
                        pass

            # Submit CPU workers — tidak pakai context manager agar bisa shutdown(cancel_futures=True)
            cpu_exec = concurrent.futures.ThreadPoolExecutor(max_workers=preprocess_workers)
            cpu_futures = [cpu_exec.submit(cpu_preprocess, idx, vp) for idx, vp in todo]

            # GPU batching loop — berjalan di worker thread ini sambil CPU bekerja
            gpu_batch    = []   # list of "ready" items
            received     = 0    # item masuk ke queue (semua tipe)
            n_processed  = 0    # video berhasil di-SigLIP

            try:
                while received < n_todo:
                    if self.cancel_batch:
                        break
                    try:
                        item = prep_queue.get(timeout=1.0)
                    except _q.Empty:
                        if all(f.done() for f in cpu_futures):
                            break
                        continue

                    received += 1

                    if self.cancel_batch:
                        break

                    if item["type"] == "ready":
                        gpu_batch.append(item)

                    # Flush batch ketika penuh ATAU ini item terakhir
                    if gpu_batch and (len(gpu_batch) >= siglip_batch_size or received == n_todo):
                        batch_input = [
                            {
                                "pil_images":        b["pil_images"],
                                "landmark_results":  b["landmark_results"],
                                "rel_path":          b["rel_path"],
                                "siglip_cache_path": b["siglip_cache_path"],
                            }
                            for b in gpu_batch
                        ]
                        try:
                            batch_results = run_siglip_batch(batch_input, prompts, ths, cfg=self.rules)
                        except Exception as e:
                            print(f"[Error] SigLIP batch failed: {e}")
                            batch_results = []

                        current_rel = os.path.relpath(
                            self.video_files[self.current_index], self.root_folder
                        )
                        with self.save_lock:
                            for b_item, res in zip(gpu_batch, batch_results):
                                self._apply_siglip_result(b_item["rel_path"], res)
                                n_processed += 1
                                n_done_so_far[0] += 1
                                _done = n_done_so_far[0]
                                _elapsed = _time.monotonic() - batch_start_time
                                _eta = _fmt_eta(_done, n_todo, _elapsed)
                                def _upd_prog(d=_done, e=_eta):
                                    self.right_panel.lbl_batch_status.configure(
                                        text=f"{d}/{n_todo}{e}",
                                        text_color="#10b981",
                                    )
                                self.root.after(0, _upd_prog)
                                if b_item["rel_path"] == current_rel:
                                    for i, lbl in enumerate(LABELS):
                                        fsc = list(res["per_label"][i]["frame_scores"])
                                        thr_i = res["per_label"][i].get("threshold")
                                        self.root.after(0, lambda l=lbl, s=fsc, t=thr_i:
                                            self.right_panel.update_ai_score_bar(l, s, t))
                                    self.root.after(0, self.refresh_frame_gallery)

                        # Periodic save + stats refresh setiap ~50 video
                        if n_processed % 50 < siglip_batch_size:
                            with self.save_lock:
                                save_frame_annotations(self.path_json_frames, self.frame_annotations)
                                save_batch_history(self.path_json_batch_history, self.batch_history)
                            _bh_snap = dict(self.batch_history)
                            self.root.after(0, lambda bh=_bh_snap:
                                self.right_panel.update_statistics(bh))

                        gpu_batch = []
            finally:
                # Selalu shutdown executor — cancel_futures=True agar futures yang
                # belum mulai tidak dieksekusi, futures yang sedang jalan selesai sendiri
                cpu_exec.shutdown(wait=True, cancel_futures=True)

            # Simpan semua data sekaligus setelah batch selesai (bukan per-video)
            with self.save_lock:
                save_frame_annotations(self.path_json_frames, self.frame_annotations)
                save_batch_history(self.path_json_batch_history, self.batch_history)

            def on_finish():
                self.batch_running = False
                import core.landmark_analyzer as _lm
                _lm._DBG_LAND = True
                elapsed_total = _time.monotonic() - batch_start_time
                if elapsed_total < 60:
                    elapsed_str = f"{int(elapsed_total)}d"
                elif elapsed_total < 3600:
                    elapsed_str = f"{int(elapsed_total/60)}m{int(elapsed_total%60):02d}d"
                else:
                    h = int(elapsed_total / 3600)
                    m = int((elapsed_total % 3600) / 60)
                    elapsed_str = f"{h}j{m:02d}m"
                done_final = n_done_so_far[0]
                if self.cancel_batch:
                    msg = f"Dihentikan  {done_final}/{n_todo}  ({elapsed_str})"
                    color = "#ef4444"
                else:
                    msg = f"Selesai  {done_final}/{n_todo}  ({elapsed_str})"
                    color = "#10b981"
                self.right_panel.lbl_batch_status.configure(text=msg, text_color=color)
                self.right_panel.btn_proses_semua.configure(
                    text="Batch Semua", fg_color="#10b981", hover_color="#059669"
                )
                thrs = [v.get() for v in self.right_panel.threshold_vars]
                update_batch_meta(self.path_json_batch_history, self.rules, thrs)
                # Sync manual_labels untuk video-video baru yang baru diproses batch
                # (non-destructive: tidak timpa entri yang sudah ada / manual edits)
                self._sync_manual_missing_from_ai()
                self.right_panel.update_statistics(self.batch_history)
                self.right_panel.update_manual_statistics(self.manual_labels)
                # Pulihkan state tombol sesuai gate kalibrasi (bukan paksa normal)
                self._update_neutral_label()

            self.root.after(0, on_finish)

        threading.Thread(target=worker, daemon=True).start()

def _start_siglip_preload_bg():
    """Preload SigLIP model in background thread — starts ~800ms after window shows."""
    def _load():
        try:
            from core.siglip_model import preload_siglip
            preload_siglip()
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True, name="siglip-early-preload").start()


if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoLabelerApp(root)
    # Start SigLIP preload shortly after window renders — heavy import happens
    # in background so UI stays responsive; model ready before user runs inference.
    root.after(800, _start_siglip_preload_bg)

    def _on_close():
        # Tandai shutdown agar worker daemon (galeri/batch) berhenti rapi & tidak
        # melempar 'cannot schedule new futures after interpreter shutdown'.
        app._shutting_down = True
        app.cancel_batch = True
        app._lp_cancel_flag = True
        try:
            app._lp_kill_worker()          # matikan worker LP persisten (jangan jadi proses yatim)
        except Exception:
            pass
        try:
            from core.pyfeat_client import get_pyfeat_client
            get_pyfeat_client().shutdown()
        except Exception:
            pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()