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
import cv2
from dotenv import load_dotenv

load_dotenv()

from ui import LABELS, LABEL_COLORS, LeftPanel, RightPanel, RulesPanel

# Pasangan label yang saling eksklusif: jika satu = 1, pasangannya harus 0
MUTUAL_EXCLUSIVE = {
    "Confusion": "Frustration",
    "Frustration": "Confusion",
    "Boredom": "Engagement",
    "Engagement": "Boredom",
}
from utils import (
    prepare_cropped_frames,
    load_annotations, save_annotations,
    load_flagged, save_flagged,
    load_frame_annotations, save_frame_annotations,
    load_batch_history, save_batch_history,
    load_batch_meta, update_batch_meta,
    load_skipped, save_skipped,
    load_thresholds, save_thresholds,
)
from core import run_siglip_on_frames, load_rules, save_rules, DEFAULT_RULES, recalculate_batch

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VideoLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Labeler Emosi SigLIP2")
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
        self.active_frame_label = ctk.StringVar(value="Boredom")
        self.show_viz   = ctk.BooleanVar(value=False)   # Toggle landmark viz di galeri
        self.viz_images = []   # Cache viz PIL images untuk video aktif

        self.batch_running, self.cancel_batch = False, False
        self._viz_regen_requested = False
        self.save_lock = threading.RLock()

        # Gallery background-loading state
        self._gallery_version = 0
        self._gallery_cache: dict = {
            "rel_path": None, "pil_images": [], "viz_images": [],
            "landmark_results": [], "no_face_count": 0, "multi_fc": 0,
        }

        self._build_ui()

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
        self.root.bind("<Right>", lambda e: self.save_and_next())
        self.root.bind("<Left>",  lambda e: self.prev_video())

    def _build_topbar(self):
        """Bangun bar atas: tombol buka folder, label nama video, dan progress counter."""
        bar = ctk.CTkFrame(self.root, fg_color=("#e8eaef", "#181826"), corner_radius=0, height=50)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Buka Folder", command=self.open_folder,
            font=self.font_bold, fg_color="#10b981", hover_color="#059669",
            width=110, height=32, corner_radius=8,
        ).pack(side="left", padx=(14, 4), pady=9)

        ctk.CTkButton(
            bar, text="Output", command=self._change_output_folder,
            font=self.font_sm, fg_color="#6366f1", hover_color="#4f46e5",
            width=80, height=32, corner_radius=8,
        ).pack(side="left", padx=(0, 4), pady=9)

        ctk.CTkButton(
            bar, text="Rules", command=self._open_rules_panel,
            font=self.font_sm, fg_color="#7c3aed", hover_color="#6d28d9",
            width=70, height=32, corner_radius=8,
        ).pack(side="left", padx=(0, 14), pady=9)

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
        bar = ctk.CTkFrame(self.root, fg_color=("#e8eaef", "#181826"), corner_radius=0, height=50)
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
        base = os.path.join(folder, output_name)

        self.path_csv_annotations    = os.path.join(base, "annotations_bener.csv")
        self.path_csv_flagged        = os.path.join(base, "flagged_videos.csv")
        self.path_json_frames        = os.path.join(base, "frame_annotations.json")
        self.path_json_batch_history = os.path.join(base, "batch_history.json")
        self.path_json_skipped       = os.path.join(base, "skipped_videos.json")
        self.path_json_thresholds    = os.path.join(base, "thresholds.json")
        self.path_json_rules         = os.path.join(base, "rules.json")
        # path_json_manual dihapus — pakai _manual_path_for() secara dinamis
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

    def _load_data(self):
        """Muat semua file persistensi (CSV + JSON) ke memory state aplikasi."""
        self.flagged_data      = load_flagged(self.path_csv_flagged)
        self.frame_annotations = load_frame_annotations(self.path_json_frames)
        self.batch_history     = load_batch_history(self.path_json_batch_history)
        self.skipped_videos    = load_skipped(self.path_json_skipped)
        self.manual_labels     = self._load_manual_labels()
        self._load_saved_thresholds()
        self._load_rules()
        self._update_flag_count()
        self.right_panel.update_statistics(self.batch_history)
        # Refresh dropdown daftar batch file yang tersedia
        self.right_panel.refresh_batch_files(os.path.dirname(self.path_json_batch_history))

    def _load_saved_thresholds(self):
        """Muat threshold tersimpan dan terapkan ke slider UI."""
        if not self.path_json_thresholds:
            return
        saved = load_thresholds(self.path_json_thresholds, LABELS)
        if saved:
            for i, val in enumerate(saved):
                self.right_panel.threshold_vars[i].set(val)
                if i < len(self.right_panel.threshold_labels):
                    entry = self.right_panel.threshold_labels[i]
                    entry.delete(0, "end")
                    entry.insert(0, f"{float(val):.2f}")
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
                    return _json.load(f)
            except Exception:
                pass
        return {}

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

    def _sync_manual_from_ai(self, rel_path: str, save: bool = True):
        """Reset manual_labels untuk satu video dari frame_annotations (label AI terbaru).
        Dipanggil setiap kali AI selesai proses atau label direset.
        save=False: hanya update dict di memori, tidak tulis ke disk (untuk bulk sync).
        """
        fa = self.frame_annotations.get(rel_path, {})
        self.manual_labels[rel_path] = {
            str(fi): {lbl: int(fa.get(str(fi), {}).get(lbl, 0)) for lbl in LABELS}
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

    def _save_rules(self, rules: dict):
        """Simpan rules ke disk dan update self.rules."""
        self.rules = rules
        if self.path_json_rules:
            save_rules(self.path_json_rules, rules)
        print(f"[Rules] Tersimpan ke {self.path_json_rules}")

    def _open_rules_panel(self):
        """Buka jendela Rules Editor."""
        if hasattr(self, "_rules_panel") and self._rules_panel is not None:
            try:
                if self._rules_panel.win.winfo_exists():
                    self._rules_panel.win.lift()
                    self._rules_panel.win.focus()
                    return
            except Exception:
                pass
        self._rules_panel = RulesPanel(
            self.root,
            rules=self.rules,
            threshold_vars=self.right_panel.threshold_vars,
            on_save=self._save_rules,
            on_recalculate=self._recalculate_all,
            on_rebatch=self._toggle_batch,
        )

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
                    # Sync manual_labels dari semua frame_annotations baru hasil recalculate.
                    # save=False: hanya update dict di memori, simpan sekali di akhir
                    # (sebelumnya save per-video = ribuan disk write = lag panjang).
                    for rel in self.frame_annotations:
                        self._sync_manual_from_ai(rel, save=False)
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
        Hapus batch_history sehingga Batch AI bisa dijalankan ulang dari awal.
        Tidak menghapus anotasi manual — hanya riwayat AI.
        """
        if not self.path_json_batch_history:
            return
        if not messagebox.askyesno(
            "Restart Batch",
            "Hapus riwayat batch AI? Semua video akan diproses ulang.\n"
            "Anotasi manual TIDAK terhapus."
        ):
            return
        self.batch_history = {}
        save_batch_history(self.path_json_batch_history, self.batch_history)
        self.right_panel.lbl_batch_status.configure(
            text="Batch history direset", text_color="#10b981"
        )
        print("[Batch] History direset — semua video akan diproses ulang.")

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
        if not self.cap: return
        self.current_frame    = int(float(val))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.play_start_time  = time.time()
        self.play_start_frame = self.current_frame

    def seek_to_frame(self, frame_idx: int):
        """
        Seek video ke frame yang mewakili posisi thumbnail ke-frame_idx (0-15).

        Args:
            frame_idx: Index thumbnail (0-15), bukan frame absolut.
        """
        if not self.cap or not self.video_files: return
        target = int(self.total_frames * frame_idx / 4)
        self.current_frame = self.play_start_frame = target
        self.play_start_time = time.time()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.left_panel.slider.set(target)
        ret, frame = self.cap.read()
        if ret:
            self.left_panel.show_video_frame(frame)

    # Navigation

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
        if idx == self.current_index: return
        self.save_current_state()
        self.current_index = idx
        self.jump_entry.delete(0, "end")
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
                # Gabungkan: ambil _rejected dari frame_annotations, label dari manual_labels
                fa = self.frame_annotations.get(rel_path, {})
                merged = {}
                for fi in range(2):
                    fi_str = str(fi)
                    m = dict(ml.get(fi_str, {}))
                    m["_rejected"] = fa.get(fi_str, {}).get("_rejected", False)
                    merged[fi_str] = m
                return merged
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
            result = prepare_cropped_frames(
                vp, self.root_folder, self.path_dir_cropped,
                raw_cache_dir=self.path_dir_raw_cache if hasattr(self, "path_dir_raw_cache") else None,
                cfg=self.rules if hasattr(self, "rules") else None,
            )
            if self._gallery_version != my_version:
                return  # hasil sudah kedaluwarsa — buang
            self.root.after(0, lambda: self._apply_gallery_result(rel_path, result, my_version))

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
        return sum(
            1 for i in range(n_frames)
            if self.frame_annotations.get(rel_path, {}).get(str(i), {}).get("_rejected", False)
        )

    def toggle_frame_reject(self, frame_idx: int):
        """
        Double-klik frame → toggle status rejected.

        Frame rejected ditampilkan dengan overlay merah dan tidak dihitung
        dalam voting prediksi AI saat proses inferensi.
        """
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)

        if rel_path not in self.frame_annotations:
            self.frame_annotations[rel_path] = {
                str(i): {l: 0 for l in LABELS} for i in range(2)
            }
        if str(frame_idx) not in self.frame_annotations[rel_path]:
            self.frame_annotations[rel_path][str(frame_idx)] = {l: 0 for l in LABELS}

        current = self.frame_annotations[rel_path][str(frame_idx)].get("_rejected", False)
        self.frame_annotations[rel_path][str(frame_idx)]["_rejected"] = not current

        self.refresh_frame_gallery()  # fast path — cache masih valid
        self.save_current_state()

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
                        w.writerow([
                            frame_path,
                            fdata.get("Boredom",     v["boredom"]),
                            fdata.get("Engagement",  v["engagement"]),
                            fdata.get("Confusion",   v["confusion"]),
                            fdata.get("Frustration", v["frustration"]),
                        ])
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

        messagebox.showinfo("Perbandingan AI vs Manual", "\n".join(lines))

    def _proses_satu(self):
        """
        Jalankan inferensi SigLIP pada video yang sedang ditampilkan.

        Berjalan di background thread agar UI tidak freeze.
        Hasil ditulis ke frame_annotations dan batch_history, lalu UI diupdate.
        """
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)
        prompts, ths = self.right_panel.get_prompts_and_thresholds()
        self.right_panel.lbl_batch_status.configure(text="Memproses…", text_color="#fbbf24")

        def worker():
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
            _KEY_BS = [
                "jawOpen", "eyeBlinkLeft", "eyeBlinkRight",
                "eyeWideLeft", "eyeWideRight",
                "eyeSquintLeft", "eyeSquintRight",
                "browInnerUp", "browDownLeft", "browDownRight",
                "mouthSmileLeft", "mouthSmileRight",
                "mouthUpperUpLeft", "mouthUpperUpRight",
                "noseSneerLeft", "noseSneerRight",
                "cheekSquintLeft", "cheekSquintRight",
            ]
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
                        f"  hand   forehead={_lr.hand_forehead:.3f}  chin={_lr.hand_chin:.3f}"
                    )
                    # blendshapes dalam 2 kolom
                    _bs = _lr.blendshapes
                    _bs_vals = [f"{_k.replace('Left','L').replace('Right','R'):<20}={_bs.get(_k,0):.3f}"
                                for _k in _KEY_BS]
                    for _bi in range(0, len(_bs_vals), 2):
                        _lines.append("  bs     " + "  ".join(_bs_vals[_bi:_bi+2]))

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
        self.batch_running = True
        self.cancel_batch  = False
        # Matikan debug log saat batch — print per-frame sangat memperlambat
        import core.landmark_analyzer as _lm
        _lm._DBG_LAND = False
        self.right_panel.btn_proses_semua.configure(
            text="Hentikan Batch", fg_color="#ef4444", hover_color="#dc2626"
        )
        self.right_panel.btn_proses_satu.configure(state="disabled")

        prompts, ths = self.right_panel.get_prompts_and_thresholds()
        def worker():
            total         = len(self.video_files)
            already_done  = set(self.batch_history.keys())
            max_workers   = int(os.getenv("MAX_WORKERS", "2"))

            def process_video(idx, vp):
                if self.cancel_batch: return
                rel_path = os.path.relpath(vp, self.root_folder)

                if rel_path in already_done:
                    def upd_skip(i=idx, r=rel_path):
                        self.right_panel.lbl_batch_status.configure(
                            text=f"Skip (sudah ada) {i+1}/{total}: {os.path.basename(r)}",
                            text_color="#6b7280",
                        )
                    self.root.after(0, upd_skip)
                    return

                def upd_status(i=idx, r=rel_path):
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Batch {i+1}/{total}: {os.path.basename(r)}",
                        text_color="#fbbf24",
                    )
                self.root.after(0, upd_status)

                try:
                    imgs, no_face_count, multi_face_count, landmark_results, _ = prepare_cropped_frames(
                        vp, self.root_folder, self.path_dir_cropped,
                        raw_cache_dir=self.path_dir_raw_cache, cfg=self.rules,
                    )
                    if not imgs: return

                    if len(imgs) < 2:
                        with self.save_lock:
                            self.flagged_data.add(rel_path)
                            save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_shortflag(r=rel_path, nf=len(imgs)):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} (hanya {nf}/2 frame)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_shortflag)
                        return

                    if no_face_count == len(imgs):
                        with self.save_lock:
                            self.flagged_data.add(rel_path)
                            save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_autoflag(r=rel_path, nf=no_face_count, ni=len(imgs)):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} (semua {ni} frame gagal)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_autoflag)
                        return

                    mf_thr = int(os.getenv("MULTI_FACE_FRAMES_THRESHOLD", "1"))
                    if multi_face_count > mf_thr:
                        with self.save_lock:
                            self.flagged_data.add(rel_path)
                            save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_multiface(r=rel_path, mfc=multi_face_count):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} ({mfc} frame multi-wajah)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_multiface)
                        return

                    siglip_cache_path = self._siglip_cache_path(rel_path)
                    res = run_siglip_on_frames(imgs, prompts, ths,
                                               landmark_results=landmark_results,
                                               cfg=self.rules,
                                               siglip_cache_path=siglip_cache_path,
                                               rel_path=rel_path)

                    with self.save_lock:
                        self._apply_siglip_result(rel_path, res)

                    current_rel = os.path.relpath(
                        self.video_files[self.current_index], self.root_folder
                    )
                    if rel_path == current_rel:
                        for i, lbl in enumerate(LABELS):
                            fsc = list(res["per_label"][i]["frame_scores"])
                            thr = res["per_label"][i].get("threshold")
                            self.root.after(0, lambda l=lbl, s=fsc, t=thr:
                                self.right_panel.update_ai_score_bar(l, s, t))
                        self.root.after(0, self.refresh_frame_gallery)

                    # Cleanup referensi frame agar tidak menumpuk di memori
                    del imgs, landmark_results, res

                    # Periodic save setiap 50 video — cegah kehilangan progress jika crash
                    if idx % 50 == 49:
                        with self.save_lock:
                            save_frame_annotations(self.path_json_frames, self.frame_annotations)
                            save_batch_history(self.path_json_batch_history, self.batch_history)

                except Exception as e:
                    print(f"[Error] Failed to process {rel_path}: {e}")

            # Eksekusi semua video dengan Multi-threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_video, idx, vp) for idx, vp in enumerate(self.video_files)]
                concurrent.futures.wait(futures)

            # Simpan semua data sekaligus setelah batch selesai (bukan per-video)
            with self.save_lock:
                save_frame_annotations(self.path_json_frames, self.frame_annotations)
                save_batch_history(self.path_json_batch_history, self.batch_history)

            def on_finish():
                self.batch_running = False
                # Nyalakan kembali debug log setelah batch selesai
                import core.landmark_analyzer as _lm
                _lm._DBG_LAND = True
                msg = "Dibatalkan" if self.cancel_batch else "Selesai"
                self.right_panel.lbl_batch_status.configure(text=msg, text_color="#10b981" if not self.cancel_batch else "#ef4444")
                self.right_panel.btn_proses_semua.configure(
                    text="Proses Semua (Batch)", fg_color="#6366f1", hover_color="#4f46e5"
                )
                self.right_panel.btn_proses_satu.configure(state="normal")
                thrs = [v.get() for v in self.right_panel.threshold_vars]
                update_batch_meta(self.path_json_batch_history, self.rules, thrs)
            
            self.root.after(0, on_finish)

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoLabelerApp(root)
    root.mainloop()