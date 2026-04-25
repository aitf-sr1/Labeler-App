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
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
from dotenv import load_dotenv

load_dotenv()

from ui import LABELS, LABEL_COLORS, LeftPanel, RightPanel
from utils import (
    prepare_cropped_frames,
    load_annotations, save_annotations,
    load_flagged, save_flagged,
    load_frame_annotations, save_frame_annotations,
    load_batch_history, save_batch_history,
    load_skipped, save_skipped,
    load_thresholds, save_thresholds,
)
from core import run_siglip_on_frames

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

        self.cap, self.is_playing = None, False
        self.total_frames, self.current_frame, self.after_id = 0, 0, None
        self.fps, self.play_start_time, self.play_start_frame = 30.0, 0.0, 0

        self.annotations_data  = {}
        self.flagged_data      = set()
        self.frame_annotations = {}
        self.skipped_videos    = set()
        self.batch_history     = {}

        self.label_vars = {label: ctk.StringVar(value="0") for label in LABELS}
        self.flag_var   = ctk.BooleanVar(value=False)
        self.active_frame_label = ctk.StringVar(value="Boredom")
        self.show_viz   = ctk.BooleanVar(value=False)   # Toggle landmark viz di galeri
        self.viz_images = []   # Cache viz PIL images untuk video aktif

        self.batch_running, self.cancel_batch = False, False

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
        bar = ctk.CTkFrame(self.root, fg_color=("#e5e7eb", "#1f1f1f"), corner_radius=0, height=46)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Buka Folder", command=self.open_folder,
            font=self.font_bold, fg_color="#10b981", hover_color="#059669",
            width=110, height=30,
        ).pack(side="left", padx=(14, 4), pady=8)

        ctk.CTkButton(
            bar, text="Output", command=self._change_output_folder,
            font=self.font_sm, fg_color="#6366f1", hover_color="#4f46e5",
            width=80, height=30,
        ).pack(side="left", padx=(0, 10), pady=8)

        self.lbl_info = ctk.CTkLabel(
            bar, text="Pilih folder dataset untuk memulai",
            font=self.font_sm, text_color="gray",
        )
        self.lbl_info.pack(side="left")

        self.lbl_progress = ctk.CTkLabel(bar, text="", font=self.font_sm, text_color="gray")
        self.lbl_progress.pack(side="right", padx=16)

        self.lbl_flag_count = ctk.CTkLabel(
            bar, text="", font=self.font_sm, text_color="#ef4444"
        )
        self.lbl_flag_count.pack(side="right", padx=(0, 4))

        ctk.CTkSwitch(
            bar, text="Viz", variable=self.show_viz,
            font=self.font_sm, progress_color="#6366f1",
            command=self.refresh_frame_gallery,
        ).pack(side="right", padx=(0, 12))

    def _build_bottombar(self):
        """Bangun bar bawah: navigasi prev/skip/next, toggle flag, dan jump to video."""
        bar = ctk.CTkFrame(self.root, fg_color=("#e5e7eb", "#1f1f1f"), corner_radius=0, height=46)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Prev", command=self.prev_video,
            font=self.font_sm, fg_color=("#9ca3af", "#4b5563"),
            hover_color=("#6b7280", "#374151"), width=90, height=30,
        ).pack(side="left", padx=(14, 6), pady=8)

        ctk.CTkButton(
            bar, text="Skip", command=self.skip_video,
            font=self.font_sm, fg_color=("#f59e0b", "#b45309"),
            hover_color=("#d97706", "#92400e"), width=90, height=30,
        ).pack(side="left", padx=(6, 6), pady=8)

        self.chk_flag = ctk.CTkSwitch(
            bar, text="Flag / Reject", variable=self.flag_var,
            font=self.font_sm, progress_color="#ef4444",
        )
        self.chk_flag.pack(side="left", padx=16)

        ctk.CTkButton(
            bar, text="Save & Next", command=self.save_and_next,
            font=self.font_bold, fg_color="#3b82f6", hover_color="#2563eb",
            width=130, height=30,
        ).pack(side="right", padx=14, pady=8)

        ctk.CTkButton(
            bar, text="Go", command=self.jump_to_video,
            font=self.font_sm, fg_color=("#6b7280", "#374151"),
            hover_color=("#4b5563", "#1f2937"), width=40, height=30,
        ).pack(side="right", padx=(0, 4), pady=8)

        self.jump_entry = ctk.CTkEntry(
            bar, width=52, height=30, font=self.font_sm, placeholder_text="No."
        )
        self.jump_entry.pack(side="right", pady=8)
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_video())

        ctk.CTkLabel(
            bar, text="Loncat ke:", font=("Poppins", 9), text_color="gray"
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
        self.path_dir_cropped        = os.path.join(base, "cropped_faces")
        os.makedirs(self.path_dir_cropped, exist_ok=True)

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
        self.annotations_data  = load_annotations(self.path_csv_annotations)
        self.flagged_data      = load_flagged(self.path_csv_flagged)
        self.frame_annotations = load_frame_annotations(self.path_json_frames)
        self.batch_history     = load_batch_history(self.path_json_batch_history)
        self.skipped_videos    = load_skipped(self.path_json_skipped)
        self._load_saved_thresholds()
        self._update_flag_count()

    def _load_saved_thresholds(self):
        """Muat threshold tersimpan dan terapkan ke slider UI."""
        if not self.path_json_thresholds:
            return
        saved = load_thresholds(self.path_json_thresholds, LABELS)
        if saved:
            for i, val in enumerate(saved):
                self.right_panel.threshold_vars[i].set(val)
            print(f"[Threshold] Dimuat dari disk: {[f'{v:.2f}' for v in saved]}")

    def _save_current_thresholds(self):
        """Simpan threshold saat ini ke disk."""
        if not self.path_json_thresholds:
            return
        thrs = [v.get() for v in self.right_panel.threshold_vars]
        save_thresholds(self.path_json_thresholds, LABELS, thrs)

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
        self.path_dir_cropped        = os.path.join(folder, "cropped_faces")
        os.makedirs(self.path_dir_cropped, exist_ok=True)

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

        Jika video di-flag: hapus dari annotations, tambahkan ke flagged_data.
        Jika tidak di-flag: simpan nilai label ke annotations_data.
        Selalu tulis ulang annotations_bener.csv, flagged_videos.csv, dan frame_annotations.json.
        """
        if not self.video_files: return
        vp  = self.video_files[self.current_index]
        rel = os.path.relpath(vp, self.root_folder)

        if self.flag_var.get():
            self.flagged_data.add(rel)
            self.annotations_data.pop(rel, None)
        else:
            self.flagged_data.discard(rel)
            self.annotations_data[rel] = [self.label_vars[l].get() for l in LABELS]

        save_annotations(self.path_csv_annotations, self.annotations_data)
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
        self.fps = max(30.0, self.cap.get(cv2.CAP_PROP_FPS))
        self.left_panel.slider.configure(to=self.total_frames)
        self.left_panel.slider.set(0)
        self.current_frame = self.play_start_time = self.play_start_frame = 0

        rel = os.path.relpath(vp, self.root_folder)
        self.lbl_info.configure(text=os.path.basename(rel))
        self.lbl_progress.configure(text=f"{self.current_index + 1} / {len(self.video_files)}")

        self.flag_var.set(rel in self.flagged_data)
        saved = self.annotations_data.get(rel)
        if not self.flag_var.get() and saved:
            for i, lbl in enumerate(LABELS):
                self.label_vars[lbl].set(saved[i])
        else:
            for v in self.label_vars.values():
                v.set("0")

        self._refresh_all_label_buttons()
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
        target = int(self.total_frames * frame_idx / 16)
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

    def _set_label(self, label: str, value: str):
        """Set nilai label tertentu dan refresh tombol 0/1 di panel kanan."""
        self.label_vars[label].set(value)
        self.right_panel.refresh_label_buttons(label, value)

    def _refresh_all_label_buttons(self):
        """Sync semua tombol label 0/1 di panel kanan dengan nilai label_vars saat ini."""
        for lbl in LABELS:
            self.right_panel.refresh_label_buttons(lbl, self.label_vars[lbl].get())

    def _update_vote_bar(self, label: str):
        """Hitung rasio frame positif untuk label tertentu dan update vote bar."""
        if not self.video_files: return
        rel = os.path.relpath(self.video_files[self.current_index], self.root_folder)
        total_1 = sum(
            1 for j in range(16)
            if self.frame_annotations.get(rel, {}).get(str(j), {}).get(label, 0) == 1
        )
        self.right_panel.update_vote_bar(label, total_1 / 16)

    def refresh_frame_gallery(self, *args):
        """
        Render ulang galeri 16 frame dan update semua bar (vote + AI score).

        Juga menampilkan AI score dari batch_history jika video sudah pernah diproses AI.
        """
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)
        active_lbl = self.active_frame_label.get()

        pil_images, _, multi_fc, landmark_results, viz_images = prepare_cropped_frames(
            vp, self.root_folder, self.path_dir_cropped
        )
        self.viz_images = viz_images   # cache untuk toggle
        if not pil_images:
            for cv_widget in self.left_panel.frame_canvases:
                cv_widget.delete("all")
                cv_widget.configure(highlightbackground="#333")
            return

        vid_data = self.frame_annotations.get(rel_path, {})
        display_images = self.viz_images if (self.show_viz.get() and self.viz_images) else pil_images
        self.left_panel.render_frames(display_images, vid_data, active_lbl)

        for lbl in LABELS:
            self._update_vote_bar(lbl)

        # Tampilkan AI score dari batch_history jika video sudah pernah diproses
        hist = self.batch_history.get(rel_path)
        if hist:
            for i, lbl in enumerate(LABELS):
                avg = hist["per_label"].get(str(i), {}).get("avg_score")
                if avg is not None:
                    self.right_panel.update_ai_score_bar(lbl, avg)

    def toggle_single_frame(self, frame_idx: int):
        """
        Toggle label aktif pada satu frame dan update label video secara otomatis.

        Jika >= 8 dari 16 frame positif, label video otomatis diset ke '1'.
        State langsung disimpan ke disk setelah setiap toggle.

        Args:
            frame_idx: Index frame yang di-toggle (0-15).
        """
        if not self.video_files: return
        vp       = self.video_files[self.current_index]
        rel_path = os.path.relpath(vp, self.root_folder)
        active_lbl = self.active_frame_label.get()

        if rel_path not in self.frame_annotations:
            self.frame_annotations[rel_path] = {
                str(i): {l: 0 for l in LABELS} for i in range(16)
            }
        if str(frame_idx) not in self.frame_annotations[rel_path]:
            self.frame_annotations[rel_path][str(frame_idx)] = {l: 0 for l in LABELS}

        current_val = self.frame_annotations[rel_path][str(frame_idx)][active_lbl]
        new_val     = 1 if current_val == 0 else 0
        self.frame_annotations[rel_path][str(frame_idx)][active_lbl] = new_val

        self.left_panel.update_single_frame_highlight(frame_idx, active_lbl, new_val)

        total_1 = sum(
            1 for i in range(16)
            if self.frame_annotations[rel_path].get(str(i), {}).get(active_lbl, 0) == 1
        )
        self.label_vars[active_lbl].set("1" if total_1 >= 8 else "0")
        self.right_panel.refresh_label_buttons(active_lbl, self.label_vars[active_lbl].get())
        self.right_panel.update_vote_bar(active_lbl, total_1 / 16)
        self.save_current_state()

    # SigLIP inference

    def _apply_siglip_result(self, rel_path: str, res: dict, update_label_vars: bool = False):
        """
        Tulis hasil inferensi SigLIP ke frame_annotations dan batch_history.

        Args:
            rel_path:         Relative path video terhadap root_folder.
            res:              Output dari run_siglip_on_frames().
            update_label_vars: Jika True, update juga label_vars (dipakai saat proses satu video).
        """
        if rel_path not in self.frame_annotations:
            self.frame_annotations[rel_path] = {}

        for f_idx in range(16):
            if str(f_idx) not in self.frame_annotations[rel_path]:
                self.frame_annotations[rel_path][str(f_idx)] = {}
            for i, lbl in enumerate(LABELS):
                self.frame_annotations[rel_path][str(f_idx)][lbl] = \
                    res["per_label"][i]["frame_preds"][f_idx]

        self.batch_history[rel_path] = {
            "per_label": {
                str(i): {
                    "prediction":    res["per_label"][i]["prediction"],
                    "vote_pos":      res["per_label"][i]["vote_pos"],
                    "vote_neg":      res["per_label"][i]["vote_neg"],
                    "skipped":       res["per_label"][i]["skipped"],
                    "avg_score":     res["per_label"][i]["avg_score"],
                    "siglip_avg":    res["per_label"][i].get("siglip_avg"),
                    "landmark_avg":  res["per_label"][i].get("landmark_avg"),
                    "threshold":     res["thresholds"][i],
                    "frame_scores":  res["per_label"][i]["frame_scores"],
                    "frame_preds":   res["per_label"][i]["frame_preds"],
                }
                for i in range(len(LABELS))
            },
            "thresholds": res["thresholds"],
        }
        save_batch_history(self.path_json_batch_history, self.batch_history)

        if update_label_vars:
            for i, lbl in enumerate(LABELS):
                self.label_vars[lbl].set(str(res["per_label"][i]["prediction"]))

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
                vp, self.root_folder, self.path_dir_cropped
            )
            self.viz_images = viz_imgs
            if not imgs:
                self.root.after(0, lambda: self.right_panel.lbl_batch_status.configure(
                    text="Gagal: tidak ada frame", text_color="#ef4444"
                ))
                return

            # Auto-flag jika frame kurang dari 16 (video terlalu pendek)
            if len(imgs) < 16:
                self.flagged_data.add(rel_path)
                self.annotations_data.pop(rel_path, None)
                save_flagged(self.path_csv_flagged, self.flagged_data)
                self.root.after(0, lambda nf=len(imgs): (
                    self.flag_var.set(True),
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Auto-flag: hanya {nf}/16 frame (video terlalu pendek)",
                        text_color="#ef4444"
                    ),
                    self._update_flag_count()
                ))
                return

            # Auto-flag jika >50% frame tidak terdeteksi wajah oleh MediaPipe
            if no_face_count > 8:
                self.flagged_data.add(rel_path)
                self.annotations_data.pop(rel_path, None)
                save_flagged(self.path_csv_flagged, self.flagged_data)
                self.root.after(0, lambda: (
                    self.flag_var.set(True),
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Auto-flag: wajah hanya terdeteksi {16 - no_face_count}/16 frame",
                        text_color="#ef4444"
                    ),
                    self._update_flag_count()
                ))
                return

            # Auto-flag jika terlalu banyak frame dengan >1 wajah
            mf_thr = int(os.getenv("MULTI_FACE_FRAMES_THRESHOLD", "4"))
            if multi_face_count > mf_thr:
                self.flagged_data.add(rel_path)
                self.annotations_data.pop(rel_path, None)
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

            res = run_siglip_on_frames(imgs, prompts, ths,
                                        landmark_results=landmark_results)
            self._apply_siglip_result(rel_path, res, update_label_vars=True)

            def update_ui():
                for i, lbl in enumerate(LABELS):
                    r = res["per_label"][i]
                    self.right_panel.update_ai_score_bar(lbl, r["avg_score"])
                    land = f" land={r['landmark_avg']:.3f}" if r.get("landmark_avg") is not None else ""
                    print(f"{lbl}: pred={r['prediction']} hybrid={r['avg_score']:.3f} "
                          f"siglip={r['siglip_avg']:.3f}{land} votes={r['vote_pos']}/{r['vote_neg']}")
                self._refresh_all_label_buttons()
                self.save_current_state()
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
        self.right_panel.btn_proses_semua.configure(
            text="Hentikan Batch", fg_color="#ef4444", hover_color="#dc2626"
        )
        self.right_panel.btn_proses_satu.configure(state="disabled")

        prompts, ths = self.right_panel.get_prompts_and_thresholds()

        def worker():
            total         = len(self.video_files)
            already_done  = set(self.batch_history.keys())
            skipped_count = 0

            for idx, vp in enumerate(self.video_files):
                if self.cancel_batch: break
                rel_path = os.path.relpath(vp, self.root_folder)

                if rel_path in already_done:
                    skipped_count += 1
                    def upd_skip(i=idx, r=rel_path, sc=skipped_count):
                        self.right_panel.lbl_batch_status.configure(
                            text=f"Skip (sudah ada) {i+1}/{total}: {os.path.basename(r)}  {sc} cached",
                            text_color="#6b7280",
                        )
                    self.root.after(0, upd_skip)
                    continue

                def upd_status(i=idx, r=rel_path):
                    self.right_panel.lbl_batch_status.configure(
                        text=f"Batch {i+1}/{total}: {os.path.basename(r)}",
                        text_color="#fbbf24",
                    )
                self.root.after(0, upd_status)

                try:
                    imgs, no_face_count, multi_face_count, landmark_results, _ = prepare_cropped_frames(
                        vp, self.root_folder, self.path_dir_cropped
                    )
                    if not imgs: continue

                    # Auto-flag jika frame kurang dari 16 (video terlalu pendek, mis. <1 detik)
                    if len(imgs) < 16:
                        self.flagged_data.add(rel_path)
                        self.annotations_data.pop(rel_path, None)
                        save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_shortflag(r=rel_path, nf=len(imgs)):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} (hanya {nf}/16 frame)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_shortflag)
                        continue

                    # Auto-flag jika >50% frame tidak terdeteksi wajah
                    if no_face_count > 8:
                        self.flagged_data.add(rel_path)
                        self.annotations_data.pop(rel_path, None)
                        save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_autoflag(r=rel_path, nf=no_face_count):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} (wajah {16-nf}/16)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_autoflag)
                        continue

                    # Auto-flag jika terlalu banyak frame dengan >1 wajah
                    mf_thr = int(os.getenv("MULTI_FACE_FRAMES_THRESHOLD", "4"))
                    if multi_face_count > mf_thr:
                        self.flagged_data.add(rel_path)
                        self.annotations_data.pop(rel_path, None)
                        save_flagged(self.path_csv_flagged, self.flagged_data)
                        self.root.after(0, self._update_flag_count)
                        def upd_multiface(r=rel_path, mfc=multi_face_count):
                            self.right_panel.lbl_batch_status.configure(
                                text=f"Auto-flag: {os.path.basename(r)} ({mfc} frame multi-wajah)",
                                text_color="#ef4444",
                            )
                        self.root.after(0, upd_multiface)
                        continue

                    res = run_siglip_on_frames(imgs, prompts, ths,
                                               landmark_results=landmark_results)

                    self._apply_siglip_result(rel_path, res, update_label_vars=False)

                    if rel_path not in self.flagged_data:
                        self.annotations_data[rel_path] = [
                            str(res["per_label"][i]["prediction"]) for i in range(4)
                        ]

                    current_rel = os.path.relpath(
                        self.video_files[self.current_index], self.root_folder
                    )
                    if rel_path == current_rel:
                        for i, lbl in enumerate(LABELS):
                            self.label_vars[lbl].set(str(res["per_label"][i]["prediction"]))
                            ai_score = res["per_label"][i]["avg_score"]
                            self.root.after(0, lambda l=lbl, s=ai_score:
                                self.right_panel.update_ai_score_bar(l, s))
                        self.root.after(0, self._refresh_all_label_buttons)
                        self.root.after(0, self.refresh_frame_gallery)

                    save_annotations(self.path_csv_annotations, self.annotations_data)
                    save_frame_annotations(self.path_json_frames, self.frame_annotations)

                except Exception as e:
                    print(f"[Error] Failed to process {rel_path}: {e}")

            def on_finish():
                self.batch_running = False
                msg = "Dibatalkan" if self.cancel_batch else "Selesai"
                self.right_panel.lbl_batch_status.configure(text=msg, text_color="#10b981" if not self.cancel_batch else "#ef4444")
                self.right_panel.btn_proses_semua.configure(
                    text="Proses Semua (Batch)", fg_color="#6366f1", hover_color="#4f46e5"
                )
                self.right_panel.btn_proses_satu.configure(state="normal")
            
            self.root.after(0, on_finish)

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoLabelerApp(root)
    root.mainloop()