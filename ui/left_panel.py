"""
ui/left_panel.py

Panel kiri: video player, slider, dan galeri 6 frame dengan tab label aktif.

Semua callback (seek, toggle_frame, toggle_play, slider) di-inject dari App.
Panel ini tidak menyimpan state anotasi — hanya tampilan.
"""

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

from .constants import LABELS, LABEL_COLORS


class LeftPanel:
    """
    Komponen panel kiri aplikasi.

    Bertanggung jawab atas:
        - Canvas video player + slider posisi
        - Grid 2x3 galeri frame (6 thumbnail)
        - Tab selector untuk label aktif per frame

    Interaksi frame:
        - Klik kiri  -> seek video ke posisi frame tersebut (app.seek_to_frame)
        - Klik kanan -> toggle label aktif pada frame tersebut (app.toggle_single_frame)
    """

    def __init__(self, parent, app):
        self.app              = app
        self.frame_canvases     = []
        self.frame_dot_canvases = []  # strip dot di sebelah kanan masing-masing frame
        self.frame_image_refs   = []  # Referensi ImageTk agar tidak di-GC
        self._emotion_tab_btns  = {}
        self._current_frame_annotations: dict = {}  # referensi ke frame_annotations video aktif

        self._build(parent)

    def _build(self, parent):
        left = ctk.CTkFrame(parent, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(8, 0))
        left.rowconfigure(0, weight=0)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self._build_video_player(left)
        self._build_frame_gallery(left)

    @staticmethod
    def _bind_mousewheel(scrollable_frame):
        """Aktifkan scroll mouse saat kursor masuk ke area scrollable frame."""
        canvas = scrollable_frame._parent_canvas

        def on_enter(_):
            scrollable_frame.bind_all(
                "<MouseWheel>",
                lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            )
            scrollable_frame.bind_all("<Button-4>", lambda _: canvas.yview_scroll(-1, "units"))
            scrollable_frame.bind_all("<Button-5>", lambda _: canvas.yview_scroll(1, "units"))

        def on_leave(_):
            scrollable_frame.unbind_all("<MouseWheel>")
            scrollable_frame.unbind_all("<Button-4>")
            scrollable_frame.unbind_all("<Button-5>")

        scrollable_frame.bind("<Enter>", on_enter)
        scrollable_frame.bind("<Leave>", on_leave)

    def _build_video_player(self, parent):
        vid_wrap = ctk.CTkFrame(parent, fg_color="#111111", corner_radius=8)
        vid_wrap.grid(row=0, column=0, sticky="ew")

        self.canvas_video = tk.Canvas(
            vid_wrap, width=640, height=320, bg="#111111", highlightthickness=0
        )
        self.canvas_video.pack(pady=4)
        self.canvas_image_item = self.canvas_video.create_image(0, 0, anchor="nw")

        ctrl = ctk.CTkFrame(vid_wrap, fg_color="transparent")
        ctrl.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkButton(
            ctrl, text="Play / Pause", command=self.app.toggle_play,
            font=self.app.font_sm, width=120, height=28,
        ).pack(side="left", padx=(0, 10))
        self.slider = ctk.CTkSlider(ctrl, from_=0, to=100, command=self.app.on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.slider.set(0)

    def _build_frame_gallery(self, parent):
        gallery_scroll = ctk.CTkScrollableFrame(
            parent, fg_color=("f3f4f6", "#161622"), corner_radius=8
        )
        gallery_scroll.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self._bind_mousewheel(gallery_scroll)

        hdr = ctk.CTkFrame(gallery_scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            hdr, text="Frame-level labels — klik kanan untuk toggle",
            font=self.app.font_sm, text_color="gray",
        ).pack(side="left")


        tab_row = ctk.CTkFrame(hdr, fg_color="transparent")
        tab_row.pack(side="right")
        for lbl in LABELS:
            color = LABEL_COLORS[lbl]
            b = ctk.CTkButton(
                tab_row, text=lbl, width=88, height=24,
                font=("Poppins", 10, "bold"),
                fg_color="transparent",
                border_width=1, border_color=color,
                text_color=color, hover_color=("e5e7eb", "#2a2a3a"),
                command=lambda l=lbl: self.app._set_active_tab(l),
            )
            b.pack(side="left", padx=3)
            self._emotion_tab_btns[lbl] = b

        self.lbl_frame_quality = ctk.CTkLabel(
            gallery_scroll, text="", font=("Poppins", 9), text_color="#f59e0b"
        )
        self.lbl_frame_quality.pack(anchor="w", padx=12, pady=(0, 4))

        grid_frame = ctk.CTkFrame(gallery_scroll, fg_color="transparent")
        grid_frame.pack(expand=True, fill="both", padx=10, pady=(0, 8))
        for col in range(2):
            grid_frame.columnconfigure(col, weight=1)

        self._manual_check_frames = []   # list of {lbl: BooleanVar} per frame

        for i in range(2):
            row_g, col_g = i // 2, i % 2

            # Outer cell: gambar + dot + manual checkboxes (vertikal)
            outer = tk.Frame(grid_frame, bg="#0d0d0d")
            outer.grid(row=row_g, column=col_g, padx=8, pady=8)

            # Baris atas: canvas + dot strip
            cell = tk.Frame(outer, bg="#0d0d0d")
            cell.pack()

            cv_widget = tk.Canvas(
                cell, width=360, height=360,
                bg="#111", highlightthickness=2, highlightbackground="#333",
            )
            cv_widget.pack(side="left")
            cv_widget.bind("<Button-1>", lambda e, idx=i: self.app.seek_to_frame(idx))
            cv_widget.bind("<Button-3>", lambda e, idx=i: self.app.toggle_single_frame(idx))
            cv_widget.bind("<Double-Button-1>", lambda e, idx=i: self.app.toggle_frame_reject(idx))
            self.frame_canvases.append(cv_widget)

            dot_cv = tk.Canvas(cell, width=20, height=360, bg="#0d0d0d", highlightthickness=0)
            dot_cv.pack(side="left", padx=(4, 0))
            self.frame_dot_canvases.append(dot_cv)

            # Baris bawah: checkboxes manual label (tersembunyi default)
            chk_row = tk.Frame(outer, bg="#0d0d0d")
            # TIDAK di-pack sekarang — muncul saat semi_manual aktif
            frame_vars = {}
            for lbl in LABELS:
                color = LABEL_COLORS[lbl]
                var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(
                    chk_row, text=lbl, variable=var,
                    bg="#0d0d0d", fg=color, selectcolor="#1a1a2e",
                    activebackground="#0d0d0d", activeforeground=color,
                    font=("Poppins", 9, "bold"), bd=0,
                    command=lambda fi=i, l=lbl, v=var: self.app._on_manual_check(fi, l, v.get()),
                )
                cb.pack(side="left", padx=6)
                frame_vars[lbl] = var
            self._manual_check_frames.append({"vars": frame_vars, "row": chk_row})

        ctk.CTkLabel(
            gallery_scroll,
            text="Klik kiri: seek  |  Klik kanan: toggle label  |  Double-klik: tolak frame",
            font=("Poppins", 9), text_color="#6b7280",
        ).pack(pady=(0, 6))

    def show_manual_checkboxes(self, visible: bool):
        """Tampilkan atau sembunyikan baris checkbox manual label di bawah tiap frame."""
        for entry in self._manual_check_frames:
            if visible:
                entry["row"].pack(pady=(4, 0))
            else:
                entry["row"].pack_forget()

    def update_manual_checkboxes(self, frame_idx: int, label_dict: dict):
        """Update state checkbox untuk satu frame dari dict {lbl: 0|1}."""
        if frame_idx >= len(self._manual_check_frames):
            return
        for lbl, var in self._manual_check_frames[frame_idx]["vars"].items():
            var.set(bool(label_dict.get(lbl, 0)))

    def show_loading(self):
        """Tampilkan state loading di semua canvas frame (saat prepare_cropped_frames berjalan)."""
        for cv_widget in self.frame_canvases:
            cv_widget.delete("all")
            cv_widget.configure(highlightbackground="#333")
            cv_widget.create_text(
                180, 180, text="memuat...", fill="#4b5563",
                font=("Poppins", 10), anchor="center",
            )
        for dot_cv in self.frame_dot_canvases:
            dot_cv.delete("all")
        self.lbl_frame_quality.configure(text="")

    def update_frame_quality(
        self,
        no_face_count: int,
        multi_face_count: int = 0,
        rejected_count: int = 0,
        n_frames: int = 2,
    ):
        """Tampilkan info kualitas frame di atas galeri."""
        parts = []
        if rejected_count > 0:
            parts.append(f"{rejected_count}/{n_frames} frame ditolak")
        if no_face_count > 0:
            parts.append(f"{no_face_count}/{n_frames} frame tanpa wajah")
        if multi_face_count > 0:
            parts.append(f"{multi_face_count}/{n_frames} frame multi-wajah")

        if not parts:
            self.lbl_frame_quality.configure(text="")
        else:
            color = "#ef4444" if (rejected_count > 0 or no_face_count == n_frames) else "#f59e0b"
            self.lbl_frame_quality.configure(
                text="⚠  " + "   |   ".join(parts),
                text_color=color,
            )

    def set_active_tab_highlight(self, active_label: str):
        """
        Tandai tab label yang sedang aktif dengan warna solid, sisanya outline saja.

        Args:
            active_label: Nama label yang sedang aktif (dari LABELS).
        """
        for lbl, btn in self._emotion_tab_btns.items():
            color = LABEL_COLORS[lbl]
            if lbl == active_label:
                btn.configure(
                    fg_color=color,
                    text_color=("#1a1a1a" if lbl in ("Boredom", "Engagement") else "white"),
                )
            else:
                btn.configure(fg_color="transparent", text_color=color)

    def _draw_frame_dots(self, dot_cv, frame_data: dict):
        """Gambar 4 buletan label secara vertikal di strip canvas samping frame.
        Tidak menggambar apapun jika frame belum pernah dideteksi."""
        dot_cv.delete("all")
        if not any(lbl in frame_data for lbl in LABELS):
            return
        dot_r   = 6
        gap     = 8
        n       = len(LABELS)
        total_h = n * (2 * dot_r) + (n - 1) * gap
        start_y = (360 - total_h) // 2
        cx      = 10  # tengah strip 20px
        for j, lbl in enumerate(LABELS):
            cy     = start_y + j * (2 * dot_r + gap) + dot_r
            active = frame_data.get(lbl, 0) == 1
            color  = LABEL_COLORS[lbl] if active else "#2a2a2a"
            border = LABEL_COLORS[lbl] if not active else color
            dot_cv.create_oval(
                cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                fill=color, outline=border, width=1,
            )

    def render_frames(self, pil_images: list, frame_annotations_for_video: dict, active_label: str):
        """
        Render ulang semua 6 canvas dengan thumbnail dan highlight label aktif.

        Border canvas berwarna jika frame tersebut di-label positif untuk label aktif.
        Jika frame kosong (tidak ada gambar), canvas di-clear.

        Args:
            pil_images:                  List of PIL.Image dari 6 frame video.
            frame_annotations_for_video: Dict {frame_idx: {label: 0|1}} untuk video saat ini.
            active_label:                Label yang sedang aktif di tab selector.
        """
        self._current_frame_annotations = frame_annotations_for_video  # simpan referensi untuk update dots
        active_color = LABEL_COLORS[active_label]
        self.frame_image_refs.clear()

        for i, cv_widget in enumerate(self.frame_canvases):
            if i >= len(pil_images):
                cv_widget.delete("all")
                cv_widget.configure(highlightbackground="#333")
                continue

            img    = pil_images[i].resize((360, 360))
            tk_img = ImageTk.PhotoImage(img)
            self.frame_image_refs.append(tk_img)
            cv_widget.delete("all")
            cv_widget.create_image(0, 0, anchor="nw", image=tk_img)

            frame_data  = frame_annotations_for_video.get(str(i), {})
            is_rejected = frame_data.get("_rejected", False)

            if is_rejected:
                cv_widget.configure(highlightbackground="#ef4444")
                cv_widget.create_rectangle(0, 0, 360, 360, fill="#2a0000", stipple="gray50")
                cv_widget.create_text(
                    180, 180, text="FRAME DITOLAK", fill="#ef4444",
                    font=("Poppins", 11, "bold"), anchor="center",
                )
            else:
                status = frame_data.get(active_label, 0)
                cv_widget.configure(highlightbackground=active_color if status == 1 else "#333")

            self._draw_frame_dots(self.frame_dot_canvases[i], frame_data)

    def update_single_frame_highlight(self, frame_idx: int, active_label: str, status: int):
        """
        Update border satu frame canvas tanpa re-render seluruh galeri.

        Args:
            frame_idx:    Index frame (0-15).
            active_label: Label yang sedang aktif.
            status:       0 atau 1 — nilai label frame tersebut.
        """
        color = LABEL_COLORS[active_label]
        self.frame_canvases[frame_idx].configure(
            highlightbackground=color if status == 1 else "#333"
        )
        frame_data = self._current_frame_annotations.get(str(frame_idx), {})
        self._draw_frame_dots(self.frame_dot_canvases[frame_idx], frame_data)

    def show_video_frame(self, frame_bgr):
        """
        Tampilkan satu frame BGR ke canvas video player.

        Frame di-resize ke 640x320 dan dikonversi dari BGR ke RGB sebelum ditampilkan.

        Args:
            frame_bgr: Frame dalam format BGR (output OpenCV).
        """
        img = ImageTk.PhotoImage(
            image=Image.fromarray(
                cv2.cvtColor(cv2.resize(frame_bgr, (640, 320)), cv2.COLOR_BGR2RGB)
            )
        )
        self.canvas_video.itemconfig(self.canvas_image_item, image=img)
        self.canvas_video.image = img
