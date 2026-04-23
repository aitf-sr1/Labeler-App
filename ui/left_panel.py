"""
ui/left_panel.py

Panel kiri: video player, slider, dan galeri 16 frame dengan tab label aktif.

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
        - Grid 4x4 galeri frame (16 thumbnail)
        - Tab selector untuk label aktif per frame

    Interaksi frame:
        - Klik kiri  -> seek video ke posisi frame tersebut (app.seek_to_frame)
        - Klik kanan -> toggle label aktif pada frame tersebut (app.toggle_single_frame)
    """

    def __init__(self, parent, app):
        self.app              = app
        self.frame_canvases   = []
        self.frame_image_refs = []  # Referensi ImageTk agar tidak di-GC
        self._emotion_tab_btns = {}

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

        grid_frame = ctk.CTkFrame(gallery_scroll, fg_color="transparent")
        grid_frame.pack(expand=True, fill="both", padx=10, pady=(0, 8))
        for col in range(4):
            grid_frame.columnconfigure(col, weight=1)

        for i in range(16):
            row_g, col_g = i // 4, i % 4
            cv_widget = tk.Canvas(
                grid_frame, width=110, height=110,
                bg="#111", highlightthickness=2, highlightbackground="#333",
            )
            cv_widget.grid(row=row_g, column=col_g, padx=8, pady=8)
            cv_widget.bind("<Button-1>", lambda e, idx=i: self.app.seek_to_frame(idx))
            cv_widget.bind("<Button-3>", lambda e, idx=i: self.app.toggle_single_frame(idx))
            self.frame_canvases.append(cv_widget)

        ctk.CTkLabel(
            gallery_scroll,
            text="Klik kiri: seek video  |  Klik kanan: toggle label",
            font=("Poppins", 9), text_color="#6b7280",
        ).pack(pady=(0, 6))

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

    def render_frames(self, pil_images: list, frame_annotations_for_video: dict, active_label: str):
        """
        Render ulang semua 16 canvas dengan thumbnail dan highlight label aktif.

        Border canvas berwarna jika frame tersebut di-label positif untuk label aktif.
        Jika frame kosong (tidak ada gambar), canvas di-clear.

        Args:
            pil_images:                  List of PIL.Image dari 16 frame video.
            frame_annotations_for_video: Dict {frame_idx: {label: 0|1}} untuk video saat ini.
            active_label:                Label yang sedang aktif di tab selector.
        """
        active_color = LABEL_COLORS[active_label]
        self.frame_image_refs.clear()

        for i, cv_widget in enumerate(self.frame_canvases):
            if i >= len(pil_images):
                cv_widget.delete("all")
                cv_widget.configure(highlightbackground="#333")
                continue

            img    = pil_images[i].resize((110, 110))
            tk_img = ImageTk.PhotoImage(img)
            self.frame_image_refs.append(tk_img)
            cv_widget.delete("all")
            cv_widget.create_image(0, 0, anchor="nw", image=tk_img)

            status = frame_annotations_for_video.get(str(i), {}).get(active_label, 0)
            cv_widget.configure(highlightbackground=active_color if status == 1 else "#333")

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
