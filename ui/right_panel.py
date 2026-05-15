"""
ui/right_panel.py

Panel kanan: tombol inferensi AI, toggle label 0/1, vote bar, AI score bar,
prompt editor per label, dan threshold slider.

Struktur layout (atas ke bawah):
    [Proses Video Ini] [Batch Semua]
    Status label batch
    ─────────────────────────────
    Scrollable accordion per label:
        [Header accordion — klik untuk expand]
        [Body: tombol 0/1 | vote bar | AI score bar | prompt | threshold]
"""

import tkinter as tk
import customtkinter as ctk

from .constants import LABELS, LABEL_COLORS, DEFAULT_PROMPT_GROUPS


class RightPanel:
    """
    Komponen panel kanan aplikasi.

    Atribut publik yang diakses dari App:
        pos_textboxes[i]       -- CTkTextbox prompt positif per label
        threshold_vars[i]      -- DoubleVar threshold per label
        ai_score_canvases[lbl] -- tuple (Canvas, color) untuk AI score
        ai_score_labels[lbl]   -- StringVar teks nilai AI score
        lbl_batch_status       -- CTkLabel status proses batch
        btn_proses_satu        -- CTkButton untuk proses satu video
        btn_proses_semua       -- CTkButton untuk batch semua video
    """

    def __init__(self, parent, app):
        self.app = app
        self.pos_textboxes     = []
        self.threshold_vars    = [ctk.DoubleVar(value=0.50) for _ in LABELS]
        self.ai_score_canvases = {}
        self.ai_score_labels   = {}
        self.acc_bodies        = []
        self.acc_open_flags    = []
        self.threshold_labels  = []

        self._build(parent)

    def _build(self, parent):
        right = ctk.CTkFrame(
            parent, fg_color=("f3f4f6", "#1a1a2a"), corner_radius=0, width=330
        )
        right.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
        right.pack_propagate(False)
        right.columnconfigure(0, weight=1)

        self._build_action_buttons(right)
        self._build_statistics_panel(right)
        self._build_label_accordions(right)

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

    def _build_action_buttons(self, parent):
        top_btns = ctk.CTkFrame(parent, fg_color="transparent")
        top_btns.pack(fill="x", padx=12, pady=(10, 4))

        self.btn_proses_satu = ctk.CTkButton(
            top_btns, text="Proses Video Ini",
            command=self.app._proses_satu,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=self.app.font_sm, height=30,
        )
        self.btn_proses_satu.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_proses_semua = ctk.CTkButton(
            top_btns, text="Batch Semua",
            command=self.app._toggle_batch,
            fg_color="#10b981", hover_color="#059669",
            font=self.app.font_sm, height=30,
        )
        self.btn_proses_semua.pack(side="left", fill="x", expand=True)

        self.lbl_batch_status = ctk.CTkLabel(
            parent, text="Siap", font=("Poppins", 10), text_color="gray"
        )
        self.lbl_batch_status.pack(anchor="w", padx=14, pady=(0, 2))

        self.btn_restart_batch = ctk.CTkButton(
            parent, text="Restart Batch", command=self.app._restart_batch,
            fg_color="#f59e0b", hover_color="#d97706",
            font=("Poppins", 9), height=24, width=120,
        )
        self.btn_restart_batch.pack(anchor="w", padx=14, pady=(0, 2))

        # ── Batch versioning ──────────────────────────────────────────────────
        bv_frame = ctk.CTkFrame(parent, fg_color=("e8e8f0", "#1a1a2a"), corner_radius=4)
        bv_frame.pack(fill="x", padx=14, pady=(4, 2))
        self.batch_new_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            bv_frame, text="Buat batch baru:", variable=self.batch_new_var,
            font=("Poppins", 9), text_color="gray",
            checkbox_width=14, checkbox_height=14,
        ).pack(side="left", padx=(6, 4), pady=4)
        self.batch_name_entry = ctk.CTkEntry(
            bv_frame, placeholder_text="nama file…", font=("Poppins", 9),
            height=22, fg_color=("white", "#111122"),
        )
        self.batch_name_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)

        self.btn_reset_label = ctk.CTkButton(
            parent, text="Reset Label Video Ini", command=self.app.reset_current_labels,
            fg_color="#b91c1c", hover_color="#991b1b",
            font=("Poppins", 9), height=24, width=120,
        )
        self.btn_reset_label.pack(anchor="w", padx=14, pady=(6, 2))

        # ── Split Label 2D ────────────────────────────────────────────────────
        split_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        split_hdr.pack(fill="x", padx=14, pady=(6, 0))
        self.btn_split_2d = ctk.CTkButton(
            split_hdr, text="Split Label 2D", command=self.app._split_dataset_2d,
            fg_color="#0ea5e9", hover_color="#0284c7",
            font=("Poppins", 9, "bold"), height=26,
        )
        self.btn_split_2d.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(split_hdr, text="UUID idx:", font=("Poppins", 8), text_color="gray").pack(side="left")
        self.split_uuid_depth_entry = ctk.CTkEntry(
            split_hdr, width=30, height=26, font=("Poppins", 9),
        )
        self.split_uuid_depth_entry.insert(0, "2")
        self.split_uuid_depth_entry.pack(side="left", padx=(2, 0))

        sv_frame = ctk.CTkFrame(parent, fg_color=("e8e8f0", "#1a1a2a"), corner_radius=4)
        sv_frame.pack(fill="x", padx=14, pady=(2, 2))
        self.split_new_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sv_frame, text="Folder baru:", variable=self.split_new_var,
            font=("Poppins", 9), text_color="gray",
            checkbox_width=14, checkbox_height=14,
        ).pack(side="left", padx=(6, 4), pady=4)
        ctk.CTkLabel(
            sv_frame, text="← nama batch di atas",
            font=("Poppins", 8), text_color="#6b7280",
        ).pack(side="left", padx=(0, 6), pady=4)

        ctk.CTkFrame(parent, fg_color=("d1d5db", "#2e2e3e"), height=1).pack(
            fill="x", padx=12, pady=(4, 6)
        )

    def _build_statistics_panel(self, parent):
        import os
        stats_frame = ctk.CTkFrame(parent, fg_color=("#f0f0f5", "#1e1e2e"), corner_radius=6)
        stats_frame.pack(fill="x", padx=12, pady=(0, 6))

        hdr = ctk.CTkFrame(stats_frame, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(hdr, text="Statistik AI", font=("Poppins", 10, "bold"), text_color="gray").pack(side="left")

        self.lbl_stat_total = ctk.CTkLabel(hdr, text="Total: 0", font=("Poppins", 9, "bold"), text_color="#10b981")
        self.lbl_stat_total.pack(side="right")

        # ── Batch file dropdown ───────────────────────────────────────────────
        brow = ctk.CTkFrame(stats_frame, fg_color="transparent")
        brow.pack(fill="x", padx=8, pady=(0, 2))

        ctk.CTkLabel(brow, text="Baca:", font=("Poppins", 8), text_color="gray").pack(side="left", padx=(0, 3))

        self._batch_file_var = ctk.StringVar(value="batch_history.json")
        self._batch_file_menu = ctk.CTkOptionMenu(
            brow, variable=self._batch_file_var,
            values=["batch_history.json"],
            width=175, height=22,
            font=("Poppins", 8),
            command=self._on_batch_file_select,
        )
        self._batch_file_menu.pack(side="left", fill="x", expand=True, padx=(0, 2))

        ctk.CTkButton(
            brow, text="↺", width=26, height=22,
            font=("Poppins", 11), fg_color="transparent",
            text_color=("gray", "#6b7280"), hover_color=("#e5e7eb", "#374151"),
            command=self.refresh_batch_files,
        ).pack(side="left")

        # ── Label count grid ──────────────────────────────────────────────────
        grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        grid.pack(fill="x", padx=8, pady=(2, 6))

        self.stat_labels = {}
        for i, lbl in enumerate(LABELS):
            color = LABEL_COLORS[lbl]
            row = i // 2
            col = i % 2
            var_label = ctk.CTkLabel(grid, text=f"{lbl}: 0", font=("Poppins", 9), text_color=color)
            var_label.grid(row=row, column=col, sticky="w", padx=(0, 15))
            self.stat_labels[lbl] = var_label

    def refresh_batch_files(self, base_dir=None):
        """Rescan folder output untuk daftar batch_history_*.json terbaru."""
        import os, glob
        if base_dir is None:
            bh_path = getattr(self.app, "path_json_batch_history", "")
            if not bh_path:
                return
            base_dir = os.path.dirname(bh_path)

        files = []
        if os.path.exists(os.path.join(base_dir, "batch_history.json")):
            files.append("batch_history.json")
        for f in sorted(glob.glob(os.path.join(base_dir, "batch_history_*.json"))):
            files.append(os.path.basename(f))
        if not files:
            files = ["batch_history.json"]

        self._batch_file_menu.configure(values=files)
        if self._batch_file_var.get() not in files:
            self._batch_file_var.set(files[0])

    def _on_batch_file_select(self, filename: str):
        """Load batch history yang dipilih dari dropdown dan update statistik."""
        import os
        bh_path = getattr(self.app, "path_json_batch_history", "")
        if not bh_path:
            return
        full_path = os.path.join(os.path.dirname(bh_path), filename)
        try:
            from utils import load_batch_history
            bh = load_batch_history(full_path)
            self.update_statistics(bh)
        except Exception as exc:
            print(f"[Stats] Gagal load {filename}: {exc}")

    def update_statistics(self, batch_history: dict):
        total_frames = 0
        counts = {lbl: 0 for lbl in LABELS}

        for vid_data in batch_history.values():
            per_label = vid_data.get("per_label", {})

            try:
                total_frames += len(per_label.get("0", {}).get("frame_preds", []))
            except Exception:
                total_frames += 6

            for i, lbl in enumerate(LABELS):
                v_pos = per_label.get(str(i), {}).get("vote_pos", 0)
                counts[lbl] += v_pos

        self.lbl_stat_total.configure(text=f"Total: {total_frames} img")

        for lbl in LABELS:
            self.stat_labels[lbl].configure(text=f"{lbl}: {counts[lbl]}")

    def _build_label_accordions(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent", label_text="")
        scroll.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self._bind_mousewheel(scroll)

        for i, lbl in enumerate(LABELS):
            color = LABEL_COLORS[lbl]
            wrapper = ctk.CTkFrame(scroll, fg_color="transparent")
            wrapper.pack(fill="x", pady=(4, 0), padx=2)

            hdr_btn = ctk.CTkButton(
                wrapper, text=f"  {lbl}",
                font=("Poppins", 11, "bold"),
                anchor="w", fg_color="transparent",
                text_color=color,
                hover_color=("e5e7eb", "#252535"),
                border_width=1, border_color=("d1d5db", "#333344"),
                height=32,
                command=lambda idx=i: self._toggle_acc(idx),
            )
            hdr_btn.pack(fill="x")
            self.acc_open_flags.append(hdr_btn)

            body  = ctk.CTkFrame(wrapper, fg_color=("f0f0f5", "#1e1e2e"), corner_radius=6)
            inner = ctk.CTkFrame(body, fg_color="transparent")
            inner.pack(fill="x", padx=8, pady=6)

            self._build_ai_score_bar(inner, lbl, color)
            self._build_prompt_editor(inner, i)
            self._build_threshold_slider(inner, i)

            self.acc_bodies.append(body)

    def _build_ai_score_bar(self, parent, lbl: str, color: str):
        ai_row = ctk.CTkFrame(parent, fg_color="transparent")
        ai_row.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(ai_row, text="AI score",
                     font=("Poppins", 9), text_color="gray").pack(side="left")
        sv = ctk.StringVar(value="—")
        ctk.CTkLabel(
            ai_row, textvariable=sv, font=("Poppins", 9),
            text_color=color, width=34, anchor="e",
        ).pack(side="right")
        self.ai_score_labels[lbl] = sv

        ai_bar_bg = ctk.CTkFrame(parent, fg_color=("d1d5db", "#2a2a3a"), height=4, corner_radius=2)
        ai_bar_bg.pack(fill="x", pady=(2, 6))
        sc = tk.Canvas(ai_bar_bg, height=4, bg="#2a2a3a", highlightthickness=0)
        sc.pack(fill="x")
        self.ai_score_canvases[lbl] = (sc, color)

        ctk.CTkFrame(parent, fg_color=("d1d5db", "#2e2e3e"), height=1).pack(
            fill="x", pady=(0, 6)
        )

    def _build_prompt_editor(self, parent, idx: int):
        ctk.CTkLabel(
            parent, text="Positive prompt — zero-shot",
            font=("Poppins", 9), text_color="#10b981",
        ).pack(anchor="w")
        p_box = ctk.CTkTextbox(parent, height=76, font=("Poppins", 10), wrap="word")
        p_box.insert("1.0", DEFAULT_PROMPT_GROUPS[idx][0])
        p_box.pack(fill="x", pady=(2, 6))
        self.pos_textboxes.append(p_box)

    def _build_threshold_slider(self, parent, idx: int):
        thr_row = ctk.CTkFrame(parent, fg_color="transparent")
        thr_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(thr_row, text="Threshold",
                     font=("Poppins", 9), text_color="gray").pack(side="left")
        thr_lbl_w = ctk.CTkLabel(thr_row, text="0.50",
                                  font=("Poppins", 9), text_color="gray", width=32)
        thr_lbl_w.pack(side="right")
        self.threshold_labels.append(thr_lbl_w)

        def _make_cb(ref=thr_lbl_w):
            def cb(v):
                ref.configure(text=f"{float(v):.2f}")
                self.app._save_current_thresholds()
            return cb

        ctk.CTkSlider(
            thr_row, from_=0.1, to=0.9,
            variable=self.threshold_vars[idx],
            command=_make_cb(),
        ).pack(side="left", fill="x", expand=True, padx=6)

    def _toggle_acc(self, idx: int):
        """
        Expand atau collapse body accordion pada index tertentu.

        Args:
            idx: Index label (0=Boredom, 1=Engagement, 2=Confusion, 3=Frustration).
        """
        body = self.acc_bodies[idx]
        btn  = self.acc_open_flags[idx]
        lbl  = LABELS[idx]
        if body.winfo_ismapped():
            body.pack_forget()
            btn.configure(text=f"  {lbl}")
        else:
            body.pack(fill="x", pady=(2, 0))
            btn.configure(text=f"  {lbl}")

    def update_ai_score_bar(self, label: str, score: float):
        """
        Update bar AI score dan teks nilai numeriknya untuk label tertentu.

        Args:
            label: Nama label (dari LABELS).
            score: Nilai avg_score dari hasil inferensi AI (0.0 - 1.0).
        """
        sc, color = self.ai_score_canvases[label]
        sc.update_idletasks()
        w = sc.winfo_width() or 160
        sc.delete("all")
        fill_w = int(w * score)
        if fill_w > 0:
            sc.create_rectangle(0, 0, fill_w, 4, fill=color, outline="")
        self.ai_score_labels[label].set(f"{score:.2f}")

    def get_prompts_and_thresholds(self) -> tuple:
        """
        Baca nilai prompt dan threshold dari UI untuk dikirim ke inferensi.

        Returns:
            (prompts, thresholds)
            prompts    -- list of (pos_lines, []) per label
            thresholds -- list of float per label
        """
        prompts = []
        for i in range(4):
            p = [
                line.strip()
                for line in self.pos_textboxes[i].get("1.0", "end-1c").split("\n")
                if line.strip()
            ]
            prompts.append((p, []))
        thresholds = [v.get() for v in self.threshold_vars]
        return prompts, thresholds
