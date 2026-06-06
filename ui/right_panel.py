"""
ui/right_panel.py

Panel kanan: tombol inferensi AI, toggle label 0/1, vote bar, AI score bar,
prompt editor per label, threshold slider, dan rules inline accordion.

Struktur layout (atas ke bawah):
    [Proses Video Ini] [Batch Semua]
    Status label batch
    -----------------------------------------
    Scrollable accordion per label:
        [Header accordion -- klik untuk expand]
        [Body: tombol 0/1 | vote bar | AI score bar | prompt | threshold]
    Rules accordion (collapsible, inline)
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
        rules_content          -- RulesContent instance (inline rules panel)
    """

    def __init__(self, parent, app):
        self.app = app
        self.pos_textboxes     = []
        self.threshold_vars    = [ctk.DoubleVar(value=0.50) for _ in LABELS]
        self.ai_score_canvases = {}
        self.ai_score_labels   = {}
        self.acc_bodies           = []
        self.acc_open_flags       = []
        self.threshold_labels     = []
        self.threshold_entry_vars = []  # StringVar per label — selalu sinkron tanpa perlu widget visible

        self._build(parent)

    def _build(self, parent):
        right = ctk.CTkFrame(
            parent, fg_color=("#f0f2f5", "#15151f"), corner_radius=0, width=330
        )
        right.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
        right.pack_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # Single scrollable container for ALL right panel content
        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent", label_text="")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        self._bind_mousewheel(scroll)
        self._main_scroll = scroll

        self._build_action_buttons(scroll)
        self._build_statistics_panel(scroll)
        self._build_label_accordions(scroll)

    # ── Section header helper ─────────────────────────────────────────────────
    @staticmethod
    def _section_label(parent, text: str):
        ctk.CTkLabel(
            parent, text=text,
            font=("Poppins", 8, "bold"), text_color=("#5a5a7a", "#5a5a7a"),
            anchor="w",
        ).pack(anchor="w", padx=14, pady=(8, 2))

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
        # ── Inferensi ─────────────────────────────────────────────────────────
        self._section_label(parent, "INFERENSI")
        infer_card = ctk.CTkFrame(parent, fg_color=("#e8eaef", "#1c1c2a"), corner_radius=10)
        infer_card.pack(fill="x", padx=12, pady=(0, 4))

        top_btns = ctk.CTkFrame(infer_card, fg_color="transparent")
        top_btns.pack(fill="x", padx=10, pady=(10, 4))

        self.btn_proses_satu = ctk.CTkButton(
            top_btns, text="Proses Video Ini",
            command=self.app._proses_satu,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=self.app.font_sm, height=32, corner_radius=8,
        )
        self.btn_proses_satu.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_proses_semua = ctk.CTkButton(
            top_btns, text="Batch Semua",
            command=self.app._toggle_batch,
            fg_color="#10b981", hover_color="#059669",
            font=self.app.font_sm, height=32, corner_radius=8,
        )
        self.btn_proses_semua.pack(side="left", fill="x", expand=True)

        status_row = ctk.CTkFrame(infer_card, fg_color="transparent")
        status_row.pack(fill="x", padx=10, pady=(2, 4))

        self.lbl_batch_status = ctk.CTkLabel(
            status_row, text="Siap", font=("Poppins", 9), text_color="gray", anchor="w",
        )
        self.lbl_batch_status.pack(side="left", fill="x", expand=True)

        self.btn_restart_batch = ctk.CTkButton(
            status_row, text="Restart", command=self.app._restart_batch,
            fg_color="#f59e0b", hover_color="#d97706",
            font=("Poppins", 9), height=24, width=70, corner_radius=6,
        )
        self.btn_restart_batch.pack(side="right")

        # Batch versioning
        bv_frame = ctk.CTkFrame(infer_card, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        bv_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.batch_new_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            bv_frame, text="Batch baru:", variable=self.batch_new_var,
            font=("Poppins", 9), text_color="gray",
            checkbox_width=14, checkbox_height=14,
        ).pack(side="left", padx=(8, 4), pady=6)
        self.batch_name_entry = ctk.CTkEntry(
            bv_frame, placeholder_text="nama file...", font=("Poppins", 9),
            height=24, corner_radius=6, fg_color=("white", "#111122"),
        )
        self.batch_name_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=6)

        # ── Kalibrasi netral per-ORANG (Bosch 2023 + FACS) ────────────────────
        cal_frame = ctk.CTkFrame(infer_card, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        cal_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.lbl_neutral_status = ctk.CTkLabel(
            cal_frame, text="Kalibrasi: buka folder dulu",
            font=("Poppins", 9), text_color="gray", anchor="w",
        )
        self.lbl_neutral_status.pack(fill="x", padx=8, pady=(6, 2))

        # Checkbox: pakai baseline DEFAULT (populasi) — bawaan TIDAK tercentang.
        # Tercentang → tombol proses tidak butuh frame netral (pakai anchor populasi DEFAULT_PYFEAT_CALIB).
        self.use_default_baseline_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            cal_frame, text="Pakai baseline default (populasi)",
            variable=self.use_default_baseline_var,
            command=self.app._on_default_toggle,
            font=("Poppins", 9), text_color="gray",
            checkbox_width=14, checkbox_height=14,
        ).pack(fill="x", padx=8, pady=(0, 4))
        self.btn_mark_neutral = ctk.CTkButton(
            cal_frame, text="Tandai Frame Netral Orang Ini",
            command=self.app._mark_neutral_current,
            fg_color="#0ea5e9", hover_color="#0284c7",
            font=("Poppins", 10), height=28, corner_radius=6,
        )
        self.btn_mark_neutral.pack(fill="x", padx=8, pady=(0, 4))

        # Navigasi antar-ORANG (loncat ke video pertama orang berikutnya/sebelumnya)
        nav_row = ctk.CTkFrame(cal_frame, fg_color="transparent")
        nav_row.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_prev_person = ctk.CTkButton(
            nav_row, text="◀ Orang Sebelumnya", command=self.app._prev_person,
            fg_color="#475569", hover_color="#334155",
            font=("Poppins", 9), height=26, corner_radius=6,
        )
        self.btn_prev_person.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_next_person = ctk.CTkButton(
            nav_row, text="Orang Berikutnya ▶", command=self.app._next_person,
            fg_color="#0d9488", hover_color="#0f766e",
            font=("Poppins", 9), height=26, corner_radius=6,
        )
        self.btn_next_person.pack(side="left", fill="x", expand=True)

        # ── Label & Dataset ───────────────────────────────────────────────────
        self._section_label(parent, "LABEL & DATASET")
        dataset_card = ctk.CTkFrame(parent, fg_color=("#e8eaef", "#1c1c2a"), corner_radius=10)
        dataset_card.pack(fill="x", padx=12, pady=(0, 4))

        self.btn_reset_label = ctk.CTkButton(
            dataset_card, text="Reset Label Video Ini", command=self.app.reset_current_labels,
            fg_color="#b91c1c", hover_color="#991b1b",
            font=("Poppins", 9), height=26, corner_radius=7,
        )
        self.btn_reset_label.pack(fill="x", padx=10, pady=(10, 6))

        # Split Label 2D
        split_hdr = ctk.CTkFrame(dataset_card, fg_color="transparent")
        split_hdr.pack(fill="x", padx=10, pady=(0, 4))
        self.btn_split_ai = ctk.CTkButton(
            split_hdr, text="Split AI", command=lambda: self.app._split_dataset_2d(source="ai"),
            fg_color="#0ea5e9", hover_color="#0284c7",
            font=("Poppins", 9, "bold"), height=28, corner_radius=7,
        )
        self.btn_split_ai.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_split_manual = ctk.CTkButton(
            split_hdr, text="Split Manual", command=lambda: self.app._split_dataset_2d(source="manual"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            font=("Poppins", 9, "bold"), height=28, corner_radius=7,
        )
        self.btn_split_manual.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkLabel(split_hdr, text="UUID:", font=("Poppins", 8), text_color="gray").pack(side="left")
        self.split_uuid_depth_entry = ctk.CTkEntry(
            split_hdr, width=30, height=28, font=("Poppins", 9), corner_radius=6,
        )
        self.split_uuid_depth_entry.insert(0, "2")
        self.split_uuid_depth_entry.pack(side="left", padx=(2, 0))

        sv_frame = ctk.CTkFrame(dataset_card, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        sv_frame.pack(fill="x", padx=10, pady=(0, 4))
        self.split_new_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sv_frame, text="Folder baru:", variable=self.split_new_var,
            font=("Poppins", 9), text_color="gray",
            checkbox_width=14, checkbox_height=14,
        ).pack(side="left", padx=(8, 4), pady=5)
        ctk.CTkLabel(
            sv_frame, text="nama batch di atas",
            font=("Poppins", 8), text_color="#6b7280",
        ).pack(side="left", padx=(0, 6), pady=5)

        self.btn_compare = ctk.CTkButton(
            dataset_card, text="Bandingkan AI vs Manual",
            command=self.app._compare_ai_vs_manual,
            fg_color="transparent", border_width=1, border_color="#7c3aed",
            text_color="#7c3aed", hover_color=("#ede9fe", "#2e1a4a"),
            font=("Poppins", 9), height=26, corner_radius=7,
        )
        self.btn_compare.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkButton(
            dataset_card, text="Parameter & Rules",
            command=self.app._open_rules_panel,
            fg_color="transparent", border_width=1, border_color=("#a78bfa", "#6d28d9"),
            text_color=("#7c3aed", "#a78bfa"), hover_color=("#ede9fe", "#2e1a4a"),
            font=("Poppins", 9), height=26, corner_radius=7,
        ).pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkFrame(parent, fg_color=("#c8ccd6", "#2e2e3e"), height=1).pack(
            fill="x", padx=12, pady=(4, 6)
        )

    def _build_statistics_panel(self, parent):
        import os
        self._section_label(parent, "STATISTIK")
        stats_frame = ctk.CTkFrame(parent, fg_color=("#e8eaef", "#1c1c2a"), corner_radius=10)
        stats_frame.pack(fill="x", padx=12, pady=(0, 6))

        hdr = ctk.CTkFrame(stats_frame, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(hdr, text="Hasil AI", font=("Poppins", 10, "bold"), text_color=("#4b5563", "#9ca3af")).pack(side="left")

        # ── Total diterima + ditolak dalam satu baris ─────────────────────────
        summary_row = ctk.CTkFrame(stats_frame, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        summary_row.pack(fill="x", padx=10, pady=(4, 6))
        summary_row.columnconfigure(0, weight=1)
        summary_row.columnconfigure(1, weight=1)

        acc_cell = ctk.CTkFrame(summary_row, fg_color="transparent")
        acc_cell.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=6)
        self.lbl_stat_total = ctk.CTkLabel(
            acc_cell, text="Diterima: 0",
            font=("Poppins", 9, "bold"), text_color="#10b981",
        )
        self.lbl_stat_total.pack(side="left", padx=(2, 0))

        rej_cell = ctk.CTkFrame(summary_row, fg_color="transparent")
        rej_cell.grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=6)
        self.lbl_stat_rejected = ctk.CTkLabel(
            rej_cell, text="Ditolak: 0",
            font=("Poppins", 9, "bold"), text_color="#ef4444",
        )
        self.lbl_stat_rejected.pack(side="left", padx=(2, 0))

        # ── Batch file dropdown ───────────────────────────────────────────────
        brow = ctk.CTkFrame(stats_frame, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        brow.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(brow, text="File:", font=("Poppins", 8), text_color="gray").pack(side="left", padx=(8, 3), pady=5)

        self._batch_file_var = ctk.StringVar(value="batch_history.json")
        self._batch_file_menu = ctk.CTkOptionMenu(
            brow, variable=self._batch_file_var,
            values=["batch_history.json"],
            width=160, height=24,
            font=("Poppins", 8),
            command=self._on_batch_file_select,
        )
        self._batch_file_menu.pack(side="left", fill="x", expand=True, padx=(0, 2), pady=5)

        ctk.CTkButton(
            brow, text="Refresh", width=52, height=24,
            font=("Poppins", 9), fg_color="transparent", corner_radius=6,
            text_color=("gray", "#6b7280"), hover_color=("#e5e7eb", "#374151"),
            command=self.refresh_batch_files,
        ).pack(side="left", padx=(0, 6), pady=5)

        # ── Label count grid ──────────────────────────────────────────────────
        grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=(0, 8))

        self.stat_labels = {}
        for i, lbl in enumerate(LABELS):
            color = LABEL_COLORS[lbl]
            row = i // 2
            col = i % 2
            var_label = ctk.CTkLabel(grid, text=f"{lbl}: 0", font=("Poppins", 9), text_color=color)
            var_label.grid(row=row, column=col, sticky="w", padx=(0, 15), pady=2)
            self.stat_labels[lbl] = var_label

        # ── Separator ─────────────────────────────────────────────────────────
        ctk.CTkFrame(stats_frame, fg_color=("#c0c4d0", "#2e2e3e"), height=1).pack(
            fill="x", padx=10, pady=(0, 4)
        )

        # ── Manual section header ─────────────────────────────────────────────
        mhdr = ctk.CTkFrame(stats_frame, fg_color="transparent")
        mhdr.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(mhdr, text="Hasil Manual", font=("Poppins", 10, "bold"), text_color=("#4b5563", "#9ca3af")).pack(side="left")

        # ── Manual diterima + ditolak ─────────────────────────────────────────
        msummary_row = ctk.CTkFrame(stats_frame, fg_color=("#d8dce6", "#222232"), corner_radius=8)
        msummary_row.pack(fill="x", padx=10, pady=(0, 6))
        msummary_row.columnconfigure(0, weight=1)
        msummary_row.columnconfigure(1, weight=1)

        macc_cell = ctk.CTkFrame(msummary_row, fg_color="transparent")
        macc_cell.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=6)
        self.lbl_manual_stat_total = ctk.CTkLabel(
            macc_cell, text="Diterima: 0",
            font=("Poppins", 9, "bold"), text_color="#10b981",
        )
        self.lbl_manual_stat_total.pack(side="left", padx=(2, 0))

        mrej_cell = ctk.CTkFrame(msummary_row, fg_color="transparent")
        mrej_cell.grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=6)
        self.lbl_manual_stat_rejected = ctk.CTkLabel(
            mrej_cell, text="Ditolak: 0",
            font=("Poppins", 9, "bold"), text_color="#ef4444",
        )
        self.lbl_manual_stat_rejected.pack(side="left", padx=(2, 0))

        # ── Manual label count grid ───────────────────────────────────────────
        mgrid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        mgrid.pack(fill="x", padx=10, pady=(0, 8))

        self.manual_stat_labels = {}
        for i, lbl in enumerate(LABELS):
            color = LABEL_COLORS[lbl]
            row = i // 2
            col = i % 2
            var_label = ctk.CTkLabel(mgrid, text=f"{lbl}: 0", font=("Poppins", 9), text_color=color)
            var_label.grid(row=row, column=col, sticky="w", padx=(0, 15), pady=2)
            self.manual_stat_labels[lbl] = var_label

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
        """Load batch history yang dipilih, restore rules + threshold + gallery dari __meta__."""
        import os
        bh_path = getattr(self.app, "path_json_batch_history", "")
        if not bh_path:
            return
        full_path = os.path.join(os.path.dirname(bh_path), filename)
        try:
            from utils import load_batch_history, load_batch_meta
            bh   = load_batch_history(full_path)
            meta = load_batch_meta(full_path)

            # Ganti batch aktif di app
            self.app.batch_history = bh
            self.update_statistics(bh)

            # ── Restore thresholds ──────────────────────────────────────────
            thresholds = meta.get("thresholds") or (
                next(iter(bh.values()), {}).get("thresholds") if bh else None
            )
            if thresholds and len(thresholds) == len(self.threshold_vars):
                for i, val in enumerate(thresholds):
                    self.threshold_vars[i].set(val)
                    if i < len(self.threshold_entry_vars):
                        self.threshold_entry_vars[i].set(f"{float(val):.2f}")

            # ── Restore rules dari __meta__ ─────────────────────────────────
            saved_rules = meta.get("rules")
            if saved_rules:
                self.app._save_rules(saved_rules)  # update self.app.rules + tulis rules.json
                # Sync inline rules content di left panel
                lp_rc = getattr(self.app.left_panel, 'rules_content', None)
                if lp_rc is not None:
                    lp_rc._load_from_rules(saved_rules)
                self.app._rules_panel = None
                print(f"[BatchSwitch] Rules di-restore dari {filename}")

            # ── Invalidasi gallery + minta regenerasi viz dengan rules baru ─
            self.app._viz_regen_requested = True
            self.app._gallery_cache["rel_path"] = None
            self.app.refresh_frame_gallery()

        except Exception as exc:
            print(f"[Stats] Gagal load {filename}: {exc}")

    def update_statistics(self, batch_history: dict):
        total_frames   = 0
        rejected_frames = 0
        counts = {lbl: 0 for lbl in LABELS}

        fa = getattr(self.app, "frame_annotations", {})

        for rel_path, vid_data in batch_history.items():
            per_label = vid_data.get("per_label", {})
            fa_vid    = fa.get(rel_path, {})

            frame_preds_0 = per_label.get("0", {}).get("frame_preds", [])
            n_frames = len(frame_preds_0)

            for f_idx in range(n_frames):
                if fa_vid.get(str(f_idx), {}).get("_rejected", False):
                    rejected_frames += 1
                    continue
                total_frames += 1
                for i, lbl in enumerate(LABELS):
                    preds = per_label.get(str(i), {}).get("frame_preds", [])
                    if f_idx < len(preds) and preds[f_idx] == 1:
                        counts[lbl] += 1

        self.lbl_stat_total.configure(text=f"Diterima: {total_frames}")
        self.lbl_stat_rejected.configure(text=f"Ditolak: {rejected_frames}")

        for lbl in LABELS:
            self.stat_labels[lbl].configure(text=f"{lbl}: {counts[lbl]}")

    def update_manual_statistics(self, manual_labels: dict):
        """Update panel statistik Manual dari manual_labels dict.

        _rejected dibaca dari frame_annotations (sumber otoritatif), bukan dari manual_labels.
        Entry manual_labels yang tidak ada di frame_annotations (stale) dilewati.
        """
        total_frames    = 0
        rejected_frames = 0
        counts = {lbl: 0 for lbl in LABELS}

        fa = getattr(self.app, "frame_annotations", {})

        for rel_path, frames in manual_labels.items():
            # Lewati entri stale yang tidak ada di frame_annotations aktif
            if rel_path not in fa:
                continue
            for f_key, fdata in frames.items():
                if not f_key.isdigit():
                    continue
                # _rejected disimpan di manual_labels (terpisah dari AI frame_annotations)
                if fdata.get("_rejected", False):
                    rejected_frames += 1
                    continue
                total_frames += 1
                for lbl in LABELS:
                    if fdata.get(lbl, 0) == 1:
                        counts[lbl] += 1

        self.lbl_manual_stat_total.configure(text=f"Diterima: {total_frames}")
        self.lbl_manual_stat_rejected.configure(text=f"Ditolak: {rejected_frames}")
        for lbl in LABELS:
            self.manual_stat_labels[lbl].configure(text=f"{lbl}: {counts[lbl]}")

    def _build_label_accordions(self, parent):
        self._section_label(parent, "LABEL EMOSI")

        for i, lbl in enumerate(LABELS):
            color = LABEL_COLORS[lbl]
            wrapper = ctk.CTkFrame(parent, fg_color=("#e8eaef", "#1a1a28"), corner_radius=10)
            wrapper.pack(fill="x", pady=(0, 6), padx=12)

            hdr_btn = ctk.CTkButton(
                wrapper, text=f"  {lbl}",
                font=("Poppins", 11, "bold"),
                anchor="w", fg_color="transparent",
                text_color=color,
                hover_color=("#dde0e8", "#23233a"),
                corner_radius=10,
                height=36,
                command=lambda idx=i: self._toggle_acc(idx),
            )
            hdr_btn.pack(fill="x")
            self.acc_open_flags.append(hdr_btn)

            body  = ctk.CTkFrame(wrapper, fg_color=("#e0e3ea", "#1e1e2e"), corner_radius=8)
            inner = ctk.CTkFrame(body, fg_color="transparent")
            inner.pack(fill="x", padx=10, pady=8)

            self._build_ai_score_bar(inner, lbl, color)
            self._build_prompt_editor(inner, i)
            self._build_threshold_slider(inner, i)

            self.acc_bodies.append(body)


    def _build_ai_score_bar(self, parent, lbl: str, color: str):
        # 2 bar per-frame (F0 dan F1) menggantikan 1 bar avg
        frame_canvases = []
        frame_labels   = []
        for fi in range(2):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=(0, 1))
            ctk.CTkLabel(row, text=f"F{fi}",
                         font=("Poppins", 9), text_color=("#9ca3af", "#6b7280"), width=20).pack(side="left")
            sv = ctk.StringVar(value="--")
            ctk.CTkLabel(
                row, textvariable=sv, font=("Poppins", 9, "bold"),
                text_color=color, width=36, anchor="e",
            ).pack(side="right")
            frame_labels.append(sv)

            bar_bg = ctk.CTkFrame(parent, fg_color=("#c8ccd6", "#252535"), height=7, corner_radius=3)
            bar_bg.pack(fill="x", pady=(1, 4))
            sc = tk.Canvas(bar_bg, height=7, bg="#252535", highlightthickness=0)
            sc.pack(fill="x")
            frame_canvases.append((sc, color))

        self.ai_score_labels[lbl]   = frame_labels
        self.ai_score_canvases[lbl] = frame_canvases

        ctk.CTkFrame(parent, fg_color=("#d0d3dc", "#2e2e3e"), height=1).pack(
            fill="x", pady=(2, 8)
        )

    def _build_prompt_editor(self, parent, idx: int):
        ctk.CTkLabel(
            parent, text="Positive prompt -- zero-shot",
            font=("Poppins", 8, "bold"), text_color=("#6b7280", "#6b7280"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 2))
        p_box = ctk.CTkTextbox(
            parent, height=76, font=("Poppins", 10), wrap="word", corner_radius=8,
        )
        p_box.insert("1.0", DEFAULT_PROMPT_GROUPS[idx][0])
        p_box.pack(fill="x", pady=(0, 8))
        self.pos_textboxes.append(p_box)

    def _build_threshold_slider(self, parent, idx: int):
        import tkinter as _tk
        thr_row = ctk.CTkFrame(parent, fg_color="transparent")
        thr_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(thr_row, text="Threshold",
                     font=("Poppins", 9), text_color=("#6b7280", "#6b7280")).pack(side="left")

        # StringVar sebagai sumber kebenaran teks entry — selalu bisa di-set
        # tanpa peduli apakah widget sedang visible atau tidak (fix accordion lazy-show).
        entry_var = _tk.StringVar(value=f"{self.threshold_vars[idx].get():.2f}")
        thr_entry = ctk.CTkEntry(thr_row, width=48, height=22,
                                  font=("Poppins", 9), justify="center",
                                  textvariable=entry_var)
        thr_entry.pack(side="right", padx=(4, 0))
        self.threshold_labels.append(thr_entry)
        self.threshold_entry_vars.append(entry_var)

        def _on_slider(v, sv=entry_var):
            sv.set(f"{float(v):.2f}")
            self.app._save_current_thresholds()

        def _on_entry(event, var=self.threshold_vars[idx], sv=entry_var):
            try:
                val = float(sv.get())
                val = max(0.05, min(0.95, val))
                var.set(val)
                sv.set(f"{val:.2f}")
                self.app._save_current_thresholds()
            except ValueError:
                sv.set(f"{var.get():.2f}")

        thr_entry.bind("<Return>",    _on_entry)
        thr_entry.bind("<FocusOut>",  _on_entry)

        ctk.CTkSlider(
            thr_row, from_=0.05, to=0.95,
            variable=self.threshold_vars[idx],
            command=_on_slider,
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

    def update_ai_score_bar(self, label: str, frame_scores: list, threshold: float = None):
        """
        Update 2 bar AI score per-frame untuk label tertentu.

        Args:
            label:        Nama label (dari LABELS).
            frame_scores: List skor per-frame [score_f0, score_f1].
            threshold:    Opsional -- garis threshold ditampilkan di bar.
        """
        canvases = self.ai_score_canvases[label]
        labels   = self.ai_score_labels[label]
        for fi, (sc, color) in enumerate(canvases):
            score = frame_scores[fi] if fi < len(frame_scores) else 0.0
            sc.update_idletasks()
            w = sc.winfo_width() or 160
            sc.delete("all")
            fill_w = int(w * score)
            if fill_w > 0:
                sc.create_rectangle(0, 0, fill_w, 7, fill=color, outline="")
            if threshold is not None:
                tx = int(w * threshold)
                sc.create_line(tx, 0, tx, 7, fill="#ffffff", width=1)
            labels[fi].set(f"{score:.2f}" if score > 0 else "--")

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
