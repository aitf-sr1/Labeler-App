"""
ui/rules_panel.py

Toplevel window untuk mengedit semua parameter rules (landmark scoring + hybrid weights).
Dibuka via tombol "Rules" di aplikasi utama.
"""

import json
import tkinter as tk
import customtkinter as ctk

from core.rules import DEFAULT_RULES, _deep_copy

# ── Definisi slider per seksi ─────────────────────────────────────────────────
# (section, key, label, min_val, max_val)
_SLIDER_DEFS = [
    # GAZE (shared)
    ("gaze", "scale_h",        "H Scale (iris→°)",    10.0, 60.0),
    ("gaze", "scale_v",        "V Scale (iris→°)",    10.0, 50.0),
    ("gaze", "iris_side_mult", "Iris Side Mult",       0.5,  4.0),
    ("gaze", "v_dead_zone",    "V Dead Zone (°)",      0.0, 30.0),
    # BOREDOM
    ("boredom", "gaze_dead_zone",     "Gaze Dead Zone (°)",    0.0, 20.0),
    ("boredom", "gaze_range",         "Gaze Range (°)",        4.0, 25.0),
    ("boredom", "pitch_nunduk_th",    "Pitch Nunduk Th (°)",   5.0, 25.0),
    ("boredom", "pitch_nunduk_range", "Nunduk Range (°)",      3.0, 20.0),
    ("boredom", "pitch_up_th",        "Pitch Up Th (°)",       3.0, 20.0),
    ("boredom", "pitch_up_range",     "Pitch Up Range (°)",    5.0, 20.0),
    ("boredom", "blink_dead_zone",    "Blink Dead Zone",       0.0,  0.5),
    ("boredom", "blink_range",        "Blink Range",           0.1,  1.0),
    ("boredom", "yawn_threshold",     "Yawn Threshold",        0.1,  0.8),
    ("boredom", "sig_expr_weight",    "Expr Weight",           0.0,  1.0),
    ("boredom", "frus_suppress",      "Frus Suppress",         0.0,  1.0),
    # ENGAGEMENT
    ("engagement", "nunduk_gate_range", "Nunduk Gate Range",   2.0, 15.0),
    ("engagement", "tegak_dead_zone",   "Tegak Dead Zone (°)", 0.0, 10.0),
    ("engagement", "tegak_range",       "Tegak Range (°)",     5.0, 25.0),
    ("engagement", "blink_heavy_th",    "Heavy Blink Th",      0.2,  0.9),
    ("engagement", "blink_heavy_min",   "Min Engagement",      0.0,  0.5),
    # CONFUSION
    ("confusion", "iris_up_dead_zone",  "Iris Up Dead Zone",   0.0,  0.4),
    ("confusion", "iris_up_range",      "Iris Up Range",       0.1,  0.6),
    ("confusion", "look_up_threshold",  "LookUp Threshold",    0.1,  0.7),
    ("confusion", "pitch_start",        "Pitch Start (°)",     0.0, 15.0),
    ("confusion", "pitch_range",        "Pitch Range (°)",     5.0, 30.0),
    ("confusion", "brow_dn_th",         "BrowDown Th",         0.1,  0.5),
    ("confusion", "brow_in_th",         "BrowInnerUp Th",      0.1,  0.5),
    ("confusion", "brow_in_co_gate",    "BrowInner CoGate",    0.1,  0.5),
    ("confusion", "smile_penalty_th",   "Smile Penalty Th",    0.0,  0.3),
    ("confusion", "jaw_start",          "Jaw Start",           0.0,  0.2),
    ("confusion", "jaw_peak",           "Jaw Peak",            0.1,  0.5),
    ("confusion", "jaw_end",            "Jaw End",             0.3,  0.8),
    ("confusion", "pucker_th",          "Pucker Th",           0.1,  0.6),
    ("confusion", "pucker_gate_th",     "Pucker Gate Th",      0.1,  0.6),
    ("confusion", "gaze_gate_dead",     "Gaze Gate Dead (°)",  0.0, 15.0),
    ("confusion", "gaze_gate_range",    "Gaze Gate Range (°)", 5.0, 25.0),
    # FRUSTRATION
    ("frustration", "brow_dn_th",       "BrowDown Th",         0.1,  0.6),
    ("frustration", "nose_sneer_th",    "NoseSneer Th",        0.05, 0.5),
    ("frustration", "cheek_squint_th",  "CheekSquint Th",      0.1,  0.6),
    ("frustration", "mouth_press_th",   "MouthPress Th",       0.1,  0.6),
    ("frustration", "eye_squint_th",    "EyeSquint Th",        0.1,  0.6),
    ("frustration", "jaw_start",        "Jaw Start",           0.0,  0.3),
    ("frustration", "jaw_range",        "Jaw Range",           0.05, 0.5),
    ("frustration", "restless_dead",    "Restless Dead (°)",   0.0, 15.0),
    ("frustration", "restless_range",   "Restless Range (°)",  3.0, 20.0),
    ("frustration", "restless_w",       "Restless Weight",     0.0,  1.0),
    # HYBRID
    ("hybrid", "empirical_bias",       "SigLIP Empirical Bias", 1.0, 6.0),
    ("hybrid", "restless_bonus_max",   "Restless Bonus Max",    0.0, 0.3),
    ("hybrid", "restless_std_min",     "Restless Std Min (°)",  1.0, 8.0),
    ("hybrid", "restless_std_range",   "Restless Std Range (°)", 3.0, 15.0),
]

_SECTION_COLORS = {
    "gaze":        "#9ca3af",
    "boredom":     "#fbbf24",
    "engagement":  "#10b981",
    "confusion":   "#3b82f6",
    "frustration": "#ef4444",
    "hybrid":      "#8b5cf6",
}

_SECTION_LABELS = {
    "gaze":        "Gaze (shared)",
    "boredom":     "Boredom",
    "engagement":  "Engagement",
    "confusion":   "Confusion",
    "frustration": "Frustration",
    "hybrid":      "Hybrid Weights & SigLIP",
}

_LABEL_NAMES = ["Boredom", "Engagement", "Confusion", "Frustration"]


class RulesPanel:
    """
    Toplevel window untuk edit rules landmark + hybrid weights.

    Callback:
        on_save(rules: dict)       — dipanggil saat Save ditekan.
        on_recalculate(rules: dict) — dipanggil saat Recalculate ditekan.
    """

    def __init__(self, parent, rules: dict, threshold_vars=None, on_save=None, on_recalculate=None):
        self._rules_ref = _deep_copy(rules)  # working copy
        self._threshold_vars = threshold_vars or []   # shared DoubleVar dari right panel
        self._on_save = on_save
        self._on_recalculate = on_recalculate

        self.win = ctk.CTkToplevel(parent)
        self.win.title("Rules Editor — Parameter Landmark & Hybrid")
        self.win.geometry("560x720")
        self.win.minsize(480, 500)
        self.win.lift()
        self.win.focus_force()
        # grab_set() didelay agar window sudah fully rendered sebelum grab
        self.win.after(150, self.win.grab_set)

        self._vars: dict = {}       # (section, key) → DoubleVar or list of DoubleVar
        self._lbl_vars: dict = {}   # (section, key) → StringVar for display
        self._build()
        self._load_from_rules(self._rules_ref)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        # Main layout: scroll area + batch strip + button bar
        self.win.rowconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=0)  # batch mode strip
        self.win.rowconfigure(2, weight=0)  # button bar
        self.win.columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self.win, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))

        self._build_slider_sections(scroll)
        self._build_hybrid_weight_section(scroll)
        self._build_threshold_section(scroll)

        # ── Batch mode strip ─────────────────────────────────────────────────
        bstrip = ctk.CTkFrame(self.win, fg_color=("#d1d5db", "#2a2a3a"), corner_radius=0, height=34)
        bstrip.grid(row=1, column=0, sticky="ew")
        bstrip.grid_propagate(False)

        ctk.CTkLabel(
            bstrip, text="Simpan ke:", font=("Poppins", 9),
            text_color=("#374151", "#9ca3af"),
        ).pack(side="left", padx=(10, 4), pady=6)

        self._batch_new_var = ctk.BooleanVar(value=False)
        self._batch_name_var = ctk.StringVar(value="")

        self._batch_name_entry = ctk.CTkEntry(
            bstrip, textvariable=self._batch_name_var,
            placeholder_text="nama batch baru…",
            width=140, height=22, font=("Poppins", 9),
            state="disabled",
        )

        def _toggle_batch_new():
            if self._batch_new_var.get():
                self._batch_name_entry.configure(state="normal")
                self._batch_name_entry.pack(side="left", padx=(0, 6))
            else:
                self._batch_name_entry.pack_forget()

        ctk.CTkCheckBox(
            bstrip, text="Buat batch baru", variable=self._batch_new_var,
            font=("Poppins", 9), width=120, height=22,
            command=_toggle_batch_new,
        ).pack(side="left", padx=(0, 6), pady=6)

        self._batch_name_entry.pack_forget()  # hidden until checkbox checked

        # ── Button bar ───────────────────────────────────────────────────────
        bar = ctk.CTkFrame(self.win, fg_color=("#e5e7eb", "#1f1f1f"), corner_radius=0, height=50)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkButton(
            bar, text="Reset Default", width=120, height=32,
            fg_color=("#9ca3af", "#374151"), hover_color=("#6b7280", "#1f2937"),
            font=("Poppins", 10), command=self._reset_default,
        ).pack(side="left", padx=(12, 6), pady=9)

        ctk.CTkButton(
            bar, text="Simpan", width=100, height=32,
            fg_color="#3b82f6", hover_color="#2563eb",
            font=("Poppins", 10, "bold"), command=self._save,
        ).pack(side="right", padx=(6, 12), pady=9)

        ctk.CTkButton(
            bar, text="Recalculate Batch", width=150, height=32,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=("Poppins", 10, "bold"), command=self._recalculate,
        ).pack(side="right", padx=6, pady=9)

        self.lbl_status = ctk.CTkLabel(
            bar, text="", font=("Poppins", 9), text_color="#10b981"
        )
        self.lbl_status.pack(side="right", padx=6)

    def _build_slider_sections(self, parent):
        """Bangun slider per parameter, dikelompokkan per seksi."""
        sections_seen = []
        sliders_by_section: dict = {}
        for sec, key, lbl, lo, hi in _SLIDER_DEFS:
            if sec == "hybrid" and key in ("siglip_w", "land_w"):
                continue  # handled separately
            sliders_by_section.setdefault(sec, []).append((key, lbl, lo, hi))
            if sec not in sections_seen:
                sections_seen.append(sec)

        for sec in sections_seen:
            color = _SECTION_COLORS.get(sec, "#6b7280")
            hdr = ctk.CTkLabel(
                parent, text=f"  {_SECTION_LABELS[sec]}",
                font=("Poppins", 10, "bold"), text_color=color,
                anchor="w",
            )
            hdr.pack(fill="x", padx=4, pady=(10, 2))

            divider = ctk.CTkFrame(parent, fg_color=color, height=1, corner_radius=0)
            divider.pack(fill="x", padx=4, pady=(0, 4))

            for key, lbl, lo, hi in sliders_by_section[sec]:
                self._build_one_slider(parent, sec, key, lbl, lo, hi)

    def _build_one_slider(self, parent, sec, key, lbl, lo, hi):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=1)

        ctk.CTkLabel(row, text=lbl, font=("Poppins", 9), width=155, anchor="w").pack(side="left")

        var = ctk.DoubleVar(value=0.0)
        is_fine = (hi - lo) < 2
        fmt = "{:.3f}" if is_fine else "{:.2f}"
        ev = ctk.StringVar(value="0.00")
        self._vars[(sec, key)]     = var
        self._lbl_vars[(sec, key)] = ev

        entry = ctk.CTkEntry(
            row, textvariable=ev, width=58, height=22,
            font=("Poppins", 9, "bold"), justify="right",
            fg_color=("white", "#111122"),
        )
        entry.pack(side="right", padx=(2, 0))

        def _on_slide(v, _ev=ev, _fmt=fmt):
            _ev.set(_fmt.format(float(v)))

        def _commit(_ev=ev, _var=var, _lo=lo, _hi=hi, _fmt=fmt):
            try:
                v = float(_ev.get().replace(",", "."))
                v = max(_lo, min(_hi, v))
                _var.set(v)
                _ev.set(_fmt.format(v))
            except ValueError:
                _ev.set(_fmt.format(_var.get()))

        slider = ctk.CTkSlider(row, from_=lo, to=hi, variable=var, command=_on_slide)
        slider.pack(side="left", fill="x", expand=True, padx=(4, 4))
        entry.bind("<Return>", lambda e: _commit())
        entry.bind("<FocusOut>", lambda e: _commit())

    def _build_hybrid_weight_section(self, parent):
        """Bangun slider siglip_w dan land_w per label."""
        color = _SECTION_COLORS["hybrid"]
        hdr = ctk.CTkLabel(
            parent, text="  Hybrid Weights (per label)",
            font=("Poppins", 10, "bold"), text_color=color, anchor="w",
        )
        hdr.pack(fill="x", padx=4, pady=(10, 2))
        ctk.CTkFrame(parent, fg_color=color, height=1, corner_radius=0).pack(fill="x", padx=4, pady=(0, 4))

        # Also include scalar hybrid params from _SLIDER_DEFS
        for sec, key, lbl, lo, hi in _SLIDER_DEFS:
            if sec == "hybrid" and key not in ("siglip_w", "land_w"):
                self._build_one_slider(parent, sec, key, lbl, lo, hi)

        # Per-label weight rows
        from ui.constants import LABEL_COLORS as LC
        self._weight_vars: list = []  # list of (siglip_var, land_var, siglip_sv, land_sv, siglip_slider, land_slider)

        for i, lbl_name in enumerate(_LABEL_NAMES):
            lcolor = LC.get(lbl_name, "#6b7280")
            sec_frame = ctk.CTkFrame(parent, fg_color=("f0f0f5", "#1e1e2e"), corner_radius=6)
            sec_frame.pack(fill="x", padx=6, pady=3)

            ctk.CTkLabel(sec_frame, text=lbl_name, font=("Poppins", 9, "bold"),
                         text_color=lcolor).pack(anchor="w", padx=8, pady=(4, 0))

            sv_var = ctk.DoubleVar(value=0.5)
            lv_var = ctk.DoubleVar(value=0.5)
            sv_sv  = ctk.StringVar(value="0.50")
            lv_sv  = ctk.StringVar(value="0.50")

            def _mk_cb(sv, _sv):
                def cb(v):
                    _sv.set(f"{float(v):.2f}")
                return cb

            def _mk_commit(dv, _sv, lo=0.0, hi=1.0):
                def commit():
                    try:
                        v = float(_sv.get().replace(",", "."))
                        v = max(lo, min(hi, v))
                        dv.set(v)
                        _sv.set(f"{v:.2f}")
                    except ValueError:
                        _sv.set(f"{dv.get():.2f}")
                return commit

            for label_txt, dv, sv_str in [("SigLIP", sv_var, sv_sv), ("Landmark", lv_var, lv_sv)]:
                rw = ctk.CTkFrame(sec_frame, fg_color="transparent")
                rw.pack(fill="x", padx=8, pady=1)
                ctk.CTkLabel(rw, text=label_txt, font=("Poppins", 9), width=60, anchor="w").pack(side="left")
                ent = ctk.CTkEntry(rw, textvariable=sv_str, width=52, height=20,
                                   font=("Poppins", 9, "bold"), justify="right",
                                   fg_color=("white", "#111122"))
                ent.pack(side="right", padx=(2, 0))
                commit_fn = _mk_commit(dv, sv_str)
                ent.bind("<Return>", lambda e, fn=commit_fn: fn())
                ent.bind("<FocusOut>", lambda e, fn=commit_fn: fn())
                sl = ctk.CTkSlider(rw, from_=0.0, to=1.0, variable=dv,
                                   command=_mk_cb(dv, sv_str))
                sl.pack(side="left", fill="x", expand=True, padx=(4, 4))

            ctk.CTkFrame(sec_frame, fg_color="transparent", height=2).pack()
            self._weight_vars.append((sv_var, lv_var, sv_sv, lv_sv))

    def _build_threshold_section(self, parent):
        """Slider threshold per label — share DoubleVar dengan right panel."""
        if not self._threshold_vars:
            return

        from ui.constants import LABEL_COLORS as LC
        color = "#f59e0b"

        hdr = ctk.CTkLabel(
            parent, text="  Threshold per Label",
            font=("Poppins", 10, "bold"), text_color=color, anchor="w",
        )
        hdr.pack(fill="x", padx=4, pady=(10, 2))
        ctk.CTkFrame(parent, fg_color=color, height=1, corner_radius=0).pack(fill="x", padx=4, pady=(0, 4))

        note = ctk.CTkLabel(
            parent,
            text="Mengubah threshold di sini langsung mengubah slider di panel utama.",
            font=("Poppins", 8), text_color="gray", anchor="w",
        )
        note.pack(fill="x", padx=8, pady=(0, 4))

        for i, (lbl_name, dv) in enumerate(zip(_LABEL_NAMES, self._threshold_vars)):
            lcolor = LC.get(lbl_name, "#6b7280")
            rw = ctk.CTkFrame(parent, fg_color=("f0f0f5", "#1e1e2e"), corner_radius=6)
            rw.pack(fill="x", padx=6, pady=2)

            ctk.CTkLabel(rw, text=lbl_name, font=("Poppins", 9, "bold"),
                         text_color=lcolor, width=90, anchor="w").pack(side="left", padx=(8, 4), pady=4)

            sv = ctk.StringVar(value=f"{dv.get():.2f}")

            def _mk_slide_cb(_sv=sv):
                def cb(v): _sv.set(f"{float(v):.2f}")
                return cb

            def _mk_commit_fn(_dv=dv, _sv=sv):
                def commit():
                    try:
                        v = max(0.1, min(0.9, float(_sv.get().replace(",", "."))))
                        _dv.set(v); _sv.set(f"{v:.2f}")
                    except ValueError:
                        _sv.set(f"{_dv.get():.2f}")
                return commit

            ent = ctk.CTkEntry(rw, textvariable=sv, width=52, height=22,
                               font=("Poppins", 9, "bold"), justify="right",
                               fg_color=("white", "#111122"))
            ent.pack(side="right", padx=(2, 8))
            commit_fn = _mk_commit_fn()
            ent.bind("<Return>", lambda e, fn=commit_fn: fn())
            ent.bind("<FocusOut>", lambda e, fn=commit_fn: fn())

            ctk.CTkSlider(
                rw, from_=0.1, to=0.9, variable=dv,
                command=_mk_slide_cb(),
            ).pack(side="left", fill="x", expand=True, padx=(4, 4), pady=4)

    # ── Load / collect ────────────────────────────────────────────────────────

    def _load_from_rules(self, rules: dict):
        """Isi semua slider dari dict rules."""
        for (sec, key), var in self._vars.items():
            try:
                val = rules[sec][key]
                if isinstance(val, list):
                    continue
                var.set(float(val))
                sv = self._lbl_vars.get((sec, key))
                if sv:
                    hi = next((h for s, k, _, _, h in _SLIDER_DEFS if s == sec and k == key), 2)
                    lo = next((l for s, k, _, l, _ in _SLIDER_DEFS if s == sec and k == key), 0)
                    sv.set(f"{float(val):.3f}" if (hi - lo) < 2 else f"{float(val):.2f}")
            except (KeyError, TypeError):
                pass

        # Hybrid per-label weights
        sw_list = rules.get("hybrid", {}).get("siglip_w", [0.5] * 4)
        lw_list = rules.get("hybrid", {}).get("land_w",   [0.5] * 4)
        for i, (sv_var, lv_var, sv_sv, lv_sv) in enumerate(self._weight_vars):
            sw = float(sw_list[i]) if i < len(sw_list) else 0.5
            lw = float(lw_list[i]) if i < len(lw_list) else 0.5
            sv_var.set(sw)
            lv_var.set(lw)
            sv_sv.set(f"{sw:.2f}")
            lv_sv.set(f"{lw:.2f}")

    def _collect_rules(self) -> dict:
        """Kumpulkan semua nilai slider ke dict rules."""
        rules = _deep_copy(self._rules_ref)
        for (sec, key), var in self._vars.items():
            if sec not in rules:
                rules[sec] = {}
            rules[sec][key] = round(var.get(), 4)

        # Hybrid per-label weights
        sw_list, lw_list = [], []
        for sv_var, lv_var, _, _ in self._weight_vars:
            sw_list.append(round(sv_var.get(), 4))
            lw_list.append(round(lv_var.get(), 4))
        rules["hybrid"]["siglip_w"] = sw_list
        rules["hybrid"]["land_w"]   = lw_list
        return rules

    # ── Button handlers ───────────────────────────────────────────────────────

    def _reset_default(self):
        from core.rules import DEFAULT_RULES
        self._load_from_rules(DEFAULT_RULES)
        self.lbl_status.configure(text="Reset ke default", text_color="#fbbf24")

    def _save(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        self.lbl_status.configure(text="Tersimpan ✓", text_color="#10b981")

    def _recalculate(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        if self._on_recalculate:
            # Hitung extra_path dari batch mode strip
            extra_path = None
            if self._batch_new_var.get():
                import datetime, os
                name = self._batch_name_var.get().strip()
                if not name:
                    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                extra_path = f"__batch_name__{name}"  # sentinel, resolved di app.py

            self.lbl_status.configure(text="Menghitung ulang…", text_color="#fbbf24")
            self.win.update_idletasks()
            self._on_recalculate(rules, extra_path)
