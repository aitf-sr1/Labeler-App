"""
ui/rules_panel.py

Toplevel window untuk mengedit semua parameter rules (landmark scoring + hybrid weights).
Dibuka via tombol "Rules" di aplikasi utama.

Also exports RulesContent — a frameless embedded version for use inside right_panel.
"""

import json
import tkinter as tk
import customtkinter as ctk

from core.rules import DEFAULT_RULES, _deep_copy

# ── Definisi slider per seksi ─────────────────────────────────────────────────
# (section, key, label, min_val, max_val, description)
_SLIDER_DEFS = [
    # GAZE (shared)
    ("gaze", "scale_h",        "H Scale (iris->deg)",    10.0, 60.0,
     "Pengali iris_x ke derajat horizontal. Naik → gaze_dev lebih besar dari gerakan iris kecil → Boredom ↑, Engagement ↓. Turun → butuh noleh lebih jauh baru dihitung bored/tidak fokus."),
    ("gaze", "scale_v",        "V Scale (iris->deg)",    10.0, 50.0,
     "Pengali iris_y ke derajat vertikal. Naik → pandangan sedikit ke atas/bawah lebih berpengaruh → Boredom (lihat atas) ↑, Confusion (lihat bawah) lebih sensitif. Turun → perlu gerakan iris lebih besar."),
    ("gaze", "iris_side_mult", "Iris Side Mult",       0.5,  4.0,
     "Pengali ekstra saat iris ke samping. Naik → menoleh ke samping lebih kuat deteksinya → Boredom ↑, Engagement ↓. Berguna karena iris samping lebih tidak akurat dari blendshape yaw saja."),
    ("gaze", "v_dead_zone",    "V Dead Zone Down (deg)",    0.0, 30.0,
     "Dead zone gaze ke BAWAH (deg). Naik → nunduk/ngetik lebih dilindungi, tidak dianggap bored → Boredom ↓ saat nunduk. Turun → nunduk sedikit sudah menaikkan boredom."),
    ("gaze", "v_dead_zone_up", "V Dead Zone Up (deg)",    0.0, 15.0,
     "Dead zone gaze ke ATAS (deg). Naik → pandangan ke atas lebih ditoleransi → Boredom ↓ saat mendongak. Turun → pandangan sedikit ke atas sudah trigger boredom."),
    ("gaze", "roll_dz",        "Roll Dead Zone (deg)",    0.0, 15.0,
     "Dead zone roll (kepala miring) sebelum dihitung sebagai gaze_dev. Naik → kepala miring lebih banyak ditoleransi → semua emosi berbasis gaze lebih stabil. Turun → kepala miring sedikit sudah dihitung."),
    ("gaze", "iris_blink_suppress_th", "Iris Blink Suppress Th", 0.3, 0.9,
     "Blink terkoreksi > ini → iris_y mulai di-suppress secara proporsional. Naik = iris tetap dipakai lebih lama saat mata menutup (lebih responsif tapi rawan artefak). Turun = suppress lebih agresif, hilangkan artefak iris saat berkedip."),
    # BOREDOM
    ("boredom", "gaze_dead_zone",  "Gaze Dead Zone (deg)",  0.0, 20.0,
     "Gaze deviation di bawah ini = tidak bosan. Makin besar = siswa perlu lihat lebih jauh dari layar baru dianggap bored."),
    ("boredom", "gaze_range",      "Gaze Range (deg)",      4.0, 25.0,
     "Rentang gaze di atas dead zone sampai jenuh. Makin besar = transisi boredom lebih landai/gradual."),
    ("boredom", "blink_dead_zone", "Blink Dead Zone",     0.0,  0.5,
     "Blink di bawah ini diabaikan. Mencegah kedipan normal trigger boredom."),
    ("boredom", "blink_range",     "Blink Range",         0.1,  1.0,
     "Rentang blink di atas dead zone. Makin besar = butuh mata lebih merem baru efek signifikan."),
    ("boredom", "sig_expr_weight", "Expr Weight",         0.0,  1.0,
     "Bobot sinyal blink (AU43) dalam boredom. Craig 2008: AU43 = satu-satunya sinyal yang divalidasi."),
    ("boredom", "squint_blink_correction", "Squint->Blink Correction", 0.0, 1.0,
     "Koreksi teknis: squint sedikit menutup mata → inflasi blink_avg. Bukan AU, hanya kalibrasi sensor."),
    ("boredom", "fwd_yaw_th",        "Fwd Yaw Th (deg)",    3.0,  20.0,
     "MUTLAK: |yaw| < ini = hadap depan = TIDAK BOSAN. Boredom hanya muncul kalau kepala menoleh. Default 8 deg."),
    ("boredom", "fwd_yaw_range",     "Fwd Yaw Range (deg)", 3.0,  15.0,
     "Range transisi dari threshold. Boredom 0 di yaw=th, naik linear, penuh di yaw=th+range. Default 7 deg -> penuh di 15 deg."),
    # ENGAGEMENT
    ("engagement", "tegak_dead_zone", "Tegak Dead Zone (deg)", 0.0, 10.0,
     "Gaze deviation di bawah ini = engaged penuh. Makin besar = lebih toleran pandangan agak menyimpang."),
    ("engagement", "tegak_range",     "Tegak Range (deg)",     5.0, 25.0,
     "Rentang gaze di atas dead zone. Di dead+range, engagement = 0. Makin besar = transisi lebih pelan."),
    ("engagement", "yaw_gate_th",     "Yaw Gate Th (deg)",    10.0, 30.0,
     "Yaw (kepala menoleh) > ini = engagement mulai turun. Default 20 deg = toleran sampai menoleh cukup jauh."),
    ("engagement", "yaw_gate_range",  "Yaw Gate Range (deg)",  5.0, 20.0,
     "Rentang yaw di atas threshold. Di th+range, engagement = 0. Default 10 deg."),
    ("engagement", "roll_gate_th",    "Roll Gate Th (deg)",    5.0, 20.0,
     "Roll (kepala miring) > ini = engagement mulai turun. Default 10 deg = natural tilt aman."),
    ("engagement", "roll_gate_range", "Roll Gate Range (deg)", 3.0, 15.0,
     "Rentang roll sampai engagement = 0. Default 5 deg -> nol di roll 15 deg."),
    ("engagement", "blink_heavy_th",  "Heavy Blink Th",      0.2,  0.9,
     "Blink > ini = mata terlalu merem (ngantuk). Menurunkan engagement."),
    ("engagement", "blink_heavy_min", "Min Engagement",      0.0,  0.5,
     "Engagement minimum saat blink sangat berat. Lantai engagement tidak pernah di bawah ini."),
    ("engagement", "eye_wide_boost",  "EyeWide Boost",       0.0,  0.5,
     "Inverse 'eyes barely open' (Whitehill level 2). Naik = mata lebar lebih boost engagement."),
    ("engagement", "pitch_gate_th",   "Pitch Gate Th (deg)",   0.0, 30.0,
     "Kepala mendongak > ini = engagement mulai turun. 15 deg = wajar duduk, nol di th+range."),
    ("engagement", "pitch_gate_range","Pitch Gate Range (deg)", 5.0, 40.0,
     "Rentang pitch di atas threshold sampai engagement = 0. Default 15 deg -> nol di pitch 30 deg."),
    ("engagement", "bore_suppress_th", "Bore Suppress Th",    0.0,  0.5,
     "D'Mello 2012: boredom↔engagement near-exclusive. Dead zone sebelum suppress aktif."),
    ("engagement", "bore_eng_suppress","Bore->Eng Suppress",   0.0,  1.0,
     "D'Mello 2012: boredom tinggi menekan engagement."),
    ("engagement", "conf_eng_suppress_th", "Conf Eng Supp Th", 0.2, 0.8,
     "D'Mello 2012: Confusion→Engagement/Flow signifikan (productive struggle). Threshold suppress."),
    ("engagement", "conf_eng_suppress",    "Conf->Eng Suppress", 0.0, 0.8,
     "D'Mello 2012: confusion menekan engagement tapi tidak total (productive struggle bisa co-occur)."),
    # CONFUSION — Craig et al. (2008): AU4 (95%), AU7 (78%), AU4+AU7 co-occurrence (73%), AU12 secondary (95%)
    ("confusion", "brow_dn_th",         "BrowDown Th (AU4)",   0.01, 0.5,
     "Pembagi browDown (AU4, Craig2008 95%). Kalibrasi MediaPipe: p90=0.033 → set 0.05 agar AU4 aktif."),
    ("confusion", "au7_th",             "AU7 Squint Th",       0.05, 0.4,
     "Threshold eyeSquint minimum untuk dihitung sebagai AU7 (lid tightener) dalam co-occurrence AU4+AU7."),
    ("confusion", "au4_au7_co_w",       "AU4+AU7 Co-occur W",  0.1,  1.0,
     "Bobot co-occurrence AU4+AU7 (Craig2008 73% coverage). Naik = co-occurrence lebih dominan dari AU4 sendiri."),
    ("confusion", "smile_conf_gate_th", "AU12 Smile Gate Th",  0.1,  0.5,
     "AU12 (mouthSmile/questioning smile) co-occurs 95% (Craig2008). Gate floor mencegah senyum mematikan confusion."),
    ("confusion", "smile_conf_gate_floor", "AU12 Smile Floor", 0.1,  0.5,
     "Floor gate senyum. 0.30 = confusion tetap ≥30% meski senyum penuh (AU12 co-occurs per Craig2008)."),
    ("confusion", "bore_conf_suppress_bore", "Bore->Conf Suppress", 0.0, 0.8,
     "D'Mello 2012: Confusion→Boredom terjadi at chance. Boredom tinggi menekan confusion."),
    # FRUSTRATION — Craig et al. (2008): AU1+AU2 primary (100%), AU4 secondary (Grafsgaard 2013)
    ("frustration", "brow_outer_up_th", "BrowOuterUp Th (AU2)", 0.10, 1.0,
     "Craig2008: AU2 (outer brow raise) = primary frustration signal, 100% coverage. "
     "browOuterUp median alami=0.47 → threshold 0.75 = hanya top 25% yang fire (kalibrasi MediaPipe)."),
    ("frustration", "brow_inner_up_th", "BrowInnerUp Th (AU1)", 0.10, 1.0,
     "Craig2008: AU1 (inner brow raise) = primary frustration signal, 100% coverage. "
     "browInnerUp median alami=0.43 → threshold 0.75 = hanya top 25% yang fire (kalibrasi MediaPipe)."),
    ("frustration", "brow_raise_direct_w", "AU1+AU2 Direct W",  0.3,  1.0,
     "Craig2008: AU1+AU2 co-occurrence = 100% coverage. Bobot langsung untuk sinyal primer."),
    ("frustration", "brow_dn_th",      "BrowDown Th (AU4)",   0.01, 0.6,
     "Grafsgaard 2013: AU4 (brow lowering) positively correlated with frustration. Secondary signal."),
    ("frustration", "face_weight",     "AU4 Secondary W",     0.1,  0.8,
     "Bobot AU4 secondary (Grafsgaard 2013) relatif terhadap primary AU1+AU2."),
    # HYBRID
    ("hybrid", "empirical_bias",       "SigLIP Empirical Bias", 1.0, 6.0,
     "Bias global yang ditambahkan ke logits SigLIP sebelum sigmoid. Naik → semua skor SigLIP naik (semua emosi lebih mudah positif). Turun → skor SigLIP lebih konservatif. Terlalu besar = semua label selalu 1."),
    ("hybrid", "restless_bonus_max",   "Restless Bonus Max",    0.0, 0.3,
     "Bonus maks Boredom dari variasi yaw (kepala gelisah bergerak). 0 = disabled (direkomendasikan — heuristik tanpa basis definisi). Naik → Boredom ↑ jika kepala banyak bergerak antar frame."),
    ("hybrid", "restless_std_min",     "Restless Std Min (deg)",  1.0, 8.0,
     "Std-dev yaw minimum sebelum restless bonus Boredom aktif. Naik = butuh gerakan kepala lebih banyak sebelum bonus aktif."),
    ("hybrid", "restless_std_range",   "Restless Std Range (deg)", 3.0, 15.0,
     "Rentang std-dev yaw dari min sampai bonus maks. Naik = transisi lebih pelan, bonus Boredom naik lebih bertahap."),
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


# ── Shared builder helpers ────────────────────────────────────────────────────

def _build_slider_sections(parent, vars_dict, lbl_vars_dict):
    """Bangun slider per parameter, dikelompokkan per seksi."""
    sections_seen = []
    sliders_by_section: dict = {}
    for sec, key, lbl, lo, hi, desc in _SLIDER_DEFS:
        if sec == "hybrid" and key in ("siglip_w", "land_w"):
            continue  # handled separately
        sliders_by_section.setdefault(sec, []).append((key, lbl, lo, hi, desc))
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

        for key, lbl, lo, hi, desc in sliders_by_section[sec]:
            _build_one_slider(parent, sec, key, lbl, lo, hi, desc, vars_dict, lbl_vars_dict)


def _build_one_slider(parent, sec, key, lbl, lo, hi, desc, vars_dict, lbl_vars_dict):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=8, pady=1)

    ctk.CTkLabel(row, text=lbl, font=("Poppins", 9), width=155, anchor="w").pack(side="left")

    var = ctk.DoubleVar(value=0.0)
    is_fine = (hi - lo) < 2
    fmt = "{:.3f}" if is_fine else "{:.2f}"
    ev = ctk.StringVar(value="0.00")
    vars_dict[(sec, key)]     = var
    lbl_vars_dict[(sec, key)] = ev

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

    # Deskripsi kecil di bawah slider
    if desc:
        ctk.CTkLabel(
            parent, text=f"    {desc}",
            font=("Poppins", 8), text_color=("#6b7280", "#6b7280"),
            anchor="w", wraplength=500,
        ).pack(fill="x", padx=8, pady=(0, 2))


def _build_hybrid_weight_section(parent, vars_dict, lbl_vars_dict, weight_vars_list):
    """Bangun slider siglip_w dan land_w per label."""
    color = _SECTION_COLORS["hybrid"]
    hdr = ctk.CTkLabel(
        parent, text="  Hybrid Weights (per label)",
        font=("Poppins", 10, "bold"), text_color=color, anchor="w",
    )
    hdr.pack(fill="x", padx=4, pady=(10, 2))
    ctk.CTkFrame(parent, fg_color=color, height=1, corner_radius=0).pack(fill="x", padx=4, pady=(0, 4))

    # Also include scalar hybrid params from _SLIDER_DEFS
    for sec, key, lbl, lo, hi, desc in _SLIDER_DEFS:
        if sec == "hybrid" and key not in ("siglip_w", "land_w"):
            _build_one_slider(parent, sec, key, lbl, lo, hi, desc, vars_dict, lbl_vars_dict)

    # Per-label weight rows — SigLIP dan Landmark saling terhubung (total = 1.0)
    from ui.constants import LABEL_COLORS as LC

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

        _updating = [False]

        def _mk_siglip_cb(_sv_var=sv_var, _lv_var=lv_var, _sv_sv=sv_sv, _lv_sv=lv_sv, _upd=_updating):
            def cb(v):
                if _upd[0]:
                    return
                _upd[0] = True
                val = float(v)
                comp = round(1.0 - val, 2)
                _lv_var.set(comp)
                _sv_sv.set(f"{val:.2f}")
                _lv_sv.set(f"{comp:.2f}")
                _upd[0] = False
            return cb

        def _mk_land_cb(_sv_var=sv_var, _lv_var=lv_var, _sv_sv=sv_sv, _lv_sv=lv_sv, _upd=_updating):
            def cb(v):
                if _upd[0]:
                    return
                _upd[0] = True
                val = float(v)
                comp = round(1.0 - val, 2)
                _sv_var.set(comp)
                _lv_sv.set(f"{val:.2f}")
                _sv_sv.set(f"{comp:.2f}")
                _upd[0] = False
            return cb

        def _mk_commit_linked(dv, _sv, other_dv, other_sv, _upd, lo=0.0, hi=1.0):
            def commit():
                if _upd[0]:
                    return
                _upd[0] = True
                try:
                    v = float(_sv.get().replace(",", "."))
                    v = max(lo, min(hi, v))
                    comp = round(1.0 - v, 2)
                    dv.set(v)
                    other_dv.set(comp)
                    _sv.set(f"{v:.2f}")
                    other_sv.set(f"{comp:.2f}")
                except ValueError:
                    _sv.set(f"{dv.get():.2f}")
                _upd[0] = False
            return commit

        # SigLIP row
        rw_s = ctk.CTkFrame(sec_frame, fg_color="transparent")
        rw_s.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(rw_s, text="SigLIP", font=("Poppins", 9), width=60, anchor="w").pack(side="left")
        ent_s = ctk.CTkEntry(rw_s, textvariable=sv_sv, width=52, height=20,
                             font=("Poppins", 9, "bold"), justify="right",
                             fg_color=("white", "#111122"))
        ent_s.pack(side="right", padx=(2, 0))
        commit_s = _mk_commit_linked(sv_var, sv_sv, lv_var, lv_sv, _updating)
        ent_s.bind("<Return>", lambda e, fn=commit_s: fn())
        ent_s.bind("<FocusOut>", lambda e, fn=commit_s: fn())
        ctk.CTkSlider(rw_s, from_=0.0, to=1.0, variable=sv_var,
                      command=_mk_siglip_cb()).pack(side="left", fill="x", expand=True, padx=(4, 4))

        # Landmark row
        rw_l = ctk.CTkFrame(sec_frame, fg_color="transparent")
        rw_l.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(rw_l, text="Landmark", font=("Poppins", 9), width=60, anchor="w").pack(side="left")
        ent_l = ctk.CTkEntry(rw_l, textvariable=lv_sv, width=52, height=20,
                             font=("Poppins", 9, "bold"), justify="right",
                             fg_color=("white", "#111122"))
        ent_l.pack(side="right", padx=(2, 0))
        commit_l = _mk_commit_linked(lv_var, lv_sv, sv_var, sv_sv, _updating)
        ent_l.bind("<Return>", lambda e, fn=commit_l: fn())
        ent_l.bind("<FocusOut>", lambda e, fn=commit_l: fn())
        ctk.CTkSlider(rw_l, from_=0.0, to=1.0, variable=lv_var,
                      command=_mk_land_cb()).pack(side="left", fill="x", expand=True, padx=(4, 4))

        ctk.CTkFrame(sec_frame, fg_color="transparent", height=2).pack()
        weight_vars_list.append((sv_var, lv_var, sv_sv, lv_sv))


def _build_threshold_section(parent, threshold_vars):
    """Slider threshold per label -- share DoubleVar dengan right panel."""
    if not threshold_vars:
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

    for i, (lbl_name, dv) in enumerate(zip(_LABEL_NAMES, threshold_vars)):
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


def _load_from_rules_impl(rules, vars_dict, lbl_vars_dict, weight_vars_list):
    """Isi semua slider dari dict rules."""
    for (sec, key), var in vars_dict.items():
        try:
            val = rules[sec][key]
            if isinstance(val, list):
                continue
            var.set(float(val))
            sv = lbl_vars_dict.get((sec, key))
            if sv:
                hi = next((h for s, k, _, _, h, *_ in _SLIDER_DEFS if s == sec and k == key), 2)
                lo = next((l for s, k, _, l, *_ in _SLIDER_DEFS if s == sec and k == key), 0)
                sv.set(f"{float(val):.3f}" if (hi - lo) < 2 else f"{float(val):.2f}")
        except (KeyError, TypeError):
            pass

    # Hybrid per-label weights
    sw_list = rules.get("hybrid", {}).get("siglip_w", [0.5] * 4)
    lw_list = rules.get("hybrid", {}).get("land_w",   [0.5] * 4)
    for i, (sv_var, lv_var, sv_sv, lv_sv) in enumerate(weight_vars_list):
        sw = float(sw_list[i]) if i < len(sw_list) else 0.5
        lw = float(lw_list[i]) if i < len(lw_list) else 0.5
        sv_var.set(sw)
        lv_var.set(lw)
        sv_sv.set(f"{sw:.2f}")
        lv_sv.set(f"{lw:.2f}")


def _collect_rules_impl(rules_ref, vars_dict, weight_vars_list):
    """Kumpulkan semua nilai slider ke dict rules."""
    rules = _deep_copy(rules_ref)
    for (sec, key), var in vars_dict.items():
        if sec not in rules:
            rules[sec] = {}
        rules[sec][key] = round(var.get(), 4)

    # Hybrid per-label weights
    sw_list, lw_list = [], []
    for sv_var, lv_var, _, _ in weight_vars_list:
        sw_list.append(round(sv_var.get(), 4))
        lw_list.append(round(lv_var.get(), 4))
    rules["hybrid"]["siglip_w"] = sw_list
    rules["hybrid"]["land_w"]   = lw_list
    return rules


# ── RulesContent — embedded inline version ───────────────────────────────────

class RulesContent:
    """
    Inline (non-window) version of RulesPanel.
    Builds rules content into a given parent frame.
    Used by right_panel._build_rules_section().

    Public API:
        self.frame          -- the outer CTkScrollableFrame
        self.lbl_status     -- CTkLabel for feedback messages
        _load_from_rules(rules)  -- populate sliders from rules dict
        _collect_rules()         -- read sliders back to dict
    """

    def __init__(self, parent, rules: dict, threshold_vars=None, on_save=None,
                 on_recalculate=None, on_rebatch=None, on_close=None):
        self._rules_ref      = _deep_copy(rules)
        self._threshold_vars = threshold_vars or []
        self._on_save        = on_save
        self._on_recalculate = on_recalculate
        self._on_rebatch     = on_rebatch
        self._on_close       = on_close

        self._vars: dict      = {}
        self._lbl_vars: dict  = {}
        self._weight_vars: list = []

        self._built = False
        self._parent = parent

    def build(self):
        """Lazily build the UI. Called on first accordion expand."""
        if self._built:
            return
        self._built = True
        self._build(self._parent)
        self._load_from_rules(self._rules_ref)

    def _build(self, parent):
        # Scrollable content area — no fixed height, fills available space
        self.frame = ctk.CTkScrollableFrame(
            parent, fg_color="transparent",
        )
        self.frame.pack(fill="both", expand=True, padx=4, pady=4)

        _build_slider_sections(self.frame, self._vars, self._lbl_vars)
        _build_hybrid_weight_section(self.frame, self._vars, self._lbl_vars, self._weight_vars)
        _build_threshold_section(self.frame, self._threshold_vars)

        # ── Batch mode strip ─────────────────────────────────────────────────
        bstrip = ctk.CTkFrame(parent, fg_color=("#d1d5db", "#1e1e2c"), corner_radius=8)
        bstrip.pack(fill="x", padx=4, pady=(4, 2))

        ctk.CTkLabel(
            bstrip, text="Simpan ke:", font=("Poppins", 9),
            text_color=("#374151", "#9ca3af"),
        ).pack(side="left", padx=(10, 4), pady=6)

        self._batch_new_var  = ctk.BooleanVar(value=False)
        self._batch_name_var = ctk.StringVar(value="")

        self._batch_name_entry = ctk.CTkEntry(
            bstrip, textvariable=self._batch_name_var,
            placeholder_text="nama batch baru...",
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

        self._batch_name_entry.pack_forget()

        # ── Button bar ───────────────────────────────────────────────────────
        bar = ctk.CTkFrame(parent, fg_color=("#e5e7eb", "#1a1a28"), corner_radius=8)
        bar.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkButton(
            bar, text="Reset Default", width=110, height=30,
            fg_color=("#9ca3af", "#374151"), hover_color=("#6b7280", "#1f2937"),
            font=("Poppins", 10), corner_radius=7, command=self._reset_default,
        ).pack(side="left", padx=(10, 6), pady=8)

        ctk.CTkButton(
            bar, text="Simpan", width=90, height=30,
            fg_color="#6366f1", hover_color="#4f46e5",
            font=("Poppins", 10, "bold"), corner_radius=7, command=self._save_only,
        ).pack(side="right", padx=(4, 10), pady=8)

        ctk.CTkButton(
            bar, text="Recalculate", width=100, height=30,
            fg_color="#3b82f6", hover_color="#2563eb",
            font=("Poppins", 10, "bold"), corner_radius=7, command=self._save,
        ).pack(side="right", padx=4, pady=8)

        ctk.CTkButton(
            bar, text="Rebatch", width=80, height=30,
            fg_color="#dc2626", hover_color="#b91c1c",
            font=("Poppins", 10, "bold"), corner_radius=7, command=self._rebatch,
        ).pack(side="right", padx=4, pady=8)

        self.lbl_status = ctk.CTkLabel(
            bar, text="", font=("Poppins", 9), text_color="#10b981"
        )
        self.lbl_status.pack(side="right", padx=6)

    # ── Load / collect ────────────────────────────────────────────────────────

    def _load_from_rules(self, rules: dict):
        self._rules_ref = _deep_copy(rules)
        if not self._built:
            return  # will load when built
        _load_from_rules_impl(rules, self._vars, self._lbl_vars, self._weight_vars)

    def _collect_rules(self) -> dict:
        return _collect_rules_impl(self._rules_ref, self._vars, self._weight_vars)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _reset_default(self):
        _load_from_rules_impl(DEFAULT_RULES, self._vars, self._lbl_vars, self._weight_vars)
        self.lbl_status.configure(text="Reset ke default", text_color="#fbbf24")

    def _save_only(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        self.lbl_status.configure(text="Tersimpan", text_color="#10b981")
        if self._on_close:
            self._on_close()

    def _save(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        if self._on_recalculate:
            extra_path = None
            if self._batch_new_var.get():
                import datetime, os
                name = self._batch_name_var.get().strip()
                if not name:
                    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                extra_path = f"__batch_name__{name}"
            self.lbl_status.configure(text="Menyimpan + Menghitung ulang...", text_color="#fbbf24")
            self._on_recalculate(rules, extra_path)
            if self._on_close:
                self._on_close()
        else:
            self.lbl_status.configure(text="Tersimpan", text_color="#10b981")
            if self._on_close:
                self._on_close()

    def _rebatch(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        if self._on_rebatch:
            self.lbl_status.configure(text="Memulai rebatch...", text_color="#f59e0b")
            self._on_rebatch()


# ── RulesPanel — original Toplevel version (kept for backwards compat) ────────

class RulesPanel:
    """
    Toplevel window untuk edit rules landmark + hybrid weights.

    Callback:
        on_save(rules: dict)       -- dipanggil saat Save ditekan.
        on_recalculate(rules: dict) -- dipanggil saat Recalculate ditekan.
    """

    def __init__(self, parent, rules: dict, threshold_vars=None, on_save=None, on_recalculate=None, on_rebatch=None):
        self._rules_ref = _deep_copy(rules)  # working copy
        self._threshold_vars = threshold_vars or []   # shared DoubleVar dari right panel
        self._on_save = on_save
        self._on_recalculate = on_recalculate
        self._on_rebatch = on_rebatch

        self.win = ctk.CTkToplevel(parent)
        self.win.title("Rules Editor -- Parameter Landmark & Hybrid")
        self.win.geometry("560x720")
        self.win.minsize(480, 500)
        self.win.lift()
        self.win.focus_force()
        # grab_set() didelay agar window sudah fully rendered sebelum grab
        self.win.after(150, self.win.grab_set)

        self._vars: dict = {}
        self._lbl_vars: dict = {}
        self._weight_vars: list = []
        self._build()
        self._load_from_rules(self._rules_ref)

    # ── Build UI ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bind_scroll(scrollable_frame):
        """Aktifkan scroll mouse di CTkScrollableFrame (Toplevel tidak auto-bind)."""
        canvas = scrollable_frame._parent_canvas
        scrollable_frame.bind("<Enter>", lambda _: scrollable_frame.bind_all(
            "<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        ))
        scrollable_frame.bind("<Enter>", lambda _: scrollable_frame.bind_all(
            "<Button-4>", lambda _e: canvas.yview_scroll(-1, "units")
        ), add="+")
        scrollable_frame.bind("<Enter>", lambda _: scrollable_frame.bind_all(
            "<Button-5>", lambda _e: canvas.yview_scroll(1, "units")
        ), add="+")
        scrollable_frame.bind("<Leave>", lambda _: scrollable_frame.unbind_all("<MouseWheel>"))
        scrollable_frame.bind("<Leave>", lambda _: scrollable_frame.unbind_all("<Button-4>"), add="+")
        scrollable_frame.bind("<Leave>", lambda _: scrollable_frame.unbind_all("<Button-5>"), add="+")

    def _build(self):
        # Main layout: scroll area + batch strip + button bar
        self.win.rowconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=0)  # batch mode strip
        self.win.rowconfigure(2, weight=0)  # button bar
        self.win.columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self.win, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))
        self._bind_scroll(scroll)

        _build_slider_sections(scroll, self._vars, self._lbl_vars)
        _build_hybrid_weight_section(scroll, self._vars, self._lbl_vars, self._weight_vars)
        _build_threshold_section(scroll, self._threshold_vars)

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
            placeholder_text="nama batch baru...",
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
            bar, text="Reset Default", width=110, height=32,
            fg_color=("#9ca3af", "#374151"), hover_color=("#6b7280", "#1f2937"),
            font=("Poppins", 10), command=self._reset_default,
        ).pack(side="left", padx=(12, 6), pady=9)

        ctk.CTkButton(
            bar, text="Simpan", width=90, height=32,
            fg_color="#6366f1", hover_color="#4f46e5",
            font=("Poppins", 10, "bold"), command=self._save_only,
        ).pack(side="right", padx=(4, 12), pady=9)

        ctk.CTkButton(
            bar, text="Recalculate", width=100, height=32,
            fg_color="#3b82f6", hover_color="#2563eb",
            font=("Poppins", 10, "bold"), command=self._save,
        ).pack(side="right", padx=4, pady=9)

        ctk.CTkButton(
            bar, text="Rebatch", width=80, height=32,
            fg_color="#dc2626", hover_color="#b91c1c",
            font=("Poppins", 10, "bold"), command=self._rebatch,
        ).pack(side="right", padx=4, pady=9)

        self.lbl_status = ctk.CTkLabel(
            bar, text="", font=("Poppins", 9), text_color="#10b981"
        )
        self.lbl_status.pack(side="right", padx=6)

    # ── Load / collect ────────────────────────────────────────────────────────

    def _load_from_rules(self, rules: dict):
        """Isi semua slider dari dict rules."""
        _load_from_rules_impl(rules, self._vars, self._lbl_vars, self._weight_vars)

    def _collect_rules(self) -> dict:
        """Kumpulkan semua nilai slider ke dict rules."""
        return _collect_rules_impl(self._rules_ref, self._vars, self._weight_vars)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _reset_default(self):
        _load_from_rules_impl(DEFAULT_RULES, self._vars, self._lbl_vars, self._weight_vars)
        self.lbl_status.configure(text="Reset ke default", text_color="#fbbf24")

    def _save_only(self):
        """Simpan rules ke file saja -- tidak recalculate. Cepat."""
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        self.lbl_status.configure(text="Tersimpan", text_color="#10b981")

    def _save(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        if self._on_recalculate:
            extra_path = None
            if self._batch_new_var.get():
                import datetime, os
                name = self._batch_name_var.get().strip()
                if not name:
                    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                extra_path = f"__batch_name__{name}"
            self.lbl_status.configure(text="Menyimpan + Menghitung ulang...", text_color="#fbbf24")
            self.win.update_idletasks()
            self._on_recalculate(rules, extra_path)
        else:
            self.lbl_status.configure(text="Tersimpan", text_color="#10b981")

    def _rebatch(self):
        rules = self._collect_rules()
        self._rules_ref = rules
        if self._on_save:
            self._on_save(rules)
        if self._on_rebatch:
            self.lbl_status.configure(text="Memulai rebatch...", text_color="#f59e0b")
            self.win.update_idletasks()
            self._on_rebatch()
