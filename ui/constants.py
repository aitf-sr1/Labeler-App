LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

# ── Prompt SigLIP zero-shot ────────────────────────────────────────────────────
# Setiap prompt = deskripsi VISUAL penampilan wajah (SigLIP mencocokkan gambar↔teks).
# Basisnya deskripsi Action Unit FACS dari Craig 2008 + definisi emosi verbatim paper.
#
# PEMBEDA UTAMA antar-label (agar SigLIP tidak rancu):
#   • Confusion  = alis TURUN/berkerut (AU4) + kelopak menyipit/menegang (AU7)  → "tidak paham / ragu"
#   • Frustration = alis NAIK tinggi, dalam & luar (AU1+AU2)                     → "tertekan / gagal / tidak puas"
#   • Boredom    = kelopak turun/setengah menutup (AU43)                         → "lesu / tak berminat"
#   • Engagement = tatapan lurus ke depan (kamera), mata terbuka & waspada (Whitehill holistik)
#
# CATATAN: SigLIP HANYA melihat CROP WAJAH 224x224 — tangan, postur tubuh, & layar TIDAK terlihat.
# Maka semua prompt DIBATASI ke fitur wajah yang tampak (mata, alis, kelopak, mulut, arah tatapan).
# Sinyal tangan (Confusion/Frustration) ditangani terpisah lewat HandLandmarker, bukan di prompt ini.
DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # Paper basis (verbatim def Craig 2008: "the state of being weary and restless through lack of interest"):
    #   Craig et al. (2008) Table 2: AU43 (eye closure) = satu-satunya AU tervalidasi (coverage 40%)
    #   Whitehill et al. (2014) §2.2 level 1: "looking away from computer ... eyes completely closed"
    #   D'Mello & Graesser (2012): boredom = disengagement, wajah datar tanpa ekspresi
    (
        "a student with heavy drooping eyelids and half-closed eyes looking weary and disinterested\n"
        "a student with a blank flat expressionless face and glazed unfocused eyes showing lack of interest\n"
        "a student with eyes nearly closed and a dull vacant stare, mentally absent from the lesson\n"
        "a student with eyes turned away to the side, droopy heavy eyelids and no expression\n"
        "a student with slowly closing sleepy eyelids and a tired uninterested face\n"
        "a weary bored student with closed or half-shut eyes and an empty detached expression",
        "",
    ),
    # 1 — Engagement
    # Paper basis (Whitehill et al. 2014 — penampilan holistik, static pixels, level 3-4):
    #   Level 3: "Engaged in task – student requires no admonition to stay on task"
    #   Level 4: "Very engaged – student could be commended for his/her level of engagement"
    #   Whitehill: forward gaze + eye openness; Craig 2008 tidak menemukan AU primer untuk engagement
    (
        "a student looking straight ahead at the camera with bright alert focused eyes\n"
        "a student with wide open attentive eyes fully concentrating on the content\n"
        "a student with a forward-facing head and steady intent gaze, clearly into the task\n"
        "a student with an alert interested attentive expression and eyes fixed forward\n"
        "a student with focused engaged eyes and an attentive face processing the material\n"
        "a visibly engaged student with bright attentive eyes looking forward and an interested expression",
        "",
    ),
    # 2 — Confusion  (alis TURUN/berkerut + kelopak menyipit — JANGAN alis naik)
    # Paper basis (verbatim def Craig 2008: "the failure to differentiate from an often similar or related other"):
    #   Craig et al. (2008) Table 2: AU4 (brow lowerer) 95%, AU7 (lid tightener) 78%, AU4+AU7 co-occur 73%
    #   (AU12 hanya sekunder/lemah di prosa — BUKAN 95%; tidak dijadikan deskriptor utama)
    #   D'Mello & Graesser (2012): cognitive disequilibrium, impasse, uncertain knowledge state
    (
        "a student with a furrowed lowered brow and tightened narrowed eyelids looking uncertain\n"
        "a student knitting the eyebrows downward with squinting tightened eyes, not understanding\n"
        "a student with a deeply furrowed wrinkled brow and tightened lids, unsure while trying to understand\n"
        "a student with lowered drawn-together eyebrows and narrowed eyes showing doubt and uncertainty\n"
        "a puzzled student with a lowered brow and slightly parted open lips, unsure what to do next\n"
        "a student frowning with lowered knitted brows and tightened squinting eyelids, unable to make sense of the problem",
        "",
    ),
    # 3 — Frustration  (alis NAIK tinggi, dalam & luar — JANGAN alis turun)
    # Paper basis (verbatim def Craig 2008: "making vain or ineffectual efforts, however vigorous; a deep chronic
    #   sense or state of insecurity and dissatisfaction arising from unresolved problems"):
    #   Craig et al. (2008) Table 2: AU1 (inner brow raise) + AU2 (outer brow raise) = 100% coverage, saling memicu
    #   Grafsgaard et al. (2013): AU4 (brow lowering) & AU14 (mouth dimpling) korelasi positif sekunder
    #   D'Mello & Graesser (2012): impasse, gagal berulang, hopeless confusion → frustration
    (
        "a student with inner and outer eyebrows raised high in a distressed strained expression\n"
        "a student with raised arched eyebrows and a tense dissatisfied face after failing a problem\n"
        "a struggling student with lifted worried eyebrows and a strained upset look, unable to resolve the task\n"
        "a distressed student with eyebrows raised high and a tense face showing dissatisfaction\n"
        "a student with raised brows and a discouraged frustrated expression after repeated ineffective efforts\n"
        "a student with inner and outer brows lifted high and a tense exasperated face, stuck and dissatisfied",
        "",
    ),
]
