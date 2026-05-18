LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # Karena input adalah crop wajah, fokus pada EKSPRESI (mata ngantuk, menguap, tatapan kosong)
    # bukan sekadar "noleh" (yang sudah ditangani MediaPipe).
    # Termasuk: kepala nunduk + ekspresi kosong/mengantuk = bosan.
    (
        "a face of a student with heavy droopy eyelids looking extremely sleepy and tired\n"
        "a face of a student yawning widely with an open mouth showing pure exhaustion\n"
        "a face of a student with a completely blank, expressionless, and dull stare\n"
        "a face of a student resting their chin on their hand with lazy unfocused eyes\n"
        "a face of a student with half-closed eyes appearing mentally absent and disengaged\n"
        "a face of a student with relaxed facial muscles and a vacant bored expression\n"
        "a face of a student gazing far away into the distance with vacant unfocused eyes not paying attention to anything specific",
        "",
    ),
    # 1 — Engagement
    # Termasuk: mata lihat bawah ke keyboard/layar dengan ekspresi fokus = engaged.
    # Kepala tidak harus menghadap ke kamera — yang penting ekspresi aktif dan alert.
    (
        "a face of a student making direct eye contact with clear focus and engaged attention\n"
        "a face of a student with bright wide alert eyes actively watching and learning\n"
        "a face of a student with an attentive expression and a subtle interested smile\n"
        "a face of a student nodding and reacting with lively responsive facial features\n"
        "a face of a student with slightly raised eyebrows showing curiosity and focus\n"
        "a face of a student with a sharp, present, and actively involved expression\n"
        "a face of a student looking down at keyboard or notes with an alert focused and engaged expression\n"
        "a face of a student with eyes directed downward actively typing with an attentive concentrated look\n"
        "a face of a student with head turned slightly but eyes clearly tracking the screen with focused engagement",
        "",
    ),
    # 2 — Confusion
    # Fokus: ekspresi tidak mengerti (brow furrow + lost/puzzled) — hindari overlap Engagement
    (
        "a face of a student with deeply furrowed eyebrows looking puzzled and uncertain\n"
        "a face of a student squinting their eyes with visible mental effort to understand\n"
        "a face of a student with a slightly open mouth and raised inner brow looking lost\n"
        "a face of a student with knitted brows and slightly parted lips trying to comprehend something difficult\n"
        "a face of a student looking confused and lost, clearly not understanding the lesson\n"
        "a face of a student with furrowed brows and wide eyes showing confusion about the material\n"
        "a face of a student with a puzzled expression unable to follow what is being explained",
        "",
    ),
    # 3 — Frustration
    # Fokus pada otot tegang: rahang, bibir ditekan, mata tertutup kesal
    (
        "a face of a student showing visible tension with clenched jaw and hand pressed against the forehead in frustration\n"
        "a face of a student with a fierce angry expression and gritting teeth\n"
        "a face of a student sighing heavily with eyes squeezed shut in frustration\n"
        "a face of a student pinching the bridge of their nose showing mental fatigue\n"
        "a face of a student resting their head on their hand looking completely stressed out\n"
        "a face of a student rubbing their eyes forcefully looking completely overwhelmed",
        "",
    ),
]


