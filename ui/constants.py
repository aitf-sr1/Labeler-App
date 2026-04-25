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
    (
        "a face of a student with heavy droopy eyelids looking extremely sleepy and tired\n"
        "a face of a student yawning widely with an open mouth showing pure exhaustion\n"
        "a face of a student with a completely blank, expressionless, and dull stare\n"
        "a face of a student resting their chin on their hand with lazy unfocused eyes\n"
        "a face of a student with half-closed eyes appearing mentally absent and disengaged\n"
        "a face of a student with relaxed facial muscles and a vacant bored expression",
        "",
    ),
    # 1 — Engagement
    (
        "a face of a student making direct eye contact with clear focus and engaged attention\n"
        "a face of a student with bright wide alert eyes actively watching and learning\n"
        "a face of a student with an attentive expression and a subtle interested smile\n"
        "a face of a student nodding and reacting with lively responsive facial features\n"
        "a face of a student with slightly raised eyebrows showing curiosity and focus\n"
        "a face of a student with a sharp, present, and actively involved expression",
        "",
    ),
    # 2 — Confusion
    # Fokus pada kerutan alis spesifik dan mata menyipit
    (
        "a face of a student with deeply furrowed eyebrows looking puzzled and uncertain\n"
        "a face of a student squinting their eyes with visible mental effort to understand\n"
        "a face of a student with a slightly open mouth and raised brow looking lost\n"
        "a face of a student tilting their head with a questioning and perplexed look\n"
        "a face of a student with a tense forehead and puckered lips feeling very confused\n"
        "a face of a student scratching their head feeling puzzled and confused\n"
        "a face of a student with wide bewildered eyes trying hard to process information",
        "",
    ),
    # 3 — Frustration
    # Fokus pada otot tegang: rahang, bibir ditekan, mata tertutup kesal
    (
        "a face of a student with tightly pressed lips and a strained frustrated expression\n"
        "a face of a student with a clenched jaw and tense angry eyebrows\n"
        "a face of a student sighing heavily with eyes squeezed shut in defeat\n"
        "a face of a student pinching the bridge of their nose showing mental fatigue\n"
        "a face of a student resting their head on their hand looking completely stressed out\n"
        "a face of a student rubbing their eyes forcefully looking completely overwhelmed",
        "",
    ),
]


