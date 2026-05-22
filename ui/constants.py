LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # Karena input adalah crop wajah, fokus pada EKSPRESI (mata ngantuk, menguap, tatapan kosong, menopang dagu)
    (
        "a face of a student with heavy droopy eyelids looking extremely sleepy and tired\n"
        "a face of a student yawning widely with an open mouth showing pure exhaustion\n"
        "a face of a student with progressively drooping eyelids and a limp unfocused expression, clearly losing focus\n"
        "a face of a student with heavy-lidded half-closed eyes and a slack expressionless face, showing zero interest\n"
        "a face of a student with half-closed eyes and a slack jaw, completely uninterested and mentally elsewhere\n"
        "a face of a student gazing far away into the distance, mentally absent and completely disengaged",
        "",
    ),
    # 1 — Engagement
    # Termasuk: mata lihat bawah ke layar dengan ekspresi fokus, mata terbuka lebar, alis sedikit naik = engaged.
    (
        "a face of a student making direct eye contact with clear focus, highly engaged and alert\n"
        "a face of a student with bright, wide, attentive eyes, actively watching and concentrating\n"
        "a face of a student with concentrated eyes actively tracking content, showing clear mental engagement\n"
        "a face of a student with a calm serious expression and steady open eyes, quietly focused and attentive\n"
        "a face of a student looking down at a screen or keyboard with an intensely concentrated and focused look\n"
        "a face of a student nodding or reacting, clearly processing information with a highly engaged expression",
        "",
    ),
    # 2 — Confusion
    # Fokus: dahi berkerut (furrowed brow), mata menyipit (squinting), kepala miring, mulut sedikit terbuka
    (
        "a face of a student with heavily furrowed brows and squinting eyes, looking very confused and puzzled\n"
        "a face of a student with a slightly open mouth and raised inner brows, looking completely lost\n"
        "a face of a student tilting their head with a deeply perplexed and confused expression\n"
        "a face of a student with pursed lips and slightly furrowed brow, looking uncertain and mentally stuck\n"
        "a face of a student with a tense questioning expression, clearly trying to process something unclear\n"
        "a face of a student with wide eyes and a slightly parted mouth, frozen in puzzlement and disbelief",
        "",
    ),
    # 3 — Frustration
    # Fokus pada otot tegang: rahang, bibir ditekan, mata tertutup kesal
    (
        "a face of a student showing visible tension with clenched jaw and hand pressed against the forehead in frustration\n"
        "a face of a student with a fierce angry expression and gritting teeth\n"
        "a face of a student sighing heavily with eyes squeezed shut in frustration\n"
        "a face of a student pinching the bridge of their nose showing mental fatigue\n"
        "a face of a student with a tense jaw and furrowed brows, looking completely overwhelmed and stressed\n"
        "a face of a student rubbing their temples or eyes forcefully with a grimacing stressed expression",
        "",
    ),
]


