LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    (
        "a student looking away from the screen with no interest\n"
        "a student turning their face away from the monitor\n"
        "a student not paying attention and looking around the room\n"
        "a student yawning with mouth wide open\n"
        "a student with eyes not directed at the screen\n"
        "a student appearing completely inattentive and distracted",
        "",
    ),
    # 1 — Engagement
    (
        "a student staring intently at the screen with focused eyes\n"
        "a student with eyes fully directed at the monitor\n"
        "a student laughing and looking cheerful while learning\n"
        "a student smiling happily and enjoying the lesson\n"
        "a student looking motivated and enthusiastic\n"
        "a student leaning forward with great interest and energy",
        "",
    ),
    # 2 — Confusion
    (
        "a student with a blank distant stare and mind elsewhere\n"
        "a student daydreaming with eyes not focused on anything\n"
        "a student resting chin or face on hand while staring blankly\n"
        "a student squinting as if trying hard to understand\n"
        "a student with a puzzled expression showing they do not understand\n"
        "a student looking lost and confused by the material",
        "",
    ),
    # 3 — Frustration
    (
        "a student looking exhausted and mentally drained from studying\n"
        "a student covering their face with both hands\n"
        "a student looking visibly stressed and overwhelmed\n"
        "a student with a vacant empty stare showing exhaustion\n"
        "a student pressing hands against face or forehead in frustration\n"
        "a student appearing burned out and unable to continue",
        "",
    ),
]
