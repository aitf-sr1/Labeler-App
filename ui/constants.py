LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # Prioritas spec: non-forward gaze, disengaged head direction, slouched posture, passive body language
    # TIDAK pakai micro-expression (droopy eyelids dll) — unreliable di 224x224
    (
        "a student looking away from the screen with their head turned to the side, completely disengaged\n"
        "a student slouching in their chair with wandering unfocused gaze directed upward\n"
        "a student with their head tilted back and eyes gazing at the ceiling, mentally absent\n"
        "a student resting their chin on their hand with a blank stare directed away from the screen\n"
        "a student with slouched posture and head drooping downward showing zero interest in learning\n"
        "a student gazing sideways with passive body language and mentally detached appearance",
        "",
    ),
    # 1 — Engagement
    # Prioritas spec: forward gaze, upright torso, active learning posture, stable visual attention
    # Termasuk: smiling/laughing naturally + forward attention = engagement (spec Rule 7)
    (
        "a student sitting upright and looking straight at the screen with focused attention\n"
        "a student actively typing on a keyboard while looking down at the screen with concentration\n"
        "a student with forward-facing head and stable focused gaze directed at learning content\n"
        "a student sitting with attentive upright posture and eyes fixed on the screen\n"
        "a student smiling and laughing naturally while paying attention to the screen\n"
        "a student with alert forward posture actively watching and processing screen content",
        "",
    ),
    # 2 — Confusion
    # Prioritas spec: slightly open mouth, forward lean, thinking posture, concentrated uncertainty
    # TIDAK pakai furrowed brows/squinting — unreliable di 224x224
    (
        "a student leaning forward with a slightly open mouth and an uncertain concentrated stare\n"
        "a student with their head slightly tilted showing a puzzled thinking expression\n"
        "a student staring at the screen with a slightly open mouth looking mentally stuck\n"
        "a student with a concentrated uncertain expression leaning closer to the screen\n"
        "a student pausing with a confused look and slightly parted lips while trying to understand\n"
        "a student with a tense thinking posture and uncertain gaze fixed on the screen",
        "",
    ),
    # 3 — Frustration
    # Prioritas spec: hand-face interaction, forehead touching, support posture, visible struggle gestures
    # TIDAK pakai fierce angry expression/gritting teeth — unreliable di 224x224
    (
        "a student rubbing their forehead with their hand showing visible mental stress\n"
        "a student covering their face with both hands in cognitive exhaustion\n"
        "a student pressing their hand against their forehead while staring at the screen in struggle\n"
        "a student pinching the bridge of their nose with a tense stressed posture\n"
        "a student supporting their head with their hand showing signs of mental overload\n"
        "a student with their hand on their face and tense shoulders showing visible frustration",
        "",
    ),
]


