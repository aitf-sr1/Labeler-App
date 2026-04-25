LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # 3 prompt NOLEH (kepala ke samping) + 3 prompt MATA WANDERING (kepala lurus tapi mata lirik)
    (
        "a student looking to the side with a blank disinterested bored expression\n"
        "a student with their face turned showing only their cheek appearing completely bored\n"
        "a student with head turned away and a slack tired unfocused look\n"
        "a student staring off with wandering unfocused eyes and a dull bored expression\n"
        "a student with eyes glancing sideways appearing distracted and mentally absent\n"
        "a student with a vacant distant stare and droopy heavy eyelids looking disengaged",
        "",
    ),
    # 1 — Engagement
    # Fokus pada EYE CONTACT dan ekspresi attentif — bukan "centered" (crop selalu centered!)
    (
        "a student making direct eye contact with clear focus and engaged attention\n"
        "a student with bright alert eyes actively watching the screen ahead\n"
        "a student with an attentive upright posture and focused interested expression\n"
        "a student nodding and reacting while looking straight at the screen\n"
        "a student leaning forward with curious raised eyebrows showing interest\n"
        "a student with a lively responsive face showing understanding and involvement",
        "",
    ),
    # 2 — Confusion
    # Furrowed brow + tilted head + searching upward + mouth agak terbuka (ngelamun)
    (
        "a student with deeply furrowed eyebrows and a lost puzzled uncertain expression\n"
        "a student tilting their head sideways with a questioning and confused look\n"
        "a student with mouth slightly open and eyes searching upward as if stuck\n"
        "a student with a blank daydreaming stare and slightly slack open mouth\n"
        "a student squinting with visible mental effort and a concentrated frown\n"
        "a student with a tense forehead and wide bewildered eyes unable to understand",
        "",
    ),
    # 3 — Frustration
    # Ekspresi stress halus — bukan hanya gesture ekstrem (tutup muka)
    (
        "a student with tightly furrowed brows and a strained pained frustrated expression\n"
        "a student appearing stressed and tense with a clenched jaw and pressed lips\n"
        "a student with a deeply annoyed irritated face showing visible displeasure\n"
        "a student sighing heavily with closed eyes and a defeated exhausted look\n"
        "a student touching their forehead or rubbing eyes from mental fatigue\n"
        "a student with squinting eyes and a stiff rigid frustrated expression",
        "",
    ),
]


