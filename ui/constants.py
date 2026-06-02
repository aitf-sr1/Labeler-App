LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_COLORS = {
    "Boredom":    "#fbbf24",
    "Engagement": "#10b981",
    "Confusion":  "#3b82f6",
    "Frustration":"#ef4444",
}

DEFAULT_PROMPT_GROUPS = [
    # 0 — Boredom
    # Paper basis:
    #   Craig et al. (2008): AU43 (eye closure) = primary boredom signal (40% coverage)
    #   Whitehill et al. (2014) §2.2: level 1 = "looking away from computer... eyes completely closed"
    #   D'Mello & Graesser (2012): boredom = disengagement, expressionless face
    (
        "a student with heavy drooping eyelids and half-closed eyes showing fatigue and disengagement\n"
        "a student with glazed unfocused eyes staring blankly with a completely expressionless face\n"
        "a student looking away from the screen with an empty blank expression mentally absent\n"
        "a student with eyes barely open and a vacant expressionless stare directed away from the screen\n"
        "a student gazing sideways or upward with droopy heavy eyelids showing total disengagement\n"
        "a student with closed or nearly closed eyes and an expressionless face mentally detached from the task",
        "",
    ),
    # 1 — Engagement
    # Paper basis:
    #   Whitehill et al. (2014): holistic facial appearance, static pixels, level 3-4 engagement
    #   Level 3: "student requires no admonition to stay on task"
    #   Level 4: "student could be commended for his/her level of engagement"
    #   Pearson r=0.85 between frame-level and clip-level engagement labels
    (
        "a student sitting upright and looking straight at the screen with focused alert eyes\n"
        "a student actively concentrating on learning content with full visual attention on the screen\n"
        "a student with wide open attentive eyes and forward-facing head engaged with screen content\n"
        "a student with focused steady gaze directed at the learning material showing active participation\n"
        "a student sitting with attentive upright posture and eyes fixed intently on the screen\n"
        "a student actively processing learning content with alert expression and stable forward attention",
        "",
    ),
    # 2 — Confusion
    # Paper basis:
    #   Craig et al. (2008): AU4 (brow lowerer) 95% coverage, AU7 (lid tightener) 78%, AU4+AU7 co-occur 73%
    #   AU12 (questioning smile/lip corner puller) 95% coverage — secondary importance
    #   D'Mello & Graesser (2012): cognitive disequilibrium, impasse, uncertain knowledge state
    (
        "a student with furrowed lowered brows and squinting tightened eyelids showing cognitive uncertainty\n"
        "a student with a furrowed brow and tightened eyelids concentrating on a difficult problem\n"
        "a student displaying lowered brows with slightly squinting eyes in a confused uncertain state\n"
        "a student with brow furrowed and eyes squinting in an uncertain concentrated thinking state\n"
        "a student with a questioning slight smile and furrowed lowered brows showing cognitive uncertainty\n"
        "a student with tightened eyelids and a lowered furrowed brow staring at the screen mentally stuck",
        "",
    ),
    # 3 — Frustration
    # Paper basis:
    #   Craig et al. (2008): AU1 (inner brow raise) + AU2 (outer brow raise) = 100% coverage, mutually trigger
    #   Grafsgaard et al. (2013): AU4 (brow lowering) positively correlated with frustration
    #   Grafsgaard et al. (2013): AU14 (mouth dimpling) positively correlated with frustration
    #   D'Mello & Graesser (2012): impasse, state of stuck, hopeless confusion
    (
        "a student with both inner and outer eyebrows raised high showing anxiety and mental struggle\n"
        "a student with raised inner and outer brows and a tense distressed expression during learning\n"
        "a student displaying raised worried eyebrows with a tense strained expression stuck on a problem\n"
        "a student with elevated brows and tense facial muscles showing visible struggle with a problem\n"
        "a student with raised brows and slight dimpling at the mouth corners showing cognitive overload\n"
        "a student with a tense furrowed worried brow expression when unable to resolve a difficulty",
        "",
    ),
]


