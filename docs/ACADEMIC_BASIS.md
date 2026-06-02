# Academic Basis of the Labeling System

This document contains **verbatim quotations** from the academic papers that form the scientific foundation of each method and design decision in this labeling application. No paraphrasing — every block-quoted sentence below is copied word-for-word from the cited source.

> **Catatan kutipan:** Nomor referensi dalam tanda kurung siku seperti `[17]`, `[55]`, dan kutipan dalam teks seperti `(Carroll & Kay, 1988; ...)` dihilangkan mengikuti praktik standar pengutipan akademis. Penanda internal paper seperti `(Link 3)` atau `(Hypothesis 4)` juga dihilangkan. Selain itu, semua kata dan frasa dalam kalimat yang dikutip adalah verbatim dari sumber asli.

---

## 1. The Four Learning Emotions

**Why these four emotions (Boredom, Engagement/Flow, Confusion, Frustration)?**

> "One finding is that confusion, frustration, boredom, and engagement/flow are the major affective states that students experience across diverse learning contexts, student populations, and methods to track emotions."

— D'Mello, S.K. & Graesser, A. (2012). *Dynamics of Affective States during Complex Learning.* Learning and Instruction, 22, 145–157. (p. 148)

> "The affect-detection phase focused on the development of computational systems that monitor conversational cues, gross body language, and facial features to detect the presence of boredom, engagement, confusion, and frustration (delight and surprise were excluded because they are extremely rare)."

— D'Mello, S., Craig, S., Fike, K., & Graesser, A. (2009). *Responding to Learners' Cognitive-Affective States with Supportive and Shakeup Dialogues.* (p. 2)

> "our dataset consists of labels for four affective states related to user engagement, viz., engagement, frustration, confusion, and boredom. Recent work has shown that the six basic expressions: anger, disgust, fear, joy, sadness, and surprise are not reliable in prolonged learning situations, as they are prone to rapid changes."

— Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016). *DAiSEE: Towards User Engagement Recognition in the Wild.* arXiv:1609.01885. (§3.2)

**Definitions used in the original study where AU-to-emotion mappings were derived:**

> "Boredom—the state of being weary and restless through lack of interest."

> "Confusion—the failure to differentiate from an often similar or related other."

> "Frustration—making vain or ineffectual efforts, however vigorous; a deep chronic sense or state of insecurity and dissatisfaction arising from unresolved problems or unfulfilled needs."

— Craig, S.D., D'Mello, S., Witherspoon, A., & Graesser, A. (2008). *Emote aloud during learning with AutoTutor: Applying the Facial Action Coding System to cognitive–affective states during learning.* Cognition & Emotion, 22(5), 777–788. (p. 781)

---

## 2. Facial Action Coding System (FACS) as the Measurement Framework

**Why use FACS/Action Units instead of basic emotion categories?**

> "The Facial Action Coding System (Ekman & Friesen, 1978) is an objective method for quantifying facial movement in terms of component actions."

— Bartlett, M.S., Hager, J.C., Ekman, P., & Sejnowski, T.J. (1999). *Measuring facial expressions by computer image analysis.* Psychophysiology, 36, 253–263. (abstract)

> "FACS provides an objective and comprehensive language for describing facial expressions and relating them back to what is known about their meaning from the behavioral science literature. Because it is comprehensive, FACS also allows for the discovery of new patterns related to emotional or situational states."

— Bartlett, M.S., Littlewort, G.C., Frank, M.G., Lainscsek, C., Fasel, I.R., & Movellan, J.R. (2006). *Automatic Recognition of Facial Actions in Spontaneous Expressions.* Journal of Multimedia, 1(6), 22–35. (p. 22)

> "The Facial Action Coding System [17] is a comprehensive framework for objectively describing facial expression in terms of Action Units, which measure the intensity of over 40 distinct facial muscles. Manual FACS coding has previously been used to study student engagement and other emotions relevant to automated teaching."

— Whitehill, J., Serpell, Z., Lin, Y-C., Foster, A., & Movellan, J.R. (2014). *The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions.* IEEE Transactions on Affective Computing. (§3.1.3)

**Why FACS outperforms subjective labeling?**

> "Spontaneous facial expressions differ from posed expressions in both which muscles are moved, and in the dynamics of the movement."

— Bartlett et al. (2006). (abstract)

> "subjective labeling of expressions has been shown to be less reliable than objective coding for finding relationships between facial expression and other state variables."

— Bartlett et al. (2006). (p. 22)

---

## 3. AU-to-Emotion Mapping — Empirical Findings

**Source: Craig et al. (2008) Table 2 — association rule mining on FACS-coded data**

> "A standard data mining procedure called the *a priori* algorithm was used to identify frequent sets of action units and to extract association rules that could conditionally detect the presence of AUs on the face. Association rules are probabilistic in nature and take the form *Antecedent → Consequent [support, confidence]*."

> "The confidence measures its certainty and is the conditional probability that a data instance containing the antecedent will contain the consequent. For example, we observed an association rule with confusion of AU4 → AU7 (see Table 2) for two action units AU4 (antecedent) and AU7 (consequent). This can be interpreted as 'the presence of action unit 4 triggers action unit 7'."

— Craig et al. (2008). (p. 783)

> "Our analyses were able to determine significant relationships with AUs for frustration, confusion, and boredom. It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together. Additionally, these AUs mutually trigger each other. That is, a raised inner brow tends to trigger a raised outer brow, and vice versa. Confusion displayed associations with AUs 4, 7, and 12. Action units 4 and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow). While boredom displayed a significant association with action unit 43 (eye closure), no association rules between AUs were observed, but some weaker non-significant trends between eye movement, such as blinks and eye closure, and AUs related to mouth movement."

— Craig et al. (2008). (p. 784)

**Table 2 summary (Craig et al. 2008) — Coverage = frequency across 100 random data sets:**

| Affect | AU | Description | Coverage |
|---|---|---|---|
| Frustration | 2 | Inner brow raise | 100% |
| Frustration | 1 | Outer brow raise | 100% |
| Frustration | 1,2 | Inner and outer brow raised together | 100% |
| Frustration | 1→2 | Presence of inner brow raise triggers outer brow raise and vice versa | 100% (rule) |
| Confusion | 4 | Brow lowerer | 95% |
| Confusion | 7 | Lid tightener | 78% |
| Confusion | 4,7 | Brow lowered with tightened lids | 73% |
| Confusion | 12* | Lip corner puller | 95% |
| Boredom | 43* | Eye closure | 40% |

*Note in original table: → implies an association rule with 100% confidence; \*Of secondary importance.*

> "The two expert raters received an overall kappa that ranged between .76 and .84."

— Craig et al. (2008). (abstract)

---

## 4. FRUSTRATION — Brow Raise (AU1+AU2) as Primary Signal

> "It appears that AUs 1, 2, and 14 were primarily associated with frustration, but a strong association was found for a link between AUs 1 and 2 occurring together."

— Craig et al. (2008). (p. 784)

> "Negative affective states, such as the frustration, disappointment, or anger can occur when a learner is stuck at an impasse or in reaction to feedback from the learning environment."

— D'Mello, S.K., Blanchard, N., Baker, R., Ocumpaugh, J., & Brawner, K. (2014). *I Feel Your Pain: A Selective Review of Affect-Sensitive Instructional Strategies.* (p. 1)

> "Hopeless confusion occurs when the impasse cannot be resolved, the student gets stuck, there is no available plan, and important goals are blocked. The model hypothesizes that learners will experience frustration in these situations."

— D'Mello & Graesser (2012). (p. 147)

---

## 5. CONFUSION — AU4+AU7 Co-occurrence and AU12

> "Confusion displayed associations with AUs 4, 7, and 12. Action units 4 and 7 occur simultaneously and the presence of AU7 (tightened lids) tends to trigger AU4 (lowered brow)."

— Craig et al. (2008). (p. 784)

> "Learners experience cognitive disequilibrium when they are confronted with a contradiction, anomaly, system breakdown, or error, and when they are uncertain about what to do next. Confusion is a key signature of the cognitive disequilibrium that occurs when an impasse is detected."

— D'Mello & Graesser (2012). (p. 146)

> "The second hypothesis is the productive confusion hypothesis (Hypothesis 2). According to this hypothesis, cognitive disequilibrium, impasses, and confusion provide learners with an opportunity to think, deliberate, and problem solve."

— D'Mello & Graesser (2012). (p. 149)

> "Confusion is considered to be the affective signature of these states. Therefore, one hypothesis is that events that confuse learners might provide valuable learning opportunities because learners need to engage in deep cognitive activities in order to resolve their confusion."

— D'Mello et al. (2014). (p. 6)

**Why Confusion and Engagement can co-exist:**

> "The above form of productive confusion associated with impasse resolution can be contrasted with hopeless confusion."

> "The Confusion → Engagement/Flow and Confusion → Frustration transitions were also significant, while the Confusion → Boredom transition occurred at chance levels, thereby confirming the productive confusion and hopeless confusion hypotheses."

— D'Mello & Graesser (2012). (p. 153)

---

## 6. BOREDOM — AU43 (Eye Closure) and Disengagement

> "While boredom displayed a significant association with action unit 43 (eye closure), no association rules between AUs were observed."

— Craig et al. (2008). (p. 784)

> "The fourth hypothesis, or the disengagement hypothesis (Hypothesis 4), states that persistent failure, which is related to frustration, eventually transitions into disengagement and boredom. According to this fourth hypothesis there should be a transition from frustration to boredom (Link 4), but frustration should not transition into engagement/flow."

— D'Mello & Graesser (2012). (p. 149)

> "Furthermore, consistent with forced-effort theories of boredom (Larson & Richards, 1991; Robinson, 1975), persistent frustration may transition into boredom, a crucial point at which the learner disengages from the learning process."

— D'Mello & Graesser (2012). (p. 147)

---

## 7. ENGAGEMENT/FLOW — Facial Appearance as Primary Signal

> "Student engagement is a key concept in contemporary education, where it is valued as a goal in its own right."

> "We found that human observers reliably agree when discriminating low versus high degrees of engagement (Cohen's κ = 0.96). When fine discrimination is required (4 distinct levels) the reliability decreases, but is still quite high (κ = 0.56). Furthermore, we found that engagement labels of 10-second video clips can be reliably predicted from the average labels of their constituent frames (Pearson r = 0.85), suggesting that static expressions contain the bulk of the information about engagement used by observers."

— Whitehill et al. (2014). (abstract)

> "This accuracy is quite high and suggests that most of the information about the appearance of engagement is contained in the static pixels, not the motion per se."

— Whitehill et al. (2014). (§2.4)

> "We hypothesize that a good deal of the information used by humans to make engagement judgements is based on the student's face."

— Whitehill et al. (2014). (p. 2)

**Engagement scale used in labeling (Whitehill et al. 2014):**

> "1: Not engaged at all – e.g., looking away from computer and obviously not thinking about task, eyes completely closed."
> "2: Nominally engaged – e.g., eyes barely open, clearly not 'into' the task."
> "3: Engaged in task – student requires no admonition to 'stay on task'."
> "4: Very engaged – student could be 'commended' for his/her level of engagement in task."

— Whitehill et al. (2014). (§2.2)

**Why engagement is modeled as a baseline state:**

> "Engagement/flow is a cognitive-affective state that sometimes has a short time span, but at other times forms part of Csikszentmihalyi's (1990) conception of flow. It is important to point out that a learner can be engaged without necessarily experiencing flow; for example, being engaged in order to avoid failure when one is anxious."

— D'Mello & Graesser (2012). (p. 146)

---

## 8. Multimodal Detection — Why Face + Body

> "The new versions of AutoTutor detect learners' boredom, confusion, and frustration by monitoring conversational cues, gross body language, and facial features."

— D'Mello et al. (2009). *Responding to Learners' Cognitive-Affective States.* (abstract)

> "The affective states are sensed by monitoring conversational cues and other discourse features, gross body movements, and facial features."

— D'Mello et al. (2014). (p. 4)

---

## 9. Dataset Reference — DAiSEE

> "we introduce DAiSEE, the first multi-label video classification dataset comprising of 9068 video snippets captured from 112 users for recognizing the user affective states of boredom, confusion, engagement, and frustration 'in the wild'. The dataset has four levels of labels namely - very low, low, high, and very high for each of the affective states, which are crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists."

— Gupta et al. (2016). *DAiSEE.* (abstract)

---

## Full Paper References

1. **Craig, S.D., D'Mello, S., Witherspoon, A., & Graesser, A. (2008).** Emote aloud during learning with AutoTutor: Applying the Facial Action Coding System to cognitive–affective states during learning. *Cognition & Emotion*, 22(5), 777–788.

2. **D'Mello, S.K. & Graesser, A. (2012).** Dynamics of Affective States during Complex Learning. *Learning and Instruction*, 22, 145–157.

3. **D'Mello, S., Craig, S., Fike, K., & Graesser, A. (2009).** Responding to Learners' Cognitive-Affective States with Supportive and Shakeup Dialogues. *AIED 2009.*

4. **D'Mello, S.K., Blanchard, N., Baker, R., Ocumpaugh, J., & Brawner, K. (2014).** I Feel Your Pain: A Selective Review of Affect-Sensitive Instructional Strategies. *International handbook on metacognition and learning technologies.*

5. **Whitehill, J., Serpell, Z., Lin, Y-C., Foster, A., & Movellan, J.R. (2014).** The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions. *IEEE Transactions on Affective Computing.*

6. **Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016).** DAiSEE: Towards User Engagement Recognition in the Wild. *arXiv:1609.01885.*

7. **Bartlett, M.S., Hager, J.C., Ekman, P., & Sejnowski, T.J. (1999).** Measuring facial expressions by computer image analysis. *Psychophysiology*, 36, 253–263.

8. **Bartlett, M.S., Littlewort, G.C., Frank, M.G., Lainscsek, C., Fasel, I.R., & Movellan, J.R. (2006).** Automatic Recognition of Facial Actions in Spontaneous Expressions. *Journal of Multimedia*, 1(6), 22–35.
