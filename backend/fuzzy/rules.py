"""
Fuzzy Rule Base and Context-Aware Rule Evaluator for SmartSnap V3.
Mamdani min-operator evaluation for antecedents.
Ensures that no positive rule for exposure or composition can overwhelm a severe blur penalty.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class FuzzyRule:
    """
    Represents a Mamdani fuzzy IF-THEN rule.
    antecedents: list of (variable_name, linguistic_term) which are combined with Mamdani AND (min).
    consequent: (variable_name, linguistic_term)
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        antecedents: List[tuple],  # e.g. [("blur", "Low"), ("brightness", "Normal")]
        consequent: tuple,         # e.g. ("quality", "Excellent")
        category_scope: Optional[List[str]] = None,  # None means applies to all categories
        weight: float = 1.0,
        explanation: str = ""
    ):
        self.rule_id = rule_id
        self.name = name
        self.antecedents = antecedents
        self.consequent = consequent
        self.category_scope = category_scope
        self.weight = weight
        self.explanation = explanation

    def is_applicable_to_category(self, category: str) -> bool:
        if self.category_scope is None or "all" in self.category_scope:
            return True
        return category.lower() in [c.lower() for c in self.category_scope]

    def evaluate_firing_strength(self, memberships: Dict[str, Dict[str, float]], category: str = "General Photography") -> float:
        """
        Calculate alpha (firing strength) using Mamdani min-operator:
        alpha = min( mu_A1(x1), mu_A2(x2), ... ) * weight
        """
        if not self.is_applicable_to_category(category):
            return 0.0

        degrees = []
        for var_name, term_name in self.antecedents:
            var_mems = memberships.get(var_name, {})
            deg = var_mems.get(term_name, 0.0)
            degrees.append(deg)

        if not degrees:
            return 0.0

        alpha = float(np.min(degrees)) * self.weight
        return float(np.round(alpha, 4))


def build_default_rule_base() -> List[FuzzyRule]:
    """
    Builds the structured Mamdani rule base covering:
    - Core quality rules
    - Severe blur and focus degradation rules
    - Context-aware rules for Portrait, Group, Landscape, Document, General
    - Dead-zone prevention universal baselines
    """
    rules: List[FuzzyRule] = []

    # ==========================================
    # 1. Critical Severe Quality Degradation Rules
    # ==========================================
    rules.append(FuzzyRule(
        "R1", "Severe Blur & Noise Hazard",
        [("blur", "High"), ("noise", "High")],
        ("quality", "Very Poor"),
        weight=1.0,
        explanation="High blur coupled with severe noise renders details completely irrecoverable."
    ))

    rules.append(FuzzyRule(
        "R2", "Blown Exposure with Flat Contrast",
        [("brightness", "Overexposed"), ("contrast", "Low")],
        ("quality", "Poor"),
        weight=0.90,
        explanation="Overexposure combined with low contrast destroys highlight detail."
    ))

    rules.append(FuzzyRule(
        "R3", "Crushed Shadows with Flat Contrast",
        [("brightness", "Dark"), ("contrast", "Low")],
        ("quality", "Poor"),
        weight=0.90,
        explanation="Dark lighting with low contrast crushes shadow detail."
    ))

    rules.append(FuzzyRule(
        "R4", "Severe Blur Degradation",
        [("blur", "High")],
        ("quality", "Very Poor"),
        weight=1.0,
        explanation="Severe optical defocus or motion blur directly destroys photographic usability."
    ))

    rules.append(FuzzyRule(
        "R5", "Severe Noise Artifacts",
        [("noise", "High")],
        ("quality", "Poor"),
        weight=0.85,
        explanation="Excessive sensor noise reduces image fidelity."
    ))

    # ==========================================
    # 2. Portrait & Group Photography Specific Rules
    # ==========================================
    rules.append(FuzzyRule(
        "R6", "Portrait Blur with Obscured Face",
        [("blur", "High"), ("face_visibility", "Poor")],
        ("quality", "Very Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=1.0,
        explanation="Unclear subject face and high blur make this portrait unusable."
    ))

    rules.append(FuzzyRule(
        "P1", "Severe Blur on Prominently Framed Portrait",
        [("blur", "High"), ("face_visibility", "Clear")],
        ("quality", "Very Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=1.0,
        explanation="A prominent portrait face that suffers from motion blur or optical defocus is completely spoiled."
    ))

    rules.append(FuzzyRule(
        "P2", "Severe Blur on Detected Portrait Face",
        [("blur", "High"), ("face_visibility", "Acceptable")],
        ("quality", "Very Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=1.0,
        explanation="Detected subject face is heavily blurred; photographic intent is lost."
    ))

    rules.append(FuzzyRule(
        "P3", "Mild Portrait Blur on Clear Subject",
        [("blur", "Medium"), ("face_visibility", "Clear")],
        ("quality", "Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.85,
        explanation="Noticeable softening or motion smear on a clear portrait subject degrades quality below keeper standard."
    ))

    rules.append(FuzzyRule(
        "P4", "Mild Portrait Blur on Visible Subject",
        [("blur", "Medium"), ("face_visibility", "Acceptable")],
        ("quality", "Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.80,
        explanation="Subject face lacks crisp eye and eyelash resolution; difficult to salvage in portraiture."
    ))

    rules.append(FuzzyRule(
        "R7", "Clear Sharp Portrait with Great Framing",
        [("blur", "Low"), ("face_visibility", "Clear"), ("composition", "Good")],
        ("quality", "Excellent"),
        category_scope=["Portrait", "Group Photograph"],
        weight=1.0,
        explanation="Crisp eye focus, well-lit prominent facial features, and balanced framing make a stunning portrait."
    ))

    rules.append(FuzzyRule(
        "P5", "Sharp Focus Portrait Baseline",
        [("blur", "Low"), ("face_visibility", "Clear")],
        ("quality", "Good"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.90,
        explanation="Clear facial features resolved with crisp subject sharpness."
    ))

    rules.append(FuzzyRule(
        "P6", "Acceptable Sharp Portrait",
        [("blur", "Low"), ("face_visibility", "Acceptable")],
        ("quality", "Good"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.80,
        explanation="Sharp focus on recognized portrait subject."
    ))

    rules.append(FuzzyRule(
        "R8", "Portrait with Poor Composition and Obscured Subject",
        [("composition", "Poor"), ("face_visibility", "Poor")],
        ("quality", "Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.80,
        explanation="Awkward composition and unreadable face positioning."
    ))

    rules.append(FuzzyRule(
        "R9", "Decent Portrait Lighting but Medium Blur",
        [("blur", "Medium"), ("brightness", "Normal"), ("face_visibility", "Acceptable")],
        ("quality", "Moderate"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.75,
        explanation="Acceptable lighting and face detection, but mild blur holds it back."
    ))

    # ==========================================
    # 3. Landscape Specific Rules (Faces NOT required)
    # ==========================================
    rules.append(FuzzyRule(
        "R10", "Pristine Landscape Clarity",
        [("blur", "Low"), ("brightness", "Normal"), ("contrast", "High"), ("noise", "Low"), ("composition", "Good")],
        ("quality", "Excellent"),
        category_scope=["Landscape", "General Photography"],
        weight=1.0,
        explanation="Crystal clarity, rich dynamic range, clean shadows, and great composition for scenery."
    ))

    rules.append(FuzzyRule(
        "R11", "Good Landscape Balanced Exposure",
        [("blur", "Low"), ("brightness", "Normal"), ("contrast", "Medium")],
        ("quality", "Good"),
        category_scope=["Landscape", "General Photography"],
        weight=0.90,
        explanation="Well-exposed scene with solid sharpness."
    ))

    rules.append(FuzzyRule(
        "R12", "Hazy or Low Contrast Landscape",
        [("contrast", "Low"), ("noise", "Medium")],
        ("quality", "Moderate"),
        category_scope=["Landscape"],
        weight=0.75,
        explanation="Low contrast or haze limits scenery impact, but can be recovered."
    ))

    rules.append(FuzzyRule(
        "R13", "Blurry Scenery",
        [("blur", "Medium"), ("composition", "Average")],
        ("quality", "Moderate"),
        category_scope=["Landscape"],
        weight=0.75,
        explanation="Landscape lacks crisp edge definition."
    ))

    # ==========================================
    # 4. Document Specific Rules (Text readability)
    # ==========================================
    rules.append(FuzzyRule(
        "R14", "Razor-Sharp Readable Document",
        [("blur", "Low"), ("contrast", "High"), ("noise", "Low")],
        ("quality", "Excellent"),
        category_scope=["Document"],
        weight=1.0,
        explanation="High text contrast, sharp letter boundaries, and clean background."
    ))

    rules.append(FuzzyRule(
        "R15", "Blurry Unreadable Document",
        [("blur", "High")],
        ("quality", "Very Poor"),
        category_scope=["Document"],
        weight=1.0,
        explanation="Document text is illegible due to blur."
    ))

    rules.append(FuzzyRule(
        "R16", "Low Contrast Document",
        [("contrast", "Low"), ("brightness", "Dark")],
        ("quality", "Poor"),
        category_scope=["Document"],
        weight=0.85,
        explanation="Dim lighting and low contrast make characters faint."
    ))

    rules.append(FuzzyRule(
        "R17", "Adequate Document with Minor Grain",
        [("blur", "Low"), ("contrast", "Medium"), ("brightness", "Normal")],
        ("quality", "Good"),
        category_scope=["Document"],
        weight=0.85,
        explanation="Document is fully readable with balanced lighting."
    ))

    # ==========================================
    # 5. General Photography Rules (Strict Sharpness Enforcement)
    # Note: Every rule leading to Good or Excellent quality REQUIRES blur is Low!
    # ==========================================
    rules.append(FuzzyRule(
        "R18", "Ideal General Photo",
        [("blur", "Low"), ("brightness", "Normal"), ("composition", "Good")],
        ("quality", "Excellent"),
        category_scope=["all"],
        weight=1.0,
        explanation="Sharp focus, optimal exposure, and pleasing composition."
    ))

    rules.append(FuzzyRule(
        "R19", "Sharp, Clean, High Contrast Photo",
        [("blur", "Low"), ("brightness", "Normal"), ("noise", "Low"), ("contrast", "High")],
        ("quality", "Good"),
        category_scope=["all"],
        weight=0.90,
        explanation="Sharp focus, clean sensor, and rich contrast."
    ))

    rules.append(FuzzyRule(
        "R20", "Medium Blur Balanced Exposure",
        [("blur", "Medium"), ("brightness", "Normal")],
        ("quality", "Moderate"),
        category_scope=["all"],
        weight=0.75,
        explanation="Exposure is solid, but mild softening or motion blur is present."
    ))

    rules.append(FuzzyRule(
        "R21", "Under-exposed but Sharp Photo",
        [("blur", "Low"), ("brightness", "Dark"), ("noise", "Low")],
        ("quality", "Moderate"),
        category_scope=["all"],
        weight=0.80,
        explanation="Sharp details preserved in dark exposure; ideal candidate for digital recovery."
    ))

    rules.append(FuzzyRule(
        "R22", "Over-exposed but Sharp Photo",
        [("blur", "Low"), ("brightness", "Overexposed")],
        ("quality", "Moderate"),
        category_scope=["all"],
        weight=0.75,
        explanation="Sharp photograph with highlight clipping; partially recoverable."
    ))

    rules.append(FuzzyRule(
        "R23", "Good Composition with High Noise",
        [("composition", "Good"), ("noise", "High"), ("blur", "Low")],
        ("quality", "Moderate"),
        category_scope=["all"],
        weight=0.75,
        explanation="Artistic framing and sharpness degraded by grain or sensor noise."
    ))

    rules.append(FuzzyRule(
        "R24", "Well-Balanced Average Snapshot",
        [("blur", "Low"), ("brightness", "Normal"), ("contrast", "Medium"), ("noise", "Low")],
        ("quality", "Good"),
        category_scope=["all"],
        weight=0.85,
        explanation="Solid everyday snapshot meeting all standard photographic criteria."
    ))

    rules.append(FuzzyRule(
        "R25", "Marginal Characteristics",
        [("blur", "Medium"), ("contrast", "Medium"), ("noise", "Medium")],
        ("quality", "Moderate"),
        category_scope=["all"],
        weight=0.70,
        explanation="Multiple characteristics are in the borderline/moderate range."
    ))

    rules.append(FuzzyRule(
        "R26", "Sub-Par Exposure and Noise",
        [("brightness", "Dark"), ("noise", "High")],
        ("quality", "Poor"),
        category_scope=["all"],
        weight=0.80,
        explanation="Dark capture marred by amplified sensor noise in shadows."
    ))

    rules.append(FuzzyRule(
        "R27", "Overexposed Scene Penalty",
        [("brightness", "Overexposed")],
        ("quality", "Poor"),
        weight=0.75,
        category_scope=["all"],
        explanation="Overexposure alone is a meaningful quality penalty; highlight detail is irrecoverable."
    ))

    rules.append(FuzzyRule(
        "R28", "Dark but Sharp Photo",
        [("blur", "Low"), ("brightness", "Dark")],
        ("quality", "Moderate"),
        weight=0.80,
        category_scope=["all"],
        explanation="Dark exposure with retained sharpness is salvageable via post-processing lift."
    ))

    rules.append(FuzzyRule(
        "R29", "Sharp Normal Exposure Baseline",
        [("blur", "Low"), ("brightness", "Normal"), ("noise", "Low")],
        ("quality", "Good"),
        weight=0.85,
        category_scope=["all"],
        explanation="Sharp focus, clean exposure, and minimal noise satisfy fundamental photographic criteria."
    ))

    rules.append(FuzzyRule(
        "R30", "Medium Blur with High Contrast",
        [("blur", "Medium"), ("contrast", "High")],
        ("quality", "Moderate"),
        weight=0.75,
        category_scope=["all"],
        explanation="Good tonal range but some softness present; overall image is borderline acceptable."
    ))

    rules.append(FuzzyRule(
        "R31", "Overexposed with High Contrast",
        [("brightness", "Overexposed"), ("contrast", "High")],
        ("quality", "Poor"),
        weight=0.80,
        category_scope=["all"],
        explanation="Blown highlights with harsh contrast makes highlight recovery unlikely."
    ))

    rules.append(FuzzyRule(
        "R32", "Overexposed with Medium Blur",
        [("brightness", "Overexposed"), ("blur", "Medium")],
        ("quality", "Poor"),
        weight=0.80,
        category_scope=["all"],
        explanation="Dual penalties from blown highlights and softening jointly degrade image usability."
    ))

    rules.append(FuzzyRule(
        "R33", "Sharp Clean High Contrast Baseline",
        [("blur", "Low"), ("brightness", "Normal"), ("noise", "Low"), ("contrast", "High")],
        ("quality", "Good"),
        weight=0.80,
        category_scope=["all"],
        explanation="Sharp focus, low sensor noise, and rich contrast produce a technically solid image."
    ))

    rules.append(FuzzyRule(
        "R34", "Poor Composition but Sharp",
        [("composition", "Poor"), ("blur", "Low")],
        ("quality", "Moderate"),
        weight=0.75,
        category_scope=["all"],
        explanation="Awkward framing holds back an otherwise technically sharp photograph."
    ))

    rules.append(FuzzyRule(
        "R35", "Overexposed but Sharp and Clean",
        [("brightness", "Overexposed"), ("blur", "Low"), ("noise", "Low")],
        ("quality", "Moderate"),
        weight=0.75,
        category_scope=["all"],
        explanation="Sharp and clean, but exposure blow-out needs recovery."
    ))

    rules.append(FuzzyRule(
        "R36", "Medium Blur Universal Fallback",
        [("blur", "Medium")],
        ("quality", "Moderate"),
        weight=0.65,
        category_scope=["all"],
        explanation="Moderate blur as a universal fallback ensures any medium-blur combination receives Moderate quality."
    ))

    rules.append(FuzzyRule(
        "R37", "Sharp Well-Composed Normal Exposure",
        [("blur", "Low"), ("composition", "Good"), ("brightness", "Normal")],
        ("quality", "Good"),
        weight=0.80,
        category_scope=["all"],
        explanation="Sharp focus and well-framed subject with balanced exposure."
    ))

    rules.append(FuzzyRule(
        "R38", "Overexposed with Moderate Contrast and Sharp Focus",
        [("blur", "Low"), ("brightness", "Overexposed"), ("contrast", "Medium")],
        ("quality", "Moderate"),
        weight=0.70,
        category_scope=["all"],
        explanation="Sharp focus with highlight clipping and moderate contrast — partially recoverable."
    ))

    # ==========================================
    # 6. Universal Dead-Zone Immunity Baselines
    # Guarantees that EVERY value of blur in [0, 100] fires at least one rule!
    # ==========================================
    rules.append(FuzzyRule(
        "U1", "Sharp Focus Baseline",
        [("blur", "Low"), ("brightness", "Normal")],
        ("quality", "Good"),
        weight=0.75,
        category_scope=["all"],
        explanation="Baseline reward for crisp focus with balanced normal exposure."
    ))

    rules.append(FuzzyRule(
        "U2", "Sharp Focus with Rich Contrast",
        [("blur", "Low"), ("brightness", "Normal"), ("contrast", "High")],
        ("quality", "Good"),
        weight=0.75,
        category_scope=["all"],
        explanation="Sharp focus and rich contrast support good quality when exposure is also in the normal range."
    ))

    rules.append(FuzzyRule(
        "R39", "Elevated noise in a sharp image",
        [("noise", "Medium"), ("blur", "Low")],
        ("quality", "Moderate"),
        weight=0.85,
        category_scope=["all"],
        explanation="Noticeable sensor noise remains a recovery concern even when focus and contrast are strong."
    ))

    # ==========================================
    # 7. Hybrid Random Forest MOS rules
    # ==========================================
    rules.append(FuzzyRule(
        "ML1", "Low Random Forest Mean Opinion Score",
        [("predicted_mos", "Low")],
        ("quality", "Poor"),
        weight=0.90,
        category_scope=["all"],
        explanation="The Random Forest predicts a low mean human-opinion score on KonIQ's 1-5 scale."
    ))

    rules.append(FuzzyRule(
        "ML2", "Mid-Range Random Forest Mean Opinion Score",
        [("predicted_mos", "Medium")],
        ("quality", "Moderate"),
        weight=0.90,
        category_scope=["all"],
        explanation="The Random Forest predicts a mid-range mean human-opinion score."
    ))

    rules.append(FuzzyRule(
        "ML3", "High Random Forest Mean Opinion Score",
        [("predicted_mos", "High")],
        ("quality", "Good"),
        weight=0.90,
        category_scope=["all"],
        explanation="The Random Forest predicts a high mean human-opinion score."
    ))

    rules.append(FuzzyRule(
        "ML4", "High Predicted MOS with Good Composition",
        [("predicted_mos", "High"), ("composition", "Good")],
        ("quality", "Excellent"),
        weight=0.85,
        category_scope=["all"],
        explanation="A high learned perceptual score is supported by independently measured balanced framing."
    ))

    rules.append(FuzzyRule(
        "ML5", "Low Predicted MOS with Poor Composition",
        [("predicted_mos", "Low"), ("composition", "Poor")],
        ("quality", "Very Poor"),
        weight=0.85,
        category_scope=["all"],
        explanation="A low learned perceptual score is compounded by independently measured poor framing."
    ))

    rules.append(FuzzyRule(
        "ML6", "Low Predicted MOS with Poor Portrait Face Visibility",
        [("predicted_mos", "Low"), ("face_visibility", "Poor")],
        ("quality", "Very Poor"),
        category_scope=["Portrait", "Group Photograph"],
        weight=0.85,
        explanation="A low learned perceptual score and poor face visibility jointly degrade a portrait."
    ))

    return rules


def build_hybrid_rule_base() -> List[FuzzyRule]:
    """Rules for RF MOS combined with direct focus and category-aware context."""
    rules = [
        FuzzyRule(
            "H1", "Severe blur dominates a moderate MOS",
            [("blur", "High")],
            ("quality", "Very Poor"),
            category_scope=["all"],
            explanation="Severe focus loss is directly observable and cannot be canceled by a moderate learned MOS.",
        ),
        FuzzyRule(
            "H2", "Low MOS baseline",
            [("predicted_mos", "Low")],
            ("quality", "Poor"),
            weight=0.90,
            category_scope=["all"],
            explanation="The regressor predicts a low mean human-opinion score; this is not itself a retake label.",
        ),
        FuzzyRule(
            "H3", "Mid MOS baseline",
            [("predicted_mos", "Medium")],
            ("quality", "Moderate"),
            weight=0.80,
            category_scope=["all"],
            explanation="A mid-distribution MOS is neutral-to-acceptable evidence, not an automatic recovery decision.",
        ),
        FuzzyRule(
            "H4", "High MOS baseline",
            [("predicted_mos", "High")],
            ("quality", "Good"),
            weight=0.85,
            category_scope=["all"],
            explanation="A high learned perceptual score supports good quality.",
        ),
        FuzzyRule(
            "H7", "Mid MOS with sharp framed portrait",
            [("predicted_mos", "Medium"), ("blur", "Low"), ("face_visibility", "Clear"), ("composition", "Average")],
            ("quality", "Good"),
            weight=0.90,
            category_scope=["Portrait", "Group Photograph"],
            explanation="A typical MOS is supported by crisp focus, visible faces, and acceptable portrait framing.",
        ),
        FuzzyRule(
            "H8", "High MOS with sharp focus",
            [("predicted_mos", "High"), ("blur", "Low")],
            ("quality", "Excellent"),
            weight=0.95,
            category_scope=["Landscape", "General Photography", "Document"],
            explanation="High learned perceptual quality is corroborated by sharp focus; portrait intent is handled with face-specific rules.",
        ),
        FuzzyRule(
            "H9", "Medium blur with mid MOS",
            [("predicted_mos", "Medium"), ("blur", "Medium")],
            ("quality", "Moderate"),
            weight=0.85,
            category_scope=["all"],
            explanation="Moderate softness with an average perceptual estimate generally remains recoverable.",
        ),
        FuzzyRule(
            "H10", "Low MOS with severe blur",
            [("predicted_mos", "Low"), ("blur", "High")],
            ("quality", "Very Poor"),
            category_scope=["all"],
            explanation="Low perceptual quality and severe focus loss jointly support the lowest fuzzy quality term.",
        ),
        FuzzyRule(
            "H11", "Poor portrait visibility with low MOS",
            [("predicted_mos", "Low"), ("face_visibility", "Poor")],
            ("quality", "Very Poor"),
            category_scope=["Portrait", "Group Photograph"],
            explanation="Poor face visibility matters for portrait/group intent, not landscape or document images.",
        ),
        FuzzyRule(
            "H12", "Poor framing with low MOS",
            [("predicted_mos", "Low"), ("composition", "Poor")],
            ("quality", "Very Poor"),
            category_scope=["all"],
            explanation="Low perceptual quality combined with clearly poor category-aware framing.",
        ),
        FuzzyRule(
            "H13", "Mid MOS and sharp readable document",
            [("predicted_mos", "Medium"), ("blur", "Low")],
            ("quality", "Excellent"),
            weight=0.85,
            category_scope=["Document"],
            explanation="Readable sharpness and an average perceptual score support retaining a document image.",
        ),
        FuzzyRule(
            "H14", "High MOS with sharp readable document",
            [("predicted_mos", "High"), ("blur", "Low")],
            ("quality", "Excellent"),
            weight=0.95,
            category_scope=["Document"],
            explanation="High perceptual quality and sharp text support excellent document capture.",
        ),
        FuzzyRule(
            "H15", "Mid MOS with sharp non-portrait scene",
            [("predicted_mos", "Medium"), ("blur", "Low")],
            ("quality", "Excellent"),
            weight=0.85,
            category_scope=["Landscape", "General Photography"],
            explanation="The population-typical MOS is supported by clear focus; scene and object quality do not require detected faces.",
        ),
        FuzzyRule(
            "H16", "High MOS with sharp visible portrait subject",
            [("predicted_mos", "High"), ("blur", "Low"), ("face_visibility", "Clear")],
            ("quality", "Excellent"),
            weight=0.95,
            category_scope=["Portrait", "Group Photograph"],
            explanation="High perceptual quality is corroborated by sharp focus and visible portrait subjects.",
        ),
        FuzzyRule(
            "H17", "High estimated noise degrades fidelity",
            [("noise", "High")],
            ("quality", "Poor"),
            weight=0.85,
            category_scope=["all"],
            explanation="The pixel-based noise estimate indicates substantial sensor or compression artifacts.",
        ),
        FuzzyRule(
            "H18", "Moderate noise in an otherwise sharp image",
            [("noise", "Medium"), ("blur", "Low")],
            ("quality", "Moderate"),
            weight=0.80,
            category_scope=["all"],
            explanation="Noticeable noise merits recovery even when the image is in focus.",
        ),
        FuzzyRule(
            "H19", "Sharp, clean, clear, well-framed portrait",
            [
                ("blur", "Low"),
                ("noise", "Low"),
                ("face_visibility", "Clear"),
                ("composition", "Good"),
            ],
            ("quality", "Good"),
            weight=0.90,
            category_scope=["Portrait", "Group Photograph"],
            explanation="Sharp focus, a clean signal, visible facial detail, and strong framing support good portrait quality alongside the learned MOS estimate.",
        ),
        FuzzyRule(
            "H20", "Balanced, high-contrast sharp portrait",
            [
                ("blur", "Low"),
                ("noise", "Low"),
                ("face_visibility", "Clear"),
                ("composition", "Good"),
                ("brightness", "Normal"),
                ("contrast", "High"),
            ],
            ("quality", "Excellent"),
            weight=0.90,
            category_scope=["Portrait", "Group Photograph"],
            explanation="Strong portrait clarity and framing, balanced lighting, and rich tonal contrast jointly support excellent technical quality.",
        ),
    ]
    return rules
