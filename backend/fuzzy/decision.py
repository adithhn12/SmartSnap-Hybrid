"""
Decision Engine for SmartSnap.
Maps defuzzified fuzzy scores to KEEP / RECOVER / RETAKE recommendations,
applies critical quality safeguards, and generates mathematical explanations.
"""

from typing import Dict, Any, List


class DecisionEngine:
    """
    Evaluates defuzzified fuzzy quality scores against decision thresholds,
    incorporates critical safety overrides, and provides detailed explanatory diagnostics.
    """

    def __init__(self, keep_threshold: float = 70.0, recover_threshold: float = 45.0):
        self.keep_threshold = keep_threshold
        self.recover_threshold = recover_threshold

    def evaluate(self, fuzzy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines recommendation and diagnostic details.
        """
        score = fuzzy_result["defuzzified_score"]
        inputs = fuzzy_result["input_values"]
        memberships = fuzzy_result["memberships"]
        category = fuzzy_result.get("category", "General Photography")
        category_key = category.casefold()
        activated_rules = fuzzy_result.get("activated_rules", [])

        # 1. Base fuzzy recommendation
        if score >= self.keep_threshold:
            base_recommendation = "KEEP"
            status_color = "emerald"
        elif score >= self.recover_threshold:
            base_recommendation = "RECOVER"
            status_color = "amber"
        else:
            base_recommendation = "RETAKE"
            status_color = "rose"

        final_recommendation = base_recommendation
        critical_flags: List[str] = []
        problems: List[str] = []
        corrections: List[str] = []

        # 2. Retain the original legacy safeguards. The hybrid pathway represents
        # severe blur and face intent in its Mamdani rules instead of bypassing
        # centroid defuzzification with crisp action overrides.
        blur_val = inputs.get("blur", 0.0)
        brightness_val = inputs.get("brightness", 50.0)
        face_val = inputs.get("face_visibility", 50.0)
        predicted_mos = inputs.get("predicted_mos")

        if fuzzy_result.get("pipeline", "legacy") != "hybrid":
            # Critical Blur: Blur > 80 is physically unrecoverable optical blur
            if blur_val >= 80.0:
                critical_flags.append(f"Severe optical blur (Index: {blur_val:.1f}/100) destroys fine high-frequency details.")
                if final_recommendation == "KEEP":
                    final_recommendation = "RECOVER"
                if blur_val >= 90.0 and final_recommendation != "RETAKE":
                    final_recommendation = "RETAKE"
                    critical_flags.append("Critical blur threshold exceeded; unrecoverable by digital sharpening.")

            # Critical Overexposure: Blown whites have 0 information
            if brightness_val >= 88.0:
                critical_flags.append("Harsh overexposure has completely clipped highlight channels.")
                if final_recommendation == "KEEP":
                    final_recommendation = "RECOVER"

            # Critical Portrait Face requirement
            if category_key in {"portrait", "group photograph"} and face_val < 30.0:
                critical_flags.append(f"Selected category '{category}' requires clear facial visibility, but face detection score is only {face_val:.1f}/100.")
                if final_recommendation == "KEEP":
                    final_recommendation = "RECOVER"
                if face_val < 15.0:
                    final_recommendation = "RETAKE"

        # 3. Diagnostic Problem Detection based on fuzzy memberships
        blur_mems = memberships.get("blur", {})
        if blur_mems.get("High", 0) > 0.25:
            problems.append(f"Substantial blur detected (High blur membership: {blur_mems.get('High', 0):.2f}). Subject detail is degraded.")
            corrections.append("Increase shutter speed (1/250s+), engage optical stabilization, or steady the camera on a tripod.")
        elif blur_mems.get("Medium", 0) > 0.35:
            problems.append(f"Mild softening or motion smear detected (Medium blur membership: {blur_mems.get('Medium', 0):.2f}).")
            corrections.append("Apply unsharp masking or local edge sharpening in post-production.")

        bright_mems = memberships.get("brightness", {})
        if bright_mems.get("Dark", 0) > 0.35:
            problems.append(f"Underexposed frame (Dark membership: {bright_mems.get('Dark', 0):.2f}).")
            corrections.append("Increase exposure compensation (+0.7 to +1.5 EV), widen aperture, or increase ambient lighting.")
        elif bright_mems.get("Overexposed", 0) > 0.35:
            problems.append(f"Overexposed highlights (Overexposed membership: {bright_mems.get('Overexposed', 0):.2f}).")
            corrections.append("Lower exposure compensation, enable HDR mode, or reduce direct glare on subject.")

        contrast_mems = memberships.get("contrast", {})
        if contrast_mems.get("Low", 0) > 0.35:
            problems.append(f"Low tonal contrast (Low contrast membership: {contrast_mems.get('Low', 0):.2f}).")
            corrections.append("Perform S-curve contrast stretching or adjust black/white clipping points.")

        noise_mems = memberships.get("noise", {})
        if noise_mems.get("High", 0) > 0.30:
            problems.append(f"High sensor noise / grain (High noise membership: {noise_mems.get('High', 0):.2f}).")
            corrections.append("Lower ISO setting, use longer exposure on tripod, or apply bilateral / non-local means denoising.")

        face_mems = memberships.get("face_visibility", {})
        if category_key in {"portrait", "group photograph"} and face_mems.get("Poor", 0) > 0.35:
            problems.append(f"Face features not clearly distinguished (Poor face membership: {face_mems.get('Poor', 0):.2f}).")
            corrections.append("Ask subject to face the primary light source and ensure focus point locks on eye/face.")

        comp_mems = memberships.get("composition", {})
        if comp_mems.get("Poor", 0) > 0.35:
            problems.append(f"Sub-optimal framing / rule-of-thirds alignment (Poor composition membership: {comp_mems.get('Poor', 0):.2f}).")
            corrections.append("Reframe subject along rule-of-thirds grid intersections and maintain breathing room around edges.")

        if predicted_mos is not None and predicted_mos <= 2.6:
            problems.append(
                f"The Random Forest predicts a low mean-opinion score ({predicted_mos:.2f}/5); "
                "this is a perceptual estimate, not a retake label."
            )
            corrections.append(
                "Review the image for subject clarity, exposure, tonal balance, and framing; "
                "reshoot if the perceptual defects cannot be corrected."
            )

        if not problems:
            if score >= self.keep_threshold:
                problems.append("High overall photographic fidelity; measurements satisfy keeper standards across evaluated characteristics.")
                corrections.append("Image is ready for cataloging, publication, or archiving with no further adjustment required.")
            else:
                problems.append(
                    f"No individual diagnostic crossed its alert threshold, but the combined score "
                    f"({score:.1f}) is below the KEEP threshold ({self.keep_threshold:.1f})."
                )
                corrections.append(
                    "Review the image's overall look and intended use; this heuristic assessment "
                    "may not capture subject appeal or other aesthetic context."
                )

        # 4. Generate Mathematical Decision Explanation
        # Distinguish outright penalties from moderate/cautionary evidence.
        pos_rules = [r for r in activated_rules if r["consequent"][1] in ["Excellent", "Good"]]
        penalty_rules = [r for r in activated_rules if r["consequent"][1] in ["Very Poor", "Poor"]]
        caution_rules = [r for r in activated_rules if r["consequent"][1] == "Moderate"]

        rule_summaries = []
        if penalty_rules:
            top_penalty = penalty_rules[0]
            rule_summaries.append(f"Primary penalty rule: [{top_penalty['rule_id']}: {top_penalty['name']} (alpha={top_penalty['firing_strength']:.2f}) -> {top_penalty['consequent'][1]}]")
        if caution_rules:
            top_caution = caution_rules[0]
            rule_summaries.append(f"Primary caution rule: [{top_caution['rule_id']}: {top_caution['name']} (alpha={top_caution['firing_strength']:.2f}) -> {top_caution['consequent'][1]}]")
        if pos_rules:
            top_pos = pos_rules[0]
            rule_summaries.append(f"Primary supporting rule: [{top_pos['rule_id']}: {top_pos['name']} (alpha={top_pos['firing_strength']:.2f}) -> {top_pos['consequent'][1]}]")

        explanation_parts = [
            f"Mamdani fuzzy inference computed an aggregated centroid quality score of {score:.1f}/100.",
            f"Category context: {category}.",
            "Influential fuzzy rule activations: " + (" | ".join(rule_summaries) if rule_summaries else "Baseline rules fired evenly."),
        ]
        if predicted_mos is not None:
            hybrid_evidence = ["focus", "noise", "framing"]
            if "brightness" in inputs:
                hybrid_evidence.append("lighting")
            if "contrast" in inputs:
                hybrid_evidence.append("contrast")
            if "face_visibility" in inputs:
                hybrid_evidence.append("category-relevant face visibility")
            explanation_parts.insert(
                0,
                f"Hybrid input: Random Forest predicted MOS {predicted_mos:.2f}/5; "
                f"the FIS combines it with measured {', '.join(hybrid_evidence)} evidence.",
            )
            if memberships.get("predicted_mos", {}).get("Low", 0.0) > 0.0 and pos_rules:
                supporting_evidence = sorted({
                    variable
                    for rule in pos_rules
                    for variable, _ in rule.get("antecedents", [])
                    if variable != "predicted_mos"
                })
                explanation_parts.append(
                    "Evidence is mixed: the low-MOS membership remains active, while "
                    "supporting CV rules also fire from "
                    f"{', '.join(supporting_evidence)} evidence. The final score aggregates both signals."
                )

        if critical_flags:
            explanation_parts.append("Special condition triggered: " + " ".join(critical_flags))

        if final_recommendation == "KEEP":
            explanation_parts.append("Conclusion: The photograph satisfies high fidelity criteria and should be KEPT.")
        elif final_recommendation == "RECOVER":
            if problems:
                explanation_parts.append(
                    "Conclusion: The combined score falls in the RECOVER band; review the measured findings before applying edits."
                )
            else:
                explanation_parts.append(
                    "Conclusion: The combined score falls in the RECOVER band, though no individual diagnostic crossed its alert threshold; consider intended use and aesthetic context."
                )
        else:
            explanation_parts.append(
                "Conclusion: The combined score falls below the RECOVER band; RETAKE is recommended unless the image remains suitable for its intended use."
            )

        full_explanation = "\n".join(explanation_parts)

        return {
            "recommendation": final_recommendation,
            "base_recommendation": base_recommendation,
            "quality_score": score,
            "status_color": status_color,
            "thresholds": {
                "keep": self.keep_threshold,
                "recover": self.recover_threshold,
            },
            "critical_flags": critical_flags,
            "problems": problems,
            "corrections": corrections,
            "explanation": full_explanation,
        }
