"""
Mamdani Fuzzy Inference System for SmartSnap.
Performs:
- Input Fuzzification
- Antecedent Conjunction (Min)
- Consequent Implication (Min clipping)
- Fuzzy Output Aggregation (Max)
- Centroid Defuzzification
"""

from typing import Dict, Any, List
import numpy as np
import skfuzzy as fuzz

from .membership import create_fuzzy_system_variables, FuzzyVariable
from .rules import build_default_rule_base, build_hybrid_rule_base, FuzzyRule


class MamdaniInferenceEngine:
    """
    Standard Mamdani Fuzzy Inference System with full intermediate state tracking
    for educational inspection and frontend visualization.
    """

    def __init__(self):
        self.variables = create_fuzzy_system_variables()
        self.rules = build_default_rule_base()
        self.hybrid_rules = build_hybrid_rule_base()
        self.quality_var: FuzzyVariable = self.variables["quality"]

    def compute(
        self,
        inputs: Dict[str, float],
        category: str = "General Photography",
        only_provided_inputs: bool = False,
        pipeline: str = "legacy",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Mamdani inference.

        Parameters:
            inputs: e.g. {'blur': 25.0, 'brightness': 55.0, 'contrast': 70.0,
                          'noise': 15.0, 'face_visibility': 85.0, 'composition': 80.0}
            category: 'Portrait', 'Landscape', 'Document', 'Group Photograph', 'General Photography'

        Returns:
            Dictionary containing input fuzzification, activated rules, aggregated curve,
            and defuzzified crisp score.
        """
        if pipeline not in {"legacy", "hybrid"}:
            raise ValueError(f"Unknown fuzzy inference pipeline: {pipeline}")
        human_focused_category = category.casefold() in {
            "portrait",
            "group photograph",
        }
        effective_inputs = dict(inputs)
        if not human_focused_category:
            effective_inputs.pop("face_visibility", None)
        if pipeline == "hybrid":
            missing_hybrid_inputs = {
                "predicted_mos",
                "blur",
                "composition",
            }.difference(effective_inputs)
            if human_focused_category and "face_visibility" not in effective_inputs:
                missing_hybrid_inputs.add("face_visibility")
            if missing_hybrid_inputs:
                raise ValueError(
                    "Hybrid Mamdani inference requires inputs: "
                    f"{sorted(missing_hybrid_inputs)}"
                )

        # 1. Fuzzification: evaluate membership for every input
        memberships: Dict[str, Dict[str, float]] = {}
        for var_name, var_obj in self.variables.items():
            if var_name == "quality":
                continue
            if var_name == "face_visibility" and not human_focused_category:
                continue
            if var_name == "predicted_mos" and var_name not in effective_inputs:
                continue
            if only_provided_inputs and var_name not in effective_inputs:
                continue
            default_value = 3.0 if var_name == "predicted_mos" else 50.0
            val = float(effective_inputs.get(var_name, default_value))
            memberships[var_name] = var_obj.evaluate_membership(val)

        # 2. Rule Evaluation and Implication
        # Consequent variable universe
        y_universe = self.quality_var.universe
        aggregated_mf = np.zeros_like(y_universe, dtype=float)

        activated_rules: List[Dict[str, Any]] = []
        all_rule_activations: List[Dict[str, Any]] = []

        selected_rules = self.hybrid_rules if pipeline == "hybrid" else self.rules
        for rule in selected_rules:
            alpha = rule.evaluate_firing_strength(memberships, category)
            consequent_term = rule.consequent[1]
            term_mf = self.quality_var.terms[consequent_term]["mf"]

            rule_record = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "antecedents": rule.antecedents,
                "consequent": rule.consequent,
                "firing_strength": float(alpha),
                "explanation": rule.explanation,
                "applicable": rule.is_applicable_to_category(category),
            }
            all_rule_activations.append(rule_record)

            if alpha > 0.0:
                activated_rules.append(rule_record)
                # Mamdani Implication: min(alpha, mf_consequent)
                clipped_mf = np.fmin(alpha, term_mf)
                # Mamdani Aggregation: max(aggregated, clipped_mf)
                aggregated_mf = np.fmax(aggregated_mf, clipped_mf)

        # Sort activated rules descending by firing strength
        activated_rules.sort(key=lambda r: r["firing_strength"], reverse=True)

        # 3. Defuzzification via Centroid
        # If no rules fired (e.g. edge condition with weight 0), handle gracefully
        if np.sum(aggregated_mf) == 0:
            defuzzified_score = 50.0
            defuzz_method = "fallback_midpoint"
        else:
            try:
                defuzzified_score = float(fuzz.defuzz(y_universe, aggregated_mf, "centroid"))
                defuzz_method = "centroid"
            except Exception:
                # Numerical integration fallback
                numerator = np.sum(y_universe * aggregated_mf)
                denominator = np.sum(aggregated_mf)
                defuzzified_score = float(numerator / denominator) if denominator != 0 else 50.0
                defuzz_method = "numerical_centroid"

        # Round crisp score to 2 decimal places
        defuzzified_score = float(np.round(np.clip(defuzzified_score, 0.0, 100.0), 2))

        # 4. Prepare aggregated curve for frontend visualization (sampled for efficiency)
        aggregated_curve = []
        for x_val, y_val in zip(y_universe[::2], aggregated_mf[::2]):
            aggregated_curve.append({
                "quality": float(x_val),
                "membership": float(np.round(y_val, 4))
            })

        return {
            "input_values": {k: float(v) for k, v in effective_inputs.items()},
            "category": category,
            "pipeline": pipeline,
            "memberships": memberships,
            "activated_rules": activated_rules,
            "all_rules_count": len(selected_rules),
            "activated_count": len(activated_rules),
            "aggregated_curve": aggregated_curve,
            "defuzzified_score": defuzzified_score,
            "defuzz_method": defuzz_method,
        }

    def get_system_metadata(self) -> Dict[str, Any]:
        """
        Returns full static schema and curves of variables and rules for the frontend.
        """
        vars_meta = {}
        for var_name, var_obj in self.variables.items():
            terms_meta = {}
            for term_name, term_info in var_obj.terms.items():
                terms_meta[term_name] = {
                    "type": term_info["type"],
                    "params": term_info["params"],
                    "description": term_info["description"],
                }
            vars_meta[var_name] = {
                "name": var_obj.name,
                "unit": var_obj.unit,
                "vmin": var_obj.vmin,
                "vmax": var_obj.vmax,
                "terms": terms_meta,
                "curve_points": var_obj.get_curve_points(),
            }

        rules_meta = [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "antecedents": r.antecedents,
                "consequent": r.consequent,
                "category_scope": r.category_scope,
                "weight": r.weight,
                "explanation": r.explanation,
                "pipeline": "legacy",
            }
            for r in self.rules
        ]
        rules_meta.extend(
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "antecedents": r.antecedents,
                "consequent": r.consequent,
                "category_scope": r.category_scope,
                "weight": r.weight,
                "explanation": r.explanation,
                "pipeline": "hybrid",
            }
            for r in self.hybrid_rules
        )

        return {
            "variables": vars_meta,
            "rules": rules_meta,
            "pipelines": {
                "legacy": "Original CV-only Mamdani rulebase used by the fuzzy playground",
                "hybrid": "RF MOS + noise/blur + category-aware face and composition Mamdani rulebase",
                "face_categories": ["Portrait", "Group Photograph"],
            },
        }
