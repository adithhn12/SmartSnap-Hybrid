"""
Fuzzy Membership Functions and Variable Definitions for SmartSnap.
Uses scikit-fuzzy functions with mathematical guarantees and explicit point-evaluation.
"""

from typing import Dict, Any, List
import numpy as np
import skfuzzy as fuzz


class FuzzyVariable:
    """
    Represents a linguistic variable (e.g., Blur, Quality) with associated fuzzy sets
    and membership functions over a continuous universe of discourse [0, 100].
    """

    def __init__(self, name: str, unit: str = "%", vmin: float = 0.0, vmax: float = 100.0, step: float = 1.0):
        self.name = name
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        self.step = step
        self.universe = np.arange(vmin, vmax + step, step)
        self.terms: Dict[str, Dict[str, Any]] = {}

    def add_trapezoidal(self, term_name: str, a: float, b: float, c: float, d: float, description: str = ""):
        """Add a trapezoidal fuzzy term."""
        mf = fuzz.trapmf(self.universe, [a, b, c, d])
        self.terms[term_name] = {
            "type": "trapezoidal",
            "params": [float(a), float(b), float(c), float(d)],
            "mf": mf,
            "description": description,
        }

    def add_triangular(self, term_name: str, a: float, b: float, c: float, description: str = ""):
        """Add a triangular fuzzy term."""
        mf = fuzz.trimf(self.universe, [a, b, c])
        self.terms[term_name] = {
            "type": "triangular",
            "params": [float(a), float(b), float(c)],
            "mf": mf,
            "description": description,
        }

    def evaluate_membership(self, value: float) -> Dict[str, float]:
        """
        Evaluate membership degree mu(value) for all linguistic terms of this variable.
        Clips input to [vmin, vmax].
        """
        val = float(np.clip(value, self.vmin, self.vmax))
        memberships = {}
        for term_name, term_info in self.terms.items():
            deg = fuzz.interp_membership(self.universe, term_info["mf"], val)
            memberships[term_name] = float(np.round(deg, 4))
        return memberships

    def get_curve_points(self) -> List[Dict[str, Any]]:
        """
        Export sampled curve points for frontend visualization.
        Returns a list of dicts: [{'x': 0, 'Low': 1.0, 'Medium': 0.0, ...}, ...]
        """
        points = []
        for x_val in self.universe[::2]:  # step of 2 for clean, lightweight payload
            pt: Dict[str, Any] = {"x": float(x_val)}
            for term_name, term_info in self.terms.items():
                deg = fuzz.interp_membership(self.universe, term_info["mf"], x_val)
                pt[term_name] = float(np.round(deg, 4))
            points.append(pt)
        return points


def create_fuzzy_system_variables() -> Dict[str, FuzzyVariable]:
    """
    Creates and returns the standardized fuzzy input and output variables for SmartSnap.
    """
    # 1. Blur (0 = razor sharp, 100 = severely blurred)
    # Note: 0-30 = Low Blur (sharp), 20-70 = Medium Blur, 60-100 = High Blur
    blur = FuzzyVariable("blur", unit="pts")
    blur.add_trapezoidal("Low", 0, 0, 20, 35, description="Sharp photograph with minimal blur")
    blur.add_triangular("Medium", 22, 48, 72, description="Noticeable softening or mild motion/focus blur")
    blur.add_trapezoidal("High", 60, 78, 100, 100, description="Severe motion or defocus blur")

    # 2. Brightness (0 = dark, 50 = normal/ideal, 100 = blown highlights/overexposed)
    brightness = FuzzyVariable("brightness", unit="pts")
    brightness.add_trapezoidal("Dark", 0, 0, 28, 44, description="Underexposed or shadows dominate")
    brightness.add_triangular("Normal", 32, 52, 72, description="Well-balanced midtone illumination")
    brightness.add_trapezoidal("Overexposed", 62, 78, 100, 100, description="Blown highlights or harsh overexposure")

    # 3. Contrast (0 = washed out / flat, 100 = rich dynamic range)
    contrast = FuzzyVariable("contrast", unit="pts")
    contrast.add_trapezoidal("Low", 0, 0, 25, 42, description="Flat histogram, low dynamic range")
    contrast.add_triangular("Medium", 30, 52, 74, description="Moderate dynamic range")
    contrast.add_trapezoidal("High", 62, 80, 100, 100, description="Vivid contrast and full tonal distribution")

    # 4. Noise (0 = clean, 100 = heavy sensor noise/grain)
    noise = FuzzyVariable("noise", unit="pts")
    noise.add_trapezoidal("Low", 0, 0, 22, 38, description="Clean sensor signal with negligible noise")
    noise.add_triangular("Medium", 26, 50, 74, description="Moderate grain or sensor noise")
    noise.add_trapezoidal("High", 62, 80, 100, 100, description="Heavy artifacting or chrominance/luminance noise")

    # 5. Face Visibility (0 = obscured/unrecognized, 100 = crisp, prominent facial feature)
    face_visibility = FuzzyVariable("face_visibility", unit="pts")
    face_visibility.add_trapezoidal("Poor", 0, 0, 25, 45, description="Faces missing, tiny, or obscured")
    face_visibility.add_triangular("Acceptable", 32, 58, 80, description="Partially visible or small faces")
    face_visibility.add_trapezoidal("Clear", 68, 84, 100, 100, description="Prominently centered, well-lit faces")

    # 6. Item subject visibility uses the same linguistic bands without face evidence.
    object_visibility = FuzzyVariable("object_visibility", unit="pts")
    object_visibility.add_trapezoidal("Poor", 0, 0, 25, 45, description="Primary object region is unclear or clipped")
    object_visibility.add_triangular("Acceptable", 32, 58, 80, description="Primary object region is partly visible or weakly framed")
    object_visibility.add_trapezoidal("Clear", 68, 84, 100, 100, description="Primary object region is distinct and comfortably framed")

    # 7. Composition (0 = off-center cutoff, 100 = classic rule-of-thirds balance)
    composition = FuzzyVariable("composition", unit="pts")
    composition.add_trapezoidal("Poor", 0, 0, 25, 45, description="Awkward edge proximity or bad balance")
    composition.add_triangular("Average", 32, 54, 76, description="Acceptable framing with room for improvement")
    composition.add_trapezoidal("Good", 64, 82, 100, 100, description="Strong rule-of-thirds and subject framing")

    # 8. Random Forest prediction in KonIQ's native MOS scale. The points follow
    # the observed MOS distribution (p05≈2.08, median≈3.28, p95≈3.85); these
    # linguistic terms are not KEEP/RECOVER/RETAKE labels.
    predicted_mos = FuzzyVariable("predicted_mos", unit="MOS", vmin=1.0, vmax=5.0, step=0.01)
    predicted_mos.add_trapezoidal(
        "Low", 1.0, 1.0, 2.1, 2.7,
        description="Low mean human opinion score on the native KonIQ 1-5 scale",
    )
    predicted_mos.add_triangular(
        "Medium", 2.4, 3.28, 3.75,
        description="Mid-range mean human opinion score on the native KonIQ 1-5 scale",
    )
    predicted_mos.add_trapezoidal(
        "High", 3.55, 3.9, 5.0, 5.0,
        description="High mean human opinion score on the native KonIQ 1-5 scale",
    )

    # 9. Output: Quality (0 - 100)
    quality = FuzzyVariable("quality", unit="pts")
    quality.add_trapezoidal("Very Poor", 0, 0, 15, 30, description="Unusable photo, severe defects")
    quality.add_triangular("Poor", 18, 34, 48, description="Noticeable defects, difficult to salvage")
    quality.add_triangular("Moderate", 38, 54, 68, description="Acceptable flaws, can be recovered")
    quality.add_triangular("Good", 60, 74, 88, description="Solid quality, minor adjustments only")
    quality.add_trapezoidal("Excellent", 80, 92, 100, 100, description="Superb clarity, lighting, and composition")

    return {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "noise": noise,
        "face_visibility": face_visibility,
        "object_visibility": object_visibility,
        "composition": composition,
        "predicted_mos": predicted_mos,
        "quality": quality,
    }
