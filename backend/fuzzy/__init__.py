"""
SmartSnap Fuzzy Logic Package.
"""

from .membership import FuzzyVariable, create_fuzzy_system_variables
from .rules import FuzzyRule, build_default_rule_base
from .inference import MamdaniInferenceEngine
from .decision import DecisionEngine

__all__ = [
    "FuzzyVariable",
    "create_fuzzy_system_variables",
    "FuzzyRule",
    "build_default_rule_base",
    "MamdaniInferenceEngine",
    "DecisionEngine",
]

