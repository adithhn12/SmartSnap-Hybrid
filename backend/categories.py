"""Canonical photography categories supported by the analysis API."""

from typing import Final


PORTRAIT_PHOTOGRAPHY: Final = "Portrait Photography"
ITEM_PHOTOGRAPHY: Final = "Item Photography"
LANDSCAPE_PHOTOGRAPHY: Final = "Landscape Photography"
GENERAL_PHOTOGRAPHY: Final = "General Photography"

PHOTOGRAPHY_CATEGORIES: Final = (
    PORTRAIT_PHOTOGRAPHY,
    ITEM_PHOTOGRAPHY,
    LANDSCAPE_PHOTOGRAPHY,
    GENERAL_PHOTOGRAPHY,
)

_CATEGORY_BY_KEY = {
    category.casefold(): category for category in PHOTOGRAPHY_CATEGORIES
}


def normalize_category(category: str) -> str:
    """Return a canonical category name or reject unsupported categories."""
    if isinstance(category, str):
        canonical = _CATEGORY_BY_KEY.get(category.strip().casefold())
        if canonical is not None:
            return canonical
    choices = ", ".join(PHOTOGRAPHY_CATEGORIES)
    raise ValueError(f"Unsupported photography category. Choose one of: {choices}.")


def category_key(category: str) -> str:
    """Return the stable internal key for a supported category."""
    return normalize_category(category).casefold().removesuffix(" photography")
