"""Cheap text features + value/unit parsing used for post-processing.

Past catalogs packed structured bits into free text, e.g. 2025:
    "Item Name: ... Bullet Point 1: ... Value: 12.0 Unit: Ounce"
and 2024 wanted predictions like "34.5 gram". These helpers cover both patterns.
"""

from __future__ import annotations

import re

import numpy as np

NUM = r"[-+]?\d+(?:[.,]\d+)?"

# alias -> canonical unit (extend on day 1 from the allowed-unit list in the problem statement)
UNIT_ALIASES = {
    "g": "gram", "gm": "gram", "gms": "gram", "gram": "gram", "grams": "gram", "gr": "gram",
    "kg": "kilogram", "kgs": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram",
    "mg": "milligram", "milligram": "milligram", "mcg": "microgram", "µg": "microgram",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce", "fl oz": "fluid ounce", "fl. oz": "fluid ounce",
    "fluid ounce": "fluid ounce", "fluid ounces": "fluid ounce",
    "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound", "ton": "ton", "tons": "ton",
    "ml": "millilitre", "millilitre": "millilitre", "milliliter": "millilitre",
    "l": "litre", "ltr": "litre", "litre": "litre", "liter": "litre", "liters": "litre", "litres": "litre",
    "cl": "centilitre", "dl": "decilitre", "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "pint": "pint", "quart": "quart", "cup": "cup", "cups": "cup", "cubic foot": "cubic foot",
    "cubic feet": "cubic foot", "cubic inch": "cubic inch",
    "mm": "millimetre", "millimeter": "millimetre", "millimetre": "millimetre",
    "cm": "centimetre", "centimeter": "centimetre", "centimetre": "centimetre",
    "m": "metre", "meter": "metre", "metre": "metre", "meters": "metre",
    "in": "inch", "inch": "inch", "inches": "inch", "ft": "foot", "foot": "foot", "feet": "foot",
    "yard": "yard", "yards": "yard",
    "w": "watt", "watt": "watt", "watts": "watt", "kw": "kilowatt", "kilowatt": "kilowatt",
    "v": "volt", "volt": "volt", "volts": "volt", "kv": "kilovolt", "mv": "millivolt",
    "count": "count", "ct": "count", "pack": "count", "pcs": "count", "pieces": "count",
}
_UNIT_RX = "|".join(sorted((re.escape(u) for u in UNIT_ALIASES), key=len, reverse=True))
VALUE_UNIT_RX = re.compile(rf"({NUM})\s*({_UNIT_RX})\b", re.IGNORECASE)


def normalize_unit(u: str) -> str | None:
    return UNIT_ALIASES.get(u.strip().lower().rstrip("."))


def parse_value_unit(text: str | None, allowed_units: set[str] | None = None) -> tuple[float, str] | None:
    """First '<number> <unit>' in text, unit canonicalised; None if nothing usable."""
    if not text:
        return None
    for m in VALUE_UNIT_RX.finditer(text):
        unit = normalize_unit(m.group(2))
        if unit and (allowed_units is None or unit in allowed_units):
            return float(m.group(1).replace(",", ".")), unit
    return None


def format_value_unit(value: float, unit: str) -> str:
    """'34.5 gram' style output; drops trailing .0 noise."""
    return f"{value:g} {unit}"


def extract_field(text: str | None, field: str) -> str | None:
    """Pull 'Field: value' out of packed catalog text, e.g. extract_field(t, 'Value')."""
    if not text:
        return None
    m = re.search(rf"{re.escape(field)}\s*:\s*([^\n]+?)(?=\s+[A-Z][\w ]{{0,30}}:|\n|$)", text)
    return m.group(1).strip() if m else None


def numeric_stats(texts: list[str | None]) -> np.ndarray:
    """Per-text [n_numbers, max, min, first, len_chars, n_words] as float32 features."""
    out = np.zeros((len(texts), 6), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t or ""
        nums = [float(x.replace(",", ".")) for x in re.findall(NUM, t)[:50]]
        if nums:
            out[i, :4] = len(nums), max(nums), min(nums), nums[0]
        out[i, 4:] = len(t), len(t.split())
    return out


def tfidf_svd(train: list[str], test: list[str], n_components: int = 256, max_features: int = 200_000, seed: int = 0):
    """Word+char TF-IDF -> TruncatedSVD. Fit on train+test text (unsupervised, allowed in most rules)."""
    from scipy.sparse import hstack
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    allt = [t or "" for t in train] + [t or "" for t in test]
    w = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    M = hstack([w.fit_transform(allt), c.fit_transform(allt)]).tocsr()
    Z = TruncatedSVD(n_components, random_state=seed).fit_transform(M).astype(np.float32)
    return Z[: len(train)], Z[len(train):]
