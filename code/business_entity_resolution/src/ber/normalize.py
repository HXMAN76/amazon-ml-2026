"""Text normalisation for business names and addresses.

All rules are language-agnostic string rules (no external lookups): accent stripping,
lowercasing, punctuation removal, abbreviation expansion and legal-suffix removal.
"""

from __future__ import annotations

import re
import unicodedata

NAME_ABBR = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
    "co": "company", "llc": "llc", "intl": "international", "int": "international",
    "svcs": "services", "svc": "services", "mfg": "manufacturing", "assoc": "associates",
    "bros": "brothers", "dept": "department", "univ": "university", "&": "and",
    "ent": "enterprises", "ind": "industries", "tech": "technologies", "sys": "systems",
}
# Legal-form words dropped when building the "core" name (multi-country: US/IN/FR forms).
LEGAL = {
    "corporation", "incorporated", "limited", "private", "company", "llc", "llp", "lp", "plc",
    "pty", "the", "and", "of", "sa", "sarl", "sas", "sasu", "eurl", "gmbh", "ag", "bv", "nv",
    "opc", "ltda", "cie", "et", "fils", "groupe", "societe",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "hwy": "highway", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "ste": "suite", "fl": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "apt": "apartment", "no": "number", "sec": "sector", "mkt": "market", "cir": "circle",
    "pkwy": "parkway", "bd": "boulevard", "av.": "avenue", "chem": "chemin", "pde": "parade",
}
COUNTRY_ALIASES = {
    "usa": "us", "u.s.": "us", "u.s.a.": "us", "united states": "us", "united states of america": "us",
    "america": "us", "in": "india", "ind": "india", "bharat": "india", "republic of india": "india",
    "fr": "france", "fra": "france", "république française": "france", "republique francaise": "france",
}

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


def _is_latin(c: str) -> bool:
    o = ord(c)
    return o < 0x250 or 0x1E00 <= o <= 0x1EFF


def strip_accents(s: str) -> str:
    """Drop accents on Latin letters only. NFKD + dropping every combining mark would also delete
    Devanagari vowel signs (matras), destroying Hindi names, so non-Latin characters are kept intact."""
    return "".join(
        "".join(d for d in unicodedata.normalize("NFKD", c) if not unicodedata.combining(d)) if _is_latin(c) else c
        for c in s
    )


def _base(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = strip_accents(s).lower().replace("&", " and ")
    # replace punctuation/symbols only: `[^\w\s]` would also split words at Devanagari combining marks
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return _WS.sub(" ", s).strip()


def norm_name(s: object) -> str:
    toks = [NAME_ABBR.get(t, t) for t in _base(s).split()]
    return " ".join(toks)


def core_name(s: object) -> str:
    """Name without legal-form words; falls back to the full name if everything is stripped."""
    toks = norm_name(s).split()
    core = [t for t in toks if t not in LEGAL]
    return " ".join(core or toks)


def norm_addr(s: object) -> str:
    toks = [ADDR_ABBR.get(t, t) for t in _base(s).split()]
    return " ".join(toks)


def norm_country(s: object) -> str:
    if not isinstance(s, str):
        return ""
    c = strip_accents(s).lower().strip()
    return COUNTRY_ALIASES.get(c, c)


def numbers(s: str) -> set[str]:
    return set(_NUM.findall(s))


def acronym(s: str) -> str:
    return "".join(t[0] for t in s.split() if t)
