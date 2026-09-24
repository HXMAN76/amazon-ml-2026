"""Deterministic text normalisation for business names and addresses (no external lookups).

Handles the noise seen in the training data: HTML entities (`&amp;`), leetspeak inside words
(`C0mpany`), DBA / alias text (`X dba Y`), domain-style names (`ipower.com`, `X | www.x.com`),
legal forms, abbreviations, glued city suffixes (`CHICAGOCDP`), and non-Latin scripts. Accents are
stripped from Latin letters only; Devanagari, Telugu, Malayalam etc. are left intact.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from dataclasses import dataclass

from anyascii import anyascii

NAME_ABBR = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
    "co": "company", "intl": "international", "int": "international", "svcs": "services",
    "svc": "services", "mfg": "manufacturing", "assoc": "associates", "bros": "brothers",
    "dept": "department", "univ": "university", "ent": "enterprises", "ind": "industries",
    "tech": "technologies", "sys": "systems", "grp": "group", "mgmt": "management",
}
# Legal-form words: moved out of the "core" name into a separate `legal` field.
LEGAL = {
    "corporation", "incorporated", "limited", "private", "company", "llc", "llp", "lp", "plc",
    "pllc", "pty", "opc", "ltda", "gmbh", "ag", "bv", "nv", "sa", "sarl", "sas", "sasu", "eurl",
    "sci", "snc", "scop", "ei", "eirl", "cie", "partners", "partnership",
}
STOP = {"the", "and", "of", "et", "de", "la", "le", "les", "du", "des"}
ADDR_ABBR = {
    "rd": "road", "street": "st", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "hwy": "highway", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "ste": "suite", "fl": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "apt": "apartment", "sec": "sector", "mkt": "market", "cir": "circle", "pkwy": "parkway",
    "bd": "boulevard", "chem": "chemin", "saint": "st", "sainte": "st",
}
ADDR_FILLER = {"door", "no", "number", "hno", "cdp"}
COUNTRY_ALIASES = {
    "usa": "us", "united states": "us", "united states of america": "us", "america": "us",
    "in": "india", "ind": "india", "bharat": "india", "fr": "france", "fra": "france",
}
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}

_DBA = re.compile(
    r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|dba|trading as|also known as|a\.k\.a\.?|aka|formerly known as|fka)\b",
    re.I,
)
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([^\s/@]+\.[a-z]{2,})(?:/\S*)?$", re.I)
_TLD = re.compile(r"(?:\.(?:co|com|org|net|gov|edu|ac))?\.[a-z]{2,}$", re.I)
_APOS = re.compile(r"['’‘`]")
_WS = re.compile(r"\s+")


def _is_latin(c: str) -> bool:
    o = ord(c)
    return o < 0x250 or 0x1E00 <= o <= 0x1EFF


def strip_accents(s: str) -> str:
    """Drop accents on Latin letters only (NFKD + dropping every mark would wreck Indic scripts)."""
    return "".join(
        "".join(d for d in unicodedata.normalize("NFKD", c) if not unicodedata.combining(d)) if _is_latin(c) else c
        for c in s
    )


_ASCII_TABLE = {ord(c): (None if c in "'`" else " ") for c in string.punctuation}


def _tokens_slow(s: str) -> list[str]:
    s = strip_accents(s).lower()
    s = _APOS.sub("", s).replace("&", " and ")
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return s.split()


def _tokens(s: str) -> list[str]:
    if s.isascii():  # fast path: about 20x quicker than the per-character Unicode path
        return s.lower().replace("&", " and ").translate(_ASCII_TABLE).split()
    return _tokens_slow(s)


def _leet_fix(tok: str) -> str:
    """`c0mpany` -> `company`, `5ecure` -> `secure`. Only tokens of 4+ chars with exactly one leet digit
    among at least three letters, so real alphanumerics (`3m`, `a1`, `24x7`) are left alone."""
    if len(tok) < 4:
        return tok
    digits = [c for c in tok if c.isdigit()]
    letters = sum(c.isalpha() for c in tok)
    if len(digits) != 1 or letters < 3 or digits[0] not in LEET:
        return tok
    return tok.replace(digits[0], LEET[digits[0]])


def _name_tokens(s: str) -> list[str]:
    return [NAME_ABBR.get(t, t) for t in (_leet_fix(t) for t in _tokens(s))]


@dataclass(frozen=True)
class NameParts:
    name1: str        # primary normalised name (all tokens)
    name2: str        # alias (dba) or domain label, "" if none
    core1: str        # name1 without legal forms and stop words
    legal: str        # legal-form tokens of name1, sorted, space separated
    is_domain: bool   # the primary name was only a domain
    has_alias: bool


def parse_name(raw: object) -> NameParts:
    if not isinstance(raw, str) or not raw.strip():
        return NameParts("", "", "", "", False, False)
    s = unicodedata.normalize("NFC", html.unescape(raw)).strip()
    parts: list[str] = []
    for chunk in re.split(r"\s*\|\s*", s):
        parts.extend(p.strip() for p in _DBA.split(chunk) if p and p.strip())
    names: list[str] = []
    domains: list[str] = []
    for p in parts:
        m = _DOMAIN.match(p.strip("[]() "))
        if m:
            label = _TLD.sub("", m.group(1)).replace("-", " ").replace(".", " ")
            domains.append(label)
        else:
            names.append(p)
    is_domain = not names and bool(domains)
    prim = names[0] if names else (domains[0] if domains else s)
    alias = names[1] if len(names) > 1 else (domains[0] if names and domains else "")
    t1 = _name_tokens(prim)
    core = [t for t in t1 if t not in LEGAL and t not in STOP] or t1
    legal = sorted({t for t in t1 if t in LEGAL})
    t2 = " ".join(_name_tokens(alias)) if alias else ""
    return NameParts(" ".join(t1), t2, " ".join(core), " ".join(legal), is_domain, bool(alias))


def norm_address(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    toks: list[str] = []
    for t in _tokens(unicodedata.normalize("NFC", html.unescape(raw))):
        if len(t) > 5 and t.endswith("cdp"):
            t = t[:-3]  # `chicagocdp` -> `chicago`
        if t in ADDR_FILLER:
            continue
        toks.append(ADDR_ABBR.get(t, t))
    return " ".join(toks)


def norm_country(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    c = strip_accents(raw).lower().strip()
    return COUNTRY_ALIASES.get(c, c)


def nonlatin_frac(s: object) -> float:
    if not isinstance(s, str):
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(c) > 0x24F and not (0x1E00 <= ord(c) <= 0x1EFF) for c in letters) / len(letters)


def romanize(s: str) -> str:
    """Latin transliteration (anyascii, offline) of non-Latin text, tokenised like the rest; ASCII passes through.
    Indic scripts become approximate phonetic Latin (`राम मीडिया` -> `ram midiya`), enough for fuzzy similarity."""
    if s.isascii():
        return s
    return " ".join(_tokens(anyascii(s)))


def normalise_row(name: str, addr: str, country: str) -> tuple:
    """Row-level entry point used by the prepare stage. Returns a tuple in COLUMNS order."""
    p = parse_name(name)
    a = norm_address(addr)
    return (
        p.name1, p.name2, p.core1, p.legal, p.is_domain, p.has_alias,
        a, norm_country(country), nonlatin_frac(name), nonlatin_frac(addr), romanize(p.core1), romanize(a),
    )


COLUMNS = ["name1", "name2", "core1", "legal", "is_domain", "has_alias", "addr", "ctry", "nl_name", "nl_addr",
           "core_rom", "addr_rom"]
