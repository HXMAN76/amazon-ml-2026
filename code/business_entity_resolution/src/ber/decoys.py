"""Synthetic look-alike decoys, used as extra negatives when fine-tuning the encoders.

84% of false positives are near copies of a business that belong to no S1 (research.md section 12): one name word with a few
letters replaced near its end (`Jarlent` -> `Jarleix`, `Olanus` -> `Olanuz`) and a house number with a digit added, dropped or
changed (`312` -> `323`, `34765` -> `34768`). True copies carry typos too, so a decoy changes letters in place and moves a number.
Built from training records only.
"""

from __future__ import annotations

import re

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"
_NUM = re.compile(r"\d+")


def _swap_letters(word: str, rng: np.random.Generator) -> str:
    """Replace 1 to 3 letters in the second half of the word by different letters (same length, case kept)."""
    pos = [i for i in range(len(word) // 2, len(word)) if word[i].isalpha()]
    if not pos:
        return word
    chars = list(word)
    for i in rng.choice(pos, size=min(len(pos), int(rng.integers(1, 4))), replace=False):
        c = chars[i].lower()
        new = LETTERS[(LETTERS.index(c) + int(rng.integers(1, 26))) % 26] if c in LETTERS else "x"
        chars[i] = new.upper() if chars[i].isupper() else new
    return "".join(chars)


def _move_number(addr: str, rng: np.random.Generator) -> str:
    """Add, drop or change the last digit of the first number in the address (unchanged if it has none)."""
    m = _NUM.search(addr)
    if not m:
        return addr
    n = m.group()
    op = int(rng.integers(3)) if len(n) > 1 else int(rng.choice([0, 2]))
    last = str(int(rng.integers(10)))
    if op == 0:
        n2 = n + last
    elif op == 1:
        n2 = n[:-1]
    else:
        n2 = n[:-1] + str((int(n[-1]) + int(rng.integers(1, 10))) % 10)
    return addr[: m.start()] + n2 + addr[m.end():]


def make_decoy(name: str, addr: str, rng: np.random.Generator) -> tuple[str, str]:
    """A look-alike of (name, addr): the longest name word edited, the house number moved, or both (never neither)."""
    words = name.split()
    can_name = any(len(w) >= 4 for w in words)
    can_num = bool(_NUM.search(addr))
    mode = int(rng.integers(3)) if can_name and can_num else (0 if can_name else 1)  # 0 name, 1 number, 2 both
    if mode in (0, 2) and can_name:
        i = max(range(len(words)), key=lambda j: len(words[j]))
        words[i] = _swap_letters(words[i], rng)
        name = " ".join(words)
    if mode in (1, 2) and can_num:
        addr = _move_number(addr, rng)
    return name, addr
