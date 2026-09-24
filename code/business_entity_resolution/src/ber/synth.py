"""Tiny synthetic entity-resolution dataset in the competition layout, for tests and smoke runs."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

WORDS = ["Alpha", "Bright", "Cedar", "Delta", "Eagle", "Falcon", "Globe", "Harbor", "Indus", "Jade", "Kilo",
         "Lotus", "Metro", "Nova", "Orion", "Prime", "Quartz", "Royal", "Summit", "Titan", "Union", "Vertex"]
KINDS = ["Systems", "Traders", "Logistics", "Foods", "Textiles", "Motors", "Labs", "Holdings", "Retail"]
LEGAL = [("Corporation", "Corp"), ("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Incorporated", "Inc"), ("", "")]
STREETS = ["Main Street", "Park Road", "Lake Avenue", "MG Road", "Rue Victor Hugo", "Oak Boulevard", "Market Street"]
ABBR = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Boulevard": "Blvd"}


def _typo(s: str, rng: random.Random) -> str:
    if len(s) > 4 and rng.random() < 0.5:
        i = rng.randrange(1, len(s) - 1)
        return s[:i] + s[i + 1 :]
    return s


def _noisy(name: str, legal: tuple[str, str], addr: str, rng: random.Random) -> tuple[str, str]:
    n = f"{name} {legal[rng.random() < 0.5]}".strip()
    if rng.random() < 0.3:
        n = " ".join(reversed(n.split()[:2])) + " " + " ".join(n.split()[2:])
    n = _typo(n, rng)
    if rng.random() < 0.3:
        n = n.upper()
    a = addr
    for k, v in ABBR.items():
        if rng.random() < 0.6:
            a = a.replace(k, v)
    if rng.random() < 0.25:
        a = ",".join(a.split(",")[:-1]) or a
    return n.strip(), a


def make(root: str | Path, split: str, n: int = 300, countries: tuple[str, ...] = ("US", "India"), seed: int = 0) -> None:
    rng = random.Random(seed)
    root = Path(root) / split
    root.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 1
    for k in range(n):
        ctry = rng.choice(countries)
        name = f"{rng.choice(WORDS)} {rng.choice(KINDS)} {rng.choice(WORDS)}"
        legal = rng.choice(LEGAL)
        addr = f"{rng.randint(1, 999)} {rng.choice(STREETS)}, City{rng.randint(1, 30)}, {rng.randint(10000, 99999)}"
        eid = f"S1-{k + 1:05d}"
        s1.append((eid, f"{name} {legal[0]}".strip(), addr, ctry))
        matches = []
        for src in (2, 3):
            if rng.random() < 0.55:  # entity present in this source (0..2 records)
                for _ in range(rng.choice((1, 1, 2))):
                    nm, ad = _noisy(name, legal, addr, rng)
                    if src == 2:
                        i, c2 = f"S2-{c2:05d}", c2 + 1
                        s2.append((i, nm, ad, ctry))
                    else:
                        i, c3 = f"S3-{c3:05d}", c3 + 1
                        s3.append((i, nm, ad, ctry))
                    matches.append(i)
        gt.append((eid, ",".join(matches)))
    # unmatched distractor records
    for src in (2, 3):
        for _ in range(n // 4):
            ctry = rng.choice(countries)
            name = f"{rng.choice(WORDS)} {rng.choice(KINDS)} {rng.choice(WORDS)}"
            addr = f"{rng.randint(1, 999)} {rng.choice(STREETS)}, City{rng.randint(1, 30)}, {rng.randint(10000, 99999)}"
            if src == 2:
                s2.append((f"S2-{c2:05d}", name, addr, ctry)); c2 += 1
            else:
                s3.append((f"S3-{c3:05d}", name, addr, ctry)); c3 += 1
    cols = ["entity_id", "business_name", "business_address", "country"]
    for i, rows in enumerate((s1, s2, s3), 1):
        pd.DataFrame(rows, columns=cols).sample(frac=1, random_state=seed).to_csv(root / f"{split}_source{i}.tsv", sep="\t", index=False)
    if split == "train":
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(root / "train_ground_truth.tsv", sep="\t", index=False)
