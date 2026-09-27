"""French-aware cross-encoder from French-ized training pairs (training data only, labels unchanged).
Usage: python src/scripts/france/xfz.py make WORKDIR N_ORIG N_FR | evalset WORKDIR N | eval WORKDIR MODEL_DIR
The US/India pair texts (`name1 | addr`, normalised, digits tagged) are rewritten into French form by one deterministic token map applied to
both records of a pair, so a copy stays a copy and a distractor stays a distractor: legal forms (llc -> sarl, incorporated -> sas, ...), category
words (bakery -> boulangerie, school -> ecole, ...) moved to the front of the name as in French names, and/et, sons/fils, group/groupe; street
types in French right after the house number (123 main st -> 123 rue main), US state codes and names mapped to French departments / regions.
  make     WORKDIR/train_fit.parquet: N_ORIG original + N_FR French-ized pairs sampled from xenc2/train_fit.parquet (distinct pairs)
  evalset  WORKDIR/eval_{orig,fr}.parquet: N labelled shortlisted train pairs whose S1 were not used to fit (xenc2F_v7/train.parquet)
  eval     average precision on both eval sets and the true share by score bin (calibration for France decisions)"""

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

LEG = {"llc": "sarl", "incorporated": "sas", "corporation": "sa", "limited": "sarl", "private": "", "company": "cie", "llp": "snc", "lp": "scs",
       "plc": "sa", "pllc": "selarl", "partners": "associes", "partnership": "associes", "ltda": "sarl", "pty": "", "opc": "sasu", "inc": "sas", "ltd": "sarl"}
WORD = {"and": "et", "sons": "fils", "son": "fils", "group": "groupe", "center": "centre", "clinic": "clinique", "school": "ecole", "pharmacy": "pharmacie",
        "bakery": "boulangerie", "church": "eglise", "hospital": "hopital", "health": "sante", "care": "soins", "dental": "dentaire", "family": "famille",
        "community": "communaute", "society": "societe", "foundation": "fondation", "academy": "academie", "institute": "institut", "university": "universite",
        "home": "maison", "house": "maison", "studio": "atelier", "workshop": "atelier", "management": "gestion", "consulting": "conseil", "consultants": "conseil",
        "trading": "commerce", "systems": "systemes", "enterprises": "entreprises", "brothers": "freres", "associates": "associes", "youth": "jeunes",
        "friends": "amis", "music": "musique", "dance": "danse", "theater": "theatre", "primary": "primaire", "elementary": "elementaire", "nursery": "maternelle",
        "committee": "comite", "council": "conseil", "heritage": "patrimoine", "office": "bureau", "sports": "sports", "restaurant": "restaurant", "shop": "boutique",
        "store": "magasin", "market": "marche", "services": "services", "service": "service", "international": "international", "national": "nationale",
        "medical": "medical", "insurance": "assurance", "bank": "banque", "hotel": "hotel", "club": "club", "association": "association", "federation": "federation",
        "union": "union", "garage": "garage", "auto": "auto", "motors": "automobiles", "construction": "construction", "transport": "transport", "logistics": "logistique",
        "technologies": "technologies", "solutions": "solutions", "industries": "industries", "the": "", "of": "de"}
CAT = {"boulangerie", "ecole", "clinique", "pharmacie", "eglise", "hopital", "sante", "soins", "academie", "institut", "universite", "maison", "atelier", "gestion",
       "conseil", "commerce", "association", "club", "federation", "union", "comite", "restaurant", "boutique", "magasin", "garage", "banque", "hotel",
       "musique", "danse", "theatre", "fondation", "centre", "communaute"}
ADDR = {"st": "rue", "street": "rue", "road": "route", "rd": "route", "drive": "allee", "lane": "impasse", "court": "cour", "highway": "route", "parkway": "boulevard",
        "circle": "rond-point", "way": "chemin", "trail": "sentier", "suite": "bureau", "floor": "etage", "building": "batiment", "near": "pres", "opposite": "face",
        "apartment": "appartement", "sector": "secteur", "market": "marche", "terrace": "terrasse", "pike": "route", "plaza": "place"}
STYPES = {"rue", "route", "avenue", "allee", "impasse", "cour", "place", "square", "boulevard", "chemin", "sentier", "rond-point", "terrasse", "quai", "cours"}
DEPTS = ["gironde", "nord", "loire atlantique", "pas de calais", "rhone", "bouches du rhone", "haute garonne", "herault", "bas rhin", "seine maritime",
         "isere", "alpes maritimes", "var", "finistere", "ille et vilaine", "maine et loire", "moselle", "calvados", "somme", "marne"]
REGIONS = ["nouvelle aquitaine", "hauts de france", "pays de la loire", "occitanie", "bretagne", "normandie", "grand est", "provence alpes cote d azur",
           "auvergne rhone alpes", "ile de france", "centre val de loire", "bourgogne franche comte"]
US_CODES = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
            "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]
US_NAMES = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
            "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
            "montana", "nebraska", "nevada", "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
            "wisconsin", "wyoming"]
STATE = {c: DEPTS[i % len(DEPTS)] for i, c in enumerate(US_CODES)} | {n: REGIONS[i % len(REGIONS)] for i, n in enumerate(US_NAMES)}
NUM = re.compile(r"\[[NP]\]")


def fr_name(s: str) -> str:
    out = []
    for t in s.split():
        t = LEG.get(t, WORD.get(t, t))
        if t:
            out.extend(t.split())
    cats = [t for t in out if t in CAT]
    rest = [t for t in out if t not in CAT]
    return " ".join(cats + rest) if cats and rest and out[0] not in CAT else " ".join(out)


def fr_addr(s: str) -> str:
    toks = []
    for t in s.split():
        t = ADDR.get(t, STATE.get(t, t))
        if t:
            toks.extend(t.split())
    i = next((k for k, t in enumerate(toks) if NUM.search(t)), None)
    if i is not None:
        j = next((k for k in range(i + 1, len(toks)) if toks[k] in STYPES), None)
        if j is not None and j > i + 1:
            toks.insert(i + 1, toks.pop(j))
    return " ".join(toks)


def frenchify(t: str) -> str:
    if " | " in t:
        n, a = t.split(" | ", 1)
        return fr_name(n) + " | " + fr_addr(a)
    return fr_name(t)


def main() -> None:
    cmd, wd = sys.argv[1], Path(sys.argv[2])
    src = Path("/home/ec2-user/SageMaker/work/xsrc")
    if cmd == "make":
        n_orig, n_fr = int(sys.argv[3]), int(sys.argv[4])
        f = pl.read_parquet(src / "train_fit.parquet").sample(fraction=1.0, shuffle=True, seed=7)
        a, b = f.head(n_orig), f.slice(n_orig, n_fr)
        b = b.with_columns(pl.col("ta").map_elements(frenchify, return_dtype=pl.Utf8), pl.col("tb").map_elements(frenchify, return_dtype=pl.Utf8))
        out = pl.concat([a, b]).sample(fraction=1.0, shuffle=True, seed=8)
        wd.mkdir(parents=True, exist_ok=True)
        out.write_parquet(wd / "train_fit.parquet")
        print(f"fit set {out.height} ({n_orig} original, {n_fr} French-ized), positives {int(out['label'].sum())}")
        for r in b.head(12).iter_rows(named=True):
            print(f"  {r['label']} | {r['ta']}  <>  {r['tb']}")
    elif cmd == "evalset":
        n = int(sys.argv[3])
        fitq = pl.read_parquet(src / "train_fit.parquet", columns=["q"]).unique()
        e = pl.read_parquet(src / "train_full.parquet").join(fitq, on="q", how="anti")
        e = e.filter(pl.col("p") >= 0.3).sample(n=min(n, e.height), seed=9)  # the decisions happen above the stack's low-p tail
        wd.mkdir(parents=True, exist_ok=True)
        e.write_parquet(wd / "eval_orig.parquet")
        e.with_columns(pl.col("ta").map_elements(frenchify, return_dtype=pl.Utf8), pl.col("tb").map_elements(frenchify, return_dtype=pl.Utf8)).write_parquet(wd / "eval_fr.parquet")
        print(f"eval set {e.height} pairs, positives {int(e['label'].sum())}")
    elif cmd == "eval":
        from sklearn.metrics import average_precision_score
        for nm in ("orig", "fr"):
            d = pl.read_parquet(wd / f"eval_{nm}_xs.parquet").join(pl.read_parquet(wd / f"eval_{nm}.parquet", columns=["q", "pid", "label", "p"]), on=["q", "pid"])
            y, xs = d["label"].to_numpy(), d["xs"].to_numpy()
            print(f"{nm}: AP {average_precision_score(y, xs):.4f}  errors at 0.5: {int(((xs >= 0.5) != (y == 1)).sum())} of {len(y)}", flush=True)
            if nm == "fr":
                b = d.with_columns(pl.col("xs").cut([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]).alias("bin")).group_by("bin").agg(pl.len(), pl.col("label").mean().alias("true_share")).sort("bin")
                print(b)


if __name__ == "__main__":
    main()
