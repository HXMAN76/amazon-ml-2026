"""xenc_fr: France-aware fine-tuning data for the cross-encoder, built from TRAIN records only (no test record is ever a training example).

France (test only, no labels) holds kinds of true copies that are rare in train and a kind of decoy train never shows (research.md 25 to 27):
initials as the whole name (`lc` for `lille club`), legal forms spelled out letter by letter (`s a r l`), glued words, and sibling
businesses whose name is the S1's with one common word swapped (`nje ecole` against `nje centre`). This stage adds synthetic pairs of those
kinds to a replay sample of the fit set the team's e5-base cross-encoder was trained on, so a short warm-started fine-tune teaches them:

  positives (label 1): a true train copy rewritten by one copy operator: `ini` (initials of its core words), `spell` (a short legal form
                       spelled out), `glue` (two neighbouring core words joined)
  negatives (label 0): a true train copy with one common core word that it shares with its S1 swapped for another common word (`swap`;
                       common = in at least 100 train S1 core names; train's own noise words are never swapped in or out), and optionally
                       the copy renamed to another train business of its country (`tenant`: another tenant at the same address)

The pool record keeps its own (noisy) address, so the model learns the name difference, not an address artefact. Also writes the pair lists
to score: the France pairs of the all-pairs test list and the locked-holdout pairs of the all-pairs train list (labelled, for calibration).

  python -m ber.stages.xenc_fr --base-dir xenc2_v7 --all-dir xenc2F_v7 --dir xencFR [--set n_replay=500000,n_swap=150000,n_tenant=0]
  python -m ber.stages.xenc train --dir xencFR --base-model $BER_WORK/xenc2/model --model-dir xencFR1/model --set epochs=1,lr=0.00001,batch=64
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from ber import config
from ber.split import holdout_q
from ber.stages.block import PID_BASE
from ber.stages.xenc import tag_digits
from ber.text import LEGAL

NOISE = {"center", "centre", "services", "service"}  # the generator's word-swap noise in train (true copies swap these; research.md 25)
DEFAULTS = {"seed": 7, "n_replay": 500_000, "n_ini": 50_000, "n_spell": 50_000, "n_glue": 50_000, "n_swap": 150_000, "n_tenant": 0, "df_min": 100}


def initials(core: str) -> str:
    """First letters of the first three core words: `lille club` -> `lc`."""
    return "".join(w[0] for w in core.split()[:3])


def spell(name: str) -> str | None:
    """Spell out the first short legal-form token letter by letter (`sarl` -> `s a r l`, `llc` -> `l l c`); None if the name has none."""
    toks = name.split()
    for i, t in enumerate(toks):
        if t in LEGAL and 2 <= len(t) <= 4 and t.isalpha():
            return " ".join(toks[:i] + list(t) + toks[i + 1:])
    return None


def glue(name: str, core: str, rng: np.random.Generator) -> str | None:
    """Join two neighbouring core words of the name (`ecole maternelle` -> `ecolematernelle`); None if there are no two."""
    toks, cw = name.split(), set(core.split())
    pos = [i for i in range(len(toks) - 1) if toks[i] in cw and toks[i + 1] in cw]
    if not pos:
        return None
    i = int(rng.choice(pos))
    return " ".join(toks[:i] + [toks[i] + toks[i + 1]] + toks[i + 2:])


def swap(name: str, s1_core: set[str], common: np.ndarray, common_set: set[str], rng: np.random.Generator) -> str | None:
    """Replace one common word the name shares with its S1 by another common word the S1 does not have; None if no word qualifies."""
    toks = name.split()
    pos = [i for i, t in enumerate(toks) if t in s1_core and t in common_set and t not in NOISE and t not in LEGAL]
    if not pos:
        return None
    i = int(rng.choice(pos))
    for _ in range(20):
        w = str(common[rng.integers(len(common))])
        if w not in s1_core and w not in NOISE and w != toks[i]:
            return " ".join(toks[:i] + [w] + toks[i + 1:])
    return None


def _records(split: str, src: int) -> pl.DataFrame:
    return pl.read_parquet(config.paths()["parquet"] / split / f"source{src}.parquet", columns=["rid", "name1", "core1", "addr", "ctry", "nl_name"])


def synth(fit: pl.DataFrame, prm: dict) -> pl.DataFrame:
    """Synthetic pairs (q, pid, p, label, ta, tb) from the true pairs of the fit set; ta is the S1 text exactly as in the fit set."""
    rng = np.random.default_rng(int(prm["seed"]))
    s1 = _records("train", 1).rename({"rid": "q", "name1": "a_name", "core1": "a_core", "addr": "a_addr", "nl_name": "a_nl"}).with_columns(pl.col("q").cast(pl.Int64))
    pool = pl.concat([_records("train", s).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid")) for s in (2, 3)]).drop("rid", "ctry")
    pos = (fit.filter(pl.col("label") == 1).select("q", "pid", "ta").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
              .join(s1, on="q", how="left").join(pool, on="pid", how="left")
              .filter((pl.col("nl_name") < 0.5) & (pl.col("a_nl") < 0.5) & (pl.col("core1").str.count_matches(" ") >= 1)))
    df = s1.select(pl.col("a_core").str.split(" ").list.unique().alias("t")).explode("t").group_by("t").len()
    common = df.filter((pl.col("len") >= int(prm["df_min"])) & (pl.col("t").str.len_chars() > 2) & pl.col("t").str.contains(r"^[a-z]+$"))["t"].to_numpy()
    common = np.array([w for w in common if w not in NOISE and w not in LEGAL])
    common_set = set(common.tolist())
    names = s1.filter(pl.col("a_nl") < 0.5).select("ctry", "a_name")
    by_ctry = {k[0]: g["a_name"].to_numpy() for k, g in names.group_by("ctry")}
    rows = pos.to_dicts()
    order = rng.permutation(len(rows))
    out, used = [], {"ini": 0, "spell": 0, "glue": 0, "swap": 0, "tenant": 0}
    want = {k: int(prm[f"n_{k}"]) for k in used}
    for j in order:
        r = rows[j]
        for kind in ("swap", "ini", "spell", "glue", "tenant"):  # one synthetic pair per true pair, the first kind still short of its quota
            if used[kind] >= want[kind]:
                continue
            if kind == "ini":
                nm, lab = initials(r["core1"]), 1
            elif kind == "spell":
                nm, lab = spell(r["name1"]), 1
            elif kind == "glue":
                nm, lab = glue(r["name1"], r["core1"], rng), 1
            elif kind == "swap":
                nm, lab = swap(r["name1"], set(r["a_core"].split()), common, common_set, rng), 0
            else:
                cand = by_ctry.get(r["ctry"])
                nm, lab = (str(cand[rng.integers(len(cand))]) if cand is not None else None), 0
                if nm is not None and set(nm.split()) & set(r["a_core"].split()):
                    nm = None  # shares a word with the S1: could be a real variant, not another tenant
            if not nm or nm == r["name1"]:
                continue
            out.append({"q": r["q"], "pid": r["pid"], "p": -1.0, "label": lab, "ta": r["ta"], "tb": tag_digits(f"{nm} | {r['addr']}"), "kind": kind})
            used[kind] += 1
            break
        if all(used[k] >= want[k] for k in used):
            break
    print(f"synthetic pairs: {used} from {len(rows)} true train pairs; common vocabulary {len(common)} words", flush=True)
    return pl.DataFrame(out, schema={"q": pl.Int64, "pid": pl.Int64, "p": pl.Float64, "label": pl.Int64, "ta": pl.Utf8, "tb": pl.Utf8, "kind": pl.Utf8})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="xenc2_v7", help="WORK folder of the fit set the base cross-encoder was trained on")
    ap.add_argument("--all-dir", default="xenc2F_v7", help="WORK folder of the all-shortlisted-pairs lists (test.parquet, train.parquet)")
    ap.add_argument("--dir", default="xencFR", help="output WORK folder")
    ap.add_argument("--set", default="", help="comma-separated overrides of " + ",".join(DEFAULTS))
    a = ap.parse_args(argv)
    prm = dict(DEFAULTS)
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        prm[k] = type(DEFAULTS[k])(v)
    P = config.paths()
    t0 = time.time()
    out = P["work"] / a.dir
    out.mkdir(parents=True, exist_ok=True)
    fit = pl.read_parquet(P["work"] / a.base_dir / "train_fit.parquet")
    syn = synth(fit, prm)
    rep = fit.sample(min(int(prm["n_replay"]), fit.height), seed=int(prm["seed"])).with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64),
                                                                                             pl.col("p").cast(pl.Float64), pl.col("label").cast(pl.Int64), pl.lit("replay").alias("kind"))
    d = pl.concat([rep.select(syn.columns), syn]).sample(fraction=1.0, shuffle=True, seed=int(prm["seed"]))
    d.write_parquet(out / "train_fit.parquet", compression="zstd")
    print(f"fit set {d.height} pairs ({int(d['label'].sum())} positive): {d.group_by('kind', 'label').len().sort('kind').rows()}", flush=True)
    fr = pl.read_parquet(P["parquet"] / "test" / "source1.parquet", columns=["rid", "ctry"]).filter(pl.col("ctry") == "france").select(pl.col("rid").cast(pl.Int64).alias("q"))
    te = pl.read_parquet(P["work"] / a.all_dir / "test.parquet").with_columns(pl.col("q").cast(pl.Int64)).join(fr, on="q", how="semi")
    te.write_parquet(out / "test.parquet", compression="zstd")
    hq = pl.DataFrame({"q": holdout_q().astype(np.int64)})
    tr = pl.read_parquet(P["work"] / a.all_dir / "train.parquet").with_columns(pl.col("q").cast(pl.Int64)).join(hq, on="q", how="semi")
    tr.write_parquet(out / "train.parquet", compression="zstd")
    print(f"to score: {te.height} France test pairs, {tr.height} holdout pairs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
