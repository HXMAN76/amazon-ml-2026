"""Build France decoy-rule variants of a model's output. Usage:
  python src/scripts/france_variants.py NAME NEWNAME --rules swap_exact,weak:80:0.9999:shared,legal:0.9999,thr:0.985,tiny:0.9999 [--country france]
Rules (a pair is dropped when any rule fires; only the given country; probabilities of everything else are unchanged, the candidate file is unchanged):
  swap_exact[:pmax]     one common word swapped (word_swap.py) and the S1 has a confident exact-name copy, p < pmax (default 0.9999)
  swap_all[:pmax]       one common word swapped, p < pmax
  weak:T:pmax[:shared]  name similarity (name_tset) < T, address similarity (addr_tset) >= 90, same house number, p < pmax; `shared`: the S1's address is
                        shared with another S1 (a multi-tenant building)
  legal:pmax            legal-form conflict (both sides name a different form), p < pmax
  legalhouse:pmax       legal-form conflict and a different house number, p < pmax
  thr:t                 p < t
  thrp:t                p < t unless the pair is protected: equal names after removing spaced legal forms, or a pool name of at most 3 letters that is a subsequence of the S1's initials
  thrpn:t               thrp that also spares one-word swaps into France's noise words (fils, groupe, services, developpement)
  thrx:t                p < t and the core names differ (exact-name pairs keep their probability: the slot-limit fit finds no decoys among exact-name pairs)
  typeswap:pmax[:R]     swap whose swapped-in word is a type word of the country's vocabulary (club, ecole, comite, ...): words whose rate among the S1's swap pairs does not fall when
                        the S1 already has three or more exact copies (ratio A/B >= R, default 0.75; see swap_words.py): decoys draw their new word from that vocabulary, true
                        swaps from generic suffix words (services, groupe, france); p < pmax
  tiny:pmax             pool name of at most 3 letters that is not a subsequence of the initials of the S1's core name, p < pmax
  typeins:pmax[:R]      the pool name is the S1's core name plus one inserted word that is a type word (the typeswap vocabulary), p < pmax
  xfr:DIR:t[:pmax]      the France-aware cross-encoder score in WORK/DIR/test_xs.parquet (stages/xenc_fr.py) is below t, p < pmax (default 1.01)
  restore:KINDS:pmin    not a rule: add back shortlisted France pairs below the decision whose pool record nobody owns, whose name relation is one of KINDS
                        (exact, spelled_legal, initials, glued, noise_swap; joined by +; france_recall.py), at the S1's address (same house number,
                        address similarity >= 90), p >= pmin, within the S1's free slots (5 S2 / 6 S3), best p first
  protect:pmin          not a rule: an S1 that the soft rules (thr, thrp, thrx, xfr) would leave with an empty list keeps its best such pair if p >= pmin. The
                        metric is per S1: emptying an S1 that has a true match costs it everything, one wrong extra pair on a full S1 costs about 0.1
Prints the number of pairs each rule drops with its decoy share from the slot-limit fit (decoy_by_category.py; worth dropping above about 26%)
and the total, then writes WORK/output/NEWNAME like reemit.py."""

import argparse
import json
import time

import numpy as np
import polars as pl

from ber import config, decision
from ber.stages.predict import emit
from word_swap import flag, strip_spaced, tok_df

PID_BASE = 10_000_000
NOISE_FR = ["fils", "groupe", "services", "developpement"]  # the words true France copies swap in (research.md 25; france_recall.py)


def is_subseq(short: str, initials: str) -> bool:
    it = iter(initials)
    return all(c in it for c in short)


def decoy_share(pairs: pl.DataFrame, slots: pl.DataFrame) -> list[float]:
    """Decoy share of a set of pairs in S2 and S3 from slot occupancy (decoy_by_category.py): true copies fill the (cap - k) free slots of an S1
    with k confident exact-name copies, decoys arrive at a rate that does not depend on k; fit rate(k) = D + T * (cap - k) / cap."""
    per = slots.join(pairs.group_by("q", "src").len().rename({"len": "n"}), on=["q", "src"], how="left").with_columns(pl.col("n").fill_null(0))
    out = []
    for src, cap in ((2, 5), (3, 6)):
        g = per.filter((pl.col("src") == src) & (pl.col("k") <= cap)).group_by("k").agg(pl.len().alias("w"), pl.col("n").mean().alias("r")).sort("k")
        k, w, r = (g[c].to_numpy().astype(float) for c in ("k", "w", "r"))
        A = np.vstack([np.ones_like(k), (cap - k) / cap]).T * np.sqrt(w)[:, None]
        (d, _), *_ = np.linalg.lstsq(A, r * np.sqrt(w), rcond=None)
        tot = float((w * r).sum())
        out.append(round(min(1.0, max(float(d), 0.0) * w.sum() / tot), 3) if tot > 0 else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("newname")
    ap.add_argument("--rules", default="")
    ap.add_argument("--country", default="france")
    ap.add_argument("--cap", action="store_true", help="after the rules, keep at most 5 S2 and 6 S3 pairs per S1 in EVERY country (the training maximum), lowest probability first")
    ap.add_argument("--no-rules", action="store_true", help="skip the country rules (use with --cap alone)")
    a = ap.parse_args()
    P = config.paths()
    t0 = time.time()
    cfg = json.loads((P["work"] / "models" / a.name / "config.json").read_text())
    thr = cfg["threshold"]
    pq = P["parquet"] / "test"
    s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "core1", "addr", "ctry"]).rename({"rid": "q", "core1": "a_core"}).with_columns(pl.col("q").cast(pl.Int64))
    s1 = s1.join(s1.group_by("addr").len().rename({"len": "addr_n"}), on="addr", how="left").drop("addr")
    pool = pl.concat([pl.read_parquet(pq / f"source{s}.parquet", columns=["rid", "core1"]).with_columns((pl.col("rid").cast(pl.Int64) + s * PID_BASE).alias("pid"), pl.lit(s).alias("src")) for s in (2, 3)]).drop("rid").rename({"core1": "b_core"})
    df = tok_df(s1.rename({"a_core": "core1"}))
    pp = pl.read_parquet(P["work"] / "output" / a.name / "pair_p.parquet").with_columns(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64))
    own = decision.assign_exclusive(pp).filter(pl.col("p") >= thr).join(s1, on="q", how="left").join(pool, on="pid", how="left")
    own = own.filter(pl.col("ctry") == a.country)
    feat = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
              .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "name_tset", "addr_tset", "house_eq", "legal_conflict")
              .join(own.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
    own = flag(own.join(feat, on=["q", "pid"], how="left"), df).with_columns((pl.col("a_core") == pl.col("b_core")).alias("core_eq"))
    exact = own.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q").len().rename({"len": "n_exact"})
    own = own.join(exact, on="q", how="left").with_columns(pl.col("n_exact").fill_null(0))
    rules_list = [] if a.no_rules else a.rules.split(",")
    protect = [float(r.split(":")[1]) for r in rules_list if r.startswith("protect:")]
    restore = [r.split(":") for r in rules_list if r.startswith("restore:")]
    rules_list = [r for r in rules_list if not r.startswith(("protect:", "restore:"))]
    exs = own.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
    slots = s1.filter(pl.col("ctry") == a.country).select("q").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
    if any(s.startswith(("typeswap", "typeins")) for s in rules_list):
        nA, nB = slots.filter(pl.col("k") >= 3).height, slots.filter(pl.col("k") == 0).height
        sw = own.filter(pl.col("swap")).join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
        sw = sw.with_columns(pl.col("a_core").str.split(" ").list.unique().alias("ta"), pl.col("b_core").str.split(" ").list.unique().alias("tb"))
        sw = sw.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).list.first().alias("xb"))
        ra = sw.filter(pl.col("k") >= 3).group_by("xb").len().rename({"len": "nA"})
        rb = sw.filter(pl.col("k") == 0).group_by("xb").len().rename({"len": "nB"})
        rr = ra.join(rb, on="xb", how="full", coalesce=True).with_columns(pl.col("nA").fill_null(0), pl.col("nB").fill_null(0)).with_columns(((pl.col("nA") / nA) / (pl.col("nB") / nB + 1e-12)).alias("ratio"))
        type_words = {}
        rr_all = rr
        own = own.with_columns(pl.col("b_core").alias("_b"))
        print(f"typeswap: slots A {nA}, B {nB}; words with ratio >= 0.75 and at least 100 pairs: {rr.filter((pl.col('ratio') >= 0.75) & (pl.col('nA') + pl.col('nB') >= 100)).height}", flush=True)
    fired = []
    for spec in rules_list:
        k = spec.split(":")
        if k[0] == "swap_exact":
            c = pl.col("swap") & (pl.col("n_exact") >= 1) & (pl.col("p") < float(k[1] if len(k) > 1 else 0.9999))
        elif k[0] == "swap_all":
            c = pl.col("swap") & (pl.col("p") < float(k[1] if len(k) > 1 else 0.9999))
        elif k[0] == "weak":
            c = (pl.col("name_tset") < float(k[1])) & (pl.col("addr_tset") >= 90) & (pl.col("house_eq") > 0.5) & (pl.col("p") < float(k[2]))
            if len(k) > 3 and k[3] == "shared":
                c = c & (pl.col("addr_n") >= 2)
        elif k[0] == "legal":
            c = (pl.col("legal_conflict") > 0.5) & (pl.col("p") < float(k[1]))
        elif k[0] == "legalhouse":  # a sibling next door: other legal form and another house number (research: emptied_samples.py examples)
            c = (pl.col("legal_conflict") > 0.5) & (pl.col("house_eq") < 0.5) & (pl.col("p") < float(k[1]))
        elif k[0] == "thr":
            c = pl.col("p") < float(k[1])
        elif k[0] == "thrp":
            tiny = own.filter((pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ")).select("q", "pid", "a_core", "b_core")
            ok = [(q, pid) for q, pid, ac, bc in tiny.iter_rows() if is_subseq(bc, "".join(t_[0] for t_ in ac.split()))]
            ok_df = pl.DataFrame({"q": [x[0] for x in ok], "pid": [x[1] for x in ok]}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_ini"))
            own = own.join(ok_df, on=["q", "pid"], how="left").with_columns(pl.col("_ini").fill_null(False).alias("ini_ok"))
            own = own.with_columns((strip_spaced(pl.col("a_core")) == strip_spaced(pl.col("b_core"))).alias("eq_norm"))
            c = (pl.col("p") < float(k[1])) & ~pl.col("eq_norm") & ~pl.col("ini_ok")
        elif k[0] == "thrpn":  # thrp that also spares one-word swaps into France's noise words (true copies: research.md 25, france_recall.py)
            if "ini_ok" not in own.columns:
                tiny = own.filter((pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ")).select("q", "pid", "a_core", "b_core")
                ok = [(q, pid) for q, pid, ac, bc in tiny.iter_rows() if is_subseq(bc, "".join(t_[0] for t_ in ac.split()))]
                ok_df = pl.DataFrame({"q": [x[0] for x in ok], "pid": [x[1] for x in ok]}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_ini"))
                own = own.join(ok_df, on=["q", "pid"], how="left").with_columns(pl.col("_ini").fill_null(False).alias("ini_ok")).drop("_ini")
                own = own.with_columns((strip_spaced(pl.col("a_core")) == strip_spaced(pl.col("b_core"))).alias("eq_norm"))
            ta, tb = pl.col("a_core").str.split(" ").list.unique(), pl.col("b_core").str.split(" ").list.unique()
            own = own.with_columns(((ta.list.set_difference(tb).list.len() == 1) & (tb.list.set_difference(ta).list.len() == 1)
                                    & tb.list.set_difference(ta).list.first().is_in(NOISE_FR)).alias("noise_swap"))
            c = (pl.col("p") < float(k[1])) & ~pl.col("eq_norm") & ~pl.col("ini_ok") & ~pl.col("noise_swap")
        elif k[0] == "thrx":
            c = (pl.col("p") < float(k[1])) & ~pl.col("core_eq")
        elif k[0] == "typeswap":
            rmin = float(k[2]) if len(k) > 2 else 0.75
            words = rr_all.filter((pl.col("ratio") >= rmin) & (pl.col("nA") + pl.col("nB") >= 100))["xb"].to_list()
            own = own.with_columns(pl.col("b_core").str.split(" ").list.unique().alias("_tb"), pl.col("a_core").str.split(" ").list.unique().alias("_ta"))
            own = own.with_columns(pl.col("_tb").list.set_difference(pl.col("_ta")).list.first().alias("_xb"))
            c = pl.col("swap") & pl.col("_xb").is_in(words) & (pl.col("p") < float(k[1]))
        elif k[0] == "typeins":
            rmin = float(k[2]) if len(k) > 2 else 0.75
            words = rr_all.filter((pl.col("ratio") >= rmin) & (pl.col("nA") + pl.col("nB") >= 100))["xb"].to_list()
            own = own.with_columns(pl.col("b_core").str.split(" ").list.unique().alias("_tb"), pl.col("a_core").str.split(" ").list.unique().alias("_ta"))
            ins = (pl.col("_ta").list.set_difference(pl.col("_tb")).list.len() == 0) & (pl.col("_tb").list.set_difference(pl.col("_ta")).list.len() == 1)
            c = ins & pl.col("_tb").list.set_difference(pl.col("_ta")).list.first().is_in(words) & (pl.col("p") < float(k[1]))
        elif k[0] == "xfr":
            xf = pl.read_parquet(P["work"] / k[1] / "test_xs.parquet").select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), pl.col("xs").alias("_xf"))
            own = own.drop("_xf", strict=False).join(xf, on=["q", "pid"], how="left")
            print(f"xfr: {own['_xf'].is_null().sum()} of {own.height} predicted pairs have no score", flush=True)
            c = (pl.col("_xf") < float(k[2])) & (pl.col("p") < float(k[3] if len(k) > 3 else 1.01))
        elif k[0] == "tiny":
            tiny = own.filter((pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ") & (pl.col("p") < float(k[1]))).select("q", "pid", "a_core", "b_core")
            bad = [(q, pid) for q, pid, ac, bc in tiny.iter_rows() if not is_subseq(bc, "".join(t[0] for t in ac.split()))]
            bad_df = pl.DataFrame({"q": [x[0] for x in bad], "pid": [x[1] for x in bad]}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_tiny"))
            own = own.join(bad_df, on=["q", "pid"], how="left").with_columns(pl.col("_tiny").fill_null(False).alias("tiny_bad"))
            c = pl.col("tiny_bad")
        else:
            raise SystemExit(f"unknown rule {spec}")
        n = own.filter(c).height
        fired.append((k[0], c))
        print(f"rule {spec}: fires on {n} of {own.height} predicted {a.country} pairs ({n / own.height:.4f}); decoy share S2, S3 {decoy_share(own.filter(c), slots)}", flush=True)
    if fired:
        any_c = pl.any_horizontal([c.fill_null(False) for _, c in fired])
        dropped = own.filter(any_c).select("q", "pid").with_columns(pl.lit(True).alias("_drop"))
        if protect:
            hard = [c.fill_null(False) for r, c in fired if r not in {"thr", "thrp", "thrx", "xfr"}]
            o2 = own.with_columns(any_c.alias("_d"), (pl.any_horizontal(hard) if hard else pl.lit(False)).alias("_h"))
            alive = o2.filter(~pl.col("_d")).select("q").unique()
            back = (o2.filter(pl.col("_d") & ~pl.col("_h") & (pl.col("p") >= protect[0])).join(alive, on="q", how="anti")
                      .sort("p", descending=True).group_by("q").first().select("q", "pid"))
            emptied = o2.select("q").unique().join(alive, on="q", how="anti").height
            dropped = dropped.join(back, on=["q", "pid"], how="anti")
            print(f"protect {protect[0]}: the rules empty {emptied} {a.country} S1; {back.height} of them keep their best soft-dropped pair "
                  f"(decoy share of those pairs S2, S3 {decoy_share(back.join(own.select('q', 'pid', 'src'), on=['q', 'pid'], how='left'), slots)})", flush=True)
    else:
        dropped = pl.DataFrame({"q": [], "pid": []}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_drop"))
    print(f"total dropped {dropped.height} ({dropped.height / own.height:.4f} of {a.country}'s predicted pairs)", flush=True)
    out = pp.join(dropped, on=["q", "pid"], how="left").with_columns(pl.when(pl.col("_drop").is_not_null()).then(pl.min_horizontal(pl.col("p"), pl.lit(thr - 1e-6))).otherwise(pl.col("p")).alias("p")).drop("_drop")
    if restore:
        from france_recall import restore_candidates

        kinds, pmin = restore[0][1].split("+"), float(restore[0][2])
        c = restore_candidates(out, thr, s1, pool).filter((pl.col("ctry") == a.country) & pl.col("kind").is_in(kinds) & (pl.col("p") >= pmin) & pl.col("slot_free"))
        fa = (pl.scan_parquet(sorted(str(f) for f in (P["work"] / "features" / "test").glob("part_*.parquet")))
                .select(pl.col("q").cast(pl.Int64), pl.col("pid").cast(pl.Int64), "house_eq", "addr_tset").join(c.lazy().select("q", "pid"), on=["q", "pid"], how="semi").collect())
        c = c.join(fa, on=["q", "pid"], how="left").filter((pl.col("house_eq") > 0.5) & (pl.col("addr_tset") >= 90))
        print(f"restore candidates at the S1's address by kind (p >= {pmin}, free slot): {sorted(c.group_by('kind').len().rows())}", flush=True)
        c = c.sort("p", descending=True).unique("pid", keep="first")  # one S1 per pool record
        c = c.with_columns(pl.col("p").rank("ordinal", descending=True).over(["q", "src"]).alias("_rk")).filter(pl.col("_rk") <= pl.col("cap") - pl.col("used"))
        print(f"restore {'+'.join(kinds)} p >= {pmin}: adds {c.height} pairs to {c['q'].n_unique()} {a.country} S1; by kind {sorted(c.group_by('kind').len().rows())}", flush=True)
        out = (out.join(c.select("q", "pid").with_columns(pl.lit(True).alias("_add")), on=["q", "pid"], how="left")
                  .with_columns(pl.when(pl.col("_add")).then(pl.max_horizontal(pl.col("p"), pl.lit(thr + 1e-4))).otherwise(pl.col("p")).cast(pl.Float32).alias("p")).drop("_add"))
    if a.cap:
        kept = decision.assign_exclusive(out).filter(pl.col("p") >= thr).with_columns(pl.when(pl.col("pid") < 3 * PID_BASE).then(5).otherwise(6).alias("cap_n"))
        kept = kept.with_columns(pl.col("p").rank("ordinal", descending=True).over(["q", pl.col("cap_n")]).alias("_rk"))
        over = kept.filter(pl.col("_rk") > pl.col("cap_n")).select("q", "pid").with_columns(pl.lit(True).alias("_cap"))
        print(f"cap 5 S2 / 6 S3 (all countries): drops {over.height} pairs", flush=True)
        out = out.join(over, on=["q", "pid"], how="left").with_columns(pl.when(pl.col("_cap").is_not_null()).then(pl.min_horizontal(pl.col("p"), pl.lit(thr - 1e-6))).otherwise(pl.col("p")).alias("p")).drop("_cap")
    (P["work"] / "output" / a.newname).mkdir(parents=True, exist_ok=True)
    out.write_parquet(P["work"] / "output" / a.newname / "pair_p.parquet", compression="zstd")
    emit(a.newname, P, out, {"threshold": thr, "exclusive": cfg["exclusive"]}, t0)


if __name__ == "__main__":
    main()
