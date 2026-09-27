"""Build France decoy-rule variants of a model's output. Usage:
  python src/scripts/france_variants.py NAME NEWNAME --rules swap_exact,weak:80:0.9999:shared,legal:0.9999,thr:0.985,tiny:0.9999 [--country france]
Rules (a pair is dropped when any rule fires; only the given country; probabilities of everything else are unchanged, the candidate file is unchanged):
  swap_exact[:pmax]     one common word swapped (word_swap.py) and the S1 has a confident exact-name copy, p < pmax (default 0.9999)
  swap_all[:pmax]       one common word swapped, p < pmax
  weak:T:pmax[:shared]  name similarity (name_tset) < T, address similarity (addr_tset) >= 90, same house number, p < pmax; `shared`: the S1's address is
                        shared with another S1 (a multi-tenant building)
  legal:pmax            legal-form conflict (both sides name a different form), p < pmax
  thr:t                 p < t
  thrp:t                p < t unless the pair is protected: equal names after removing spaced legal forms, or a pool name of at most 3 letters that is a subsequence of the S1's initials
  thrx:t                p < t and the core names differ (exact-name pairs keep their probability: the slot-limit fit finds no decoys among exact-name pairs)
  typeswap:pmax[:R]     swap whose swapped-in word is a type word of the country's vocabulary (club, ecole, comite, ...): words whose rate among the S1's swap pairs does not fall when
                        the S1 already has three or more exact copies (ratio A/B >= R, default 0.75; see swap_words.py): decoys draw their new word from that vocabulary, true
                        swaps from generic suffix words (services, groupe, france); p < pmax
  protect:p0            after the rules: an S1 whose France list is empty gets back its best pair that only a threshold rule (thr, thrp, thrx) dropped, if p >= p0
  tiny:pmax             pool name of at most 3 letters that is not a subsequence of the initials of the S1's core name, p < pmax
Prints the number of pairs each rule drops and the total, then writes WORK/output/NEWNAME like reemit.py."""

import argparse
import json
import time
from functools import reduce

import polars as pl

from ber import config, decision
from ber.stages.predict import emit
from word_swap import flag, strip_spaced, tok_df

PID_BASE = 10_000_000


def is_subseq(short: str, initials: str) -> bool:
    it = iter(initials)
    return all(c in it for c in short)


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
    if any(s.startswith("typeswap") for s in rules_list):
        exs = own.filter(pl.col("core_eq") & (pl.col("p") >= 0.999)).group_by("q", "src").len().rename({"len": "k"})
        slots = s1.filter(pl.col("ctry") == a.country).select("q").join(pl.DataFrame({"src": [2, 3]}), how="cross").join(exs, on=["q", "src"], how="left").with_columns(pl.col("k").fill_null(0))
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
    fired_specs = []
    protect_p = None
    for spec in rules_list:
        k = spec.split(":")
        if k[0] == "protect":
            protect_p = float(k[1])
            continue
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
        elif k[0] == "thr":
            c = pl.col("p") < float(k[1])
        elif k[0] == "thrp":
            tiny = own.filter((pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ")).select("q", "pid", "a_core", "b_core")
            ok = [(q, pid) for q, pid, ac, bc in tiny.iter_rows() if is_subseq(bc, "".join(t_[0] for t_ in ac.split()))]
            ok_df = pl.DataFrame({"q": [x[0] for x in ok], "pid": [x[1] for x in ok]}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_ini"))
            own = own.join(ok_df, on=["q", "pid"], how="left").with_columns(pl.col("_ini").fill_null(False).alias("ini_ok"))
            own = own.with_columns((strip_spaced(pl.col("a_core")) == strip_spaced(pl.col("b_core"))).alias("eq_norm"))
            c = (pl.col("p") < float(k[1])) & ~pl.col("eq_norm") & ~pl.col("ini_ok")
        elif k[0] == "thrx":
            c = (pl.col("p") < float(k[1])) & ~pl.col("core_eq")
        elif k[0] == "typeswap":
            rmin = float(k[2]) if len(k) > 2 else 0.75
            words = rr_all.filter((pl.col("ratio") >= rmin) & (pl.col("nA") + pl.col("nB") >= 100))["xb"].to_list()
            own = own.with_columns(pl.col("b_core").str.split(" ").list.unique().alias("_tb"), pl.col("a_core").str.split(" ").list.unique().alias("_ta"))
            own = own.with_columns(pl.col("_tb").list.set_difference(pl.col("_ta")).list.first().alias("_xb"))
            c = pl.col("swap") & pl.col("_xb").is_in(words) & (pl.col("p") < float(k[1]))
        elif k[0] == "tiny":
            tiny = own.filter((pl.col("b_core").str.len_chars() <= 3) & ~pl.col("b_core").str.contains(" ") & (pl.col("p") < float(k[1]))).select("q", "pid", "a_core", "b_core")
            bad = [(q, pid) for q, pid, ac, bc in tiny.iter_rows() if not is_subseq(bc, "".join(t[0] for t in ac.split()))]
            bad_df = pl.DataFrame({"q": [x[0] for x in bad], "pid": [x[1] for x in bad]}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_tiny"))
            own = own.join(bad_df, on=["q", "pid"], how="left").with_columns(pl.col("_tiny").fill_null(False).alias("tiny_bad"))
            c = pl.col("tiny_bad")
        else:
            raise SystemExit(f"unknown rule {spec}")
        n = own.filter(c).height
        fired.append(c)
        fired_specs.append(k[0])
        print(f"rule {spec}: fires on {n} of {own.height} predicted {a.country} pairs ({n / own.height:.4f})", flush=True)
    if fired:
        any_c = fired[0]
        for c in fired[1:]:
            any_c = any_c | c
        dropped = own.filter(any_c).select("q", "pid").with_columns(pl.lit(True).alias("_drop"))
    else:
        dropped = pl.DataFrame({"q": [], "pid": []}, schema={"q": pl.Int64, "pid": pl.Int64}).with_columns(pl.lit(True).alias("_drop"))
    if protect_p is not None and fired:
        thr_c = [c for c, sp in zip(fired, fired_specs) if sp.startswith("thr")]
        oth_c = [c for c, sp in zip(fired, fired_specs) if not sp.startswith("thr")]
        if thr_c:
            or_all = lambda cs: reduce(lambda x, y: x | y, cs)
            flags = own.with_columns(or_all(fired).alias("_dr"), or_all(thr_c).alias("_thr"), (or_all(oth_c) if oth_c else pl.lit(False)).alias("_oth"))
            kept_n = flags.filter(~pl.col("_dr")).group_by("q").len().rename({"len": "kept"})
            cand = flags.filter(pl.col("_thr") & ~pl.col("_oth") & (pl.col("p") >= protect_p)).join(kept_n, on="q", how="left").filter(pl.col("kept").is_null())
            best = cand.sort("p", descending=True).group_by("q").first().select("q", "pid").with_columns(pl.lit(True).alias("_res"))
            print(f"protect {protect_p}: restores {best.height} pairs of S1 whose list is empty after the rules", flush=True)
            dropped = dropped.join(best, on=["q", "pid"], how="left").filter(pl.col("_res").is_null()).drop("_res")
    print(f"total dropped {dropped.height} ({dropped.height / own.height:.4f} of {a.country}'s predicted pairs)", flush=True)
    out = pp.join(dropped, on=["q", "pid"], how="left").with_columns(pl.when(pl.col("_drop").is_not_null()).then(pl.min_horizontal(pl.col("p"), pl.lit(thr - 1e-6))).otherwise(pl.col("p")).alias("p")).drop("_drop")
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
