# v8 files for the portal, 27 Sep (barani)

Base to beat: `s22t2c` **0.984502**. Files: `s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/runs/<name>/output/{matching_results,candidate_pairs}.tsv`
(upload only `matching_results.tsv`). Every file passes the official validator **with `--check-ids`**; each keeps its base's candidate file (s22's
shortlist, or s27's for `v8_s27_tpp`). Code: branch `v8/france` (local, barani), scripts `src/scripts/france_variants.py` (rules `thrp`, `protect`),
`src/ber/stages/xenc_fr.py` (France-aware cross-encoder).

## What we measured tonight (label-free; slot-limit decoy shares, France against the US)
1. **The plain France threshold of t2c drops true exact-name copies.** It drops 63k pairs: the non-exact ones (33k) are about 50% / 43% (S2 / S3)
   decoys (US: 0%), but the exact-name ones it also drops (30k) are only about 19% decoys, below the 26% break-even. Samples of what it drops:
   `clinique pascal` / `clinique pascal sasu` at the same address, `beaux and cie` / `beaux and cie sa`, `nantes union sas` / `nantes union`.
   The team's protected threshold `thrp` (spares equal names after legal spacing, and initials) is the better rule.
2. **t2c leaves 3,267 France S1 with an empty list** (France empty share 5.0% -> 6.3%; US/India 5.8%; training singletons 5.6%). An S1 with a
   true match that gets an empty list scores 0. `protect:0.9` keeps the best threshold-dropped pair of such an S1 (p >= 0.9).
3. **France-aware cross-encoder** (e5-base warm-started from `xenc2/model`, one epoch on a replay of its fit set plus synthetic pairs built from
   TRAIN records only: initials, spelled legal forms `s a r l`, glued words as true copies; one common word swapped as sibling decoys; model 2 adds
   "another business at the same address"). On France it rejects 90% of type-word swaps (old model 19%) and accepts initials (99.96%) and spelled
   legal forms; its extra rejections are decoy-rich among one-word swaps (83-85%) but not elsewhere. It also rejected France's noise-word swaps
   (fils, groupe, services, developpement = true copies): those pairs keep the old score. Fed into s22's own stack for France only (US/India
   unchanged): the stack drops most type swaps by itself, so `typeswap` fires on 3.5-6k pairs instead of 25k.
4. Overnight bases: `s26` and `s22e` do not beat s22 on the holdout (paired -0.00001, CI across 0); `s27` does: +0.00009 [+0.000004, +0.00019].

## Files, in the order we suggest (adapt to the portal answers)
| # | File | What it is | France pairs dropped by the rules | Expected against s22t2c |
|---|---|---|---|---|
| 1 | `v8_tp985p` | s22 + typeswap + **thrp 0.985** + **protect 0.9** + cap | 49k (t2c: 79k) | +0.0003 to +0.001 |
| 2 | `v8_s22F12n_tpp` | same recipe on s22 with France's cross-encoder score = mean of our two France-aware models | 29k | reads the cross-encoder against #1 |
| 3 | `v8F1s_tpp` | same recipe, cross-encoder used only on one-word-swap pairs (the conservative use) | 29k | if #2 lost against #1 |
| 3' | `v8F12n_ts` | cross-encoder mean + typeswap only, no France threshold (bold) | 5k | if #2 won against #1 |
| 4 | `v8_s27_tpp` | recipe #1 on the s27 base (s27 cannot carry our France score) | 59k | about +0.0001 over #1 |
| 5 | final | the best of the above (the last upload may be what counts) | | |

Notes: our typeswap uses today's spaced-legal-aware swap flag (the recipe of the team's `s22v3`), so #1 against `s22t2c` also contains that change
(about 590 S1 differ). A higher plain threshold (`s22u3`, 0.995) would drop even more exact-name copies; `thrp` is the safer way to be stricter.
Honest outlook: these are worth about +0.001 to +0.002 together (0.9855 to 0.9865); 0.990 would need France at about 0.987 and nothing we
found gets there.
