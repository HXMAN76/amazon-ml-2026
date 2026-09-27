# v8 files for the portal, 27 Sep (barani) — updated about 06:05 IST

**Portal: `v8u_s27_AR` = 0.985578** (27 Sep, about 06:00; +0.001076 over `s22t2c` 0.984502): France F0.5 about 0.955 -> 0.962.
**Portal: `v8u_s22F12n_AR` = 0.984136** (about 06:15): the France cross-encoders inside the stack cost about 0.009 of France F0.5 (their extra rejections of exact-name and other pairs were true copies). Do not use the `s22F*` files; build on s27.

Base to beat: `s22t2c` **0.984502**. Files: `s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/runs/<name>/output/{matching_results,candidate_pairs}.tsv`
(upload only `matching_results.tsv`). Every file passes the official validator **with `--check-ids`** and keeps its base's candidate file. Code:
branch `v8/france` (local, barani): `src/scripts/france_variants.py` (rules `typeswap`, `thrpn`, `protect`, `restore`), `src/scripts/france_recall.py`.

## New best candidate (09:45): `v8u_s28_AR`
`s28` = s27's stack + the team's Qwen3-0.6B cross-encoder score (xs4; Apache-2.0, scored tonight on 4.6M band pairs): holdout **0.990770**, paired
**+0.000135 [+0.000047, +0.000226]** over s27. Same France recipe as `v8u_s27_AR`. Validator PASS with --check-ids. Expected portal about 0.9857.

## Current best candidate (17:40): `v8u_s28_FIN` (validated --check-ids; team folder; laptop output/)
Recipe (s28): typeswap + larger type-word list (43 words) + thrpn 0.9999 + legalx (French legal forms only) + refined namesake drop
(`fb_ns_ref`: exact core on another street, name on 11+ France S1 and p < 0.9999, or 6+ and p < 0.99; +`fb_nsnear_ref`) + protect + restore
+ re-add of coined/glued copies in [0.995, 0.9999) (`fb_coined_hi`, no French excess there) + alias protection (pool "dba/aka" part naming the S1).
All rules chosen by the per-1,000-S1 rate comparison France vs US/India, anchored on the labelled holdout. Expected about +0.001 to +0.0015 over
`v8u_s28_AR`. **Do not use the other session's `coined` restore** (below 0.995 those names are 5-14x the US rate: mostly decoys) or its buggy
legalx (dropped "Xyz Co DBA <S1 name>" true copies). Lists are built by jobs `v8xb_nsref.sh`, `v8xb_coined2.sh`; build job `v8h5_fin.sh`.

## Extended type-word rule (14:15): `v8u_s28_ARt`
`v8u_s28_AR` + `typeswap:1.01:0.6:30:300`: the type-word list grows from 30 to 43 learned words (lycee, pharmacie, danse, institut, musique, gestion,
groupement, patrimoine, soins, elementaire, ...; same slot test with slot ratio >= 0.6 over >= 30 pairs, both swapped words in >= 300 France S1 names,
noise words and abbreviations such as st / saint excluded). It drops 1,382 more France pairs beyond the current rules at a 0.86 / 0.91 decoy share
(the original rule reads 0.84 / 0.92). Expected about +0.0001. Validated (--check-ids). **On s29** (better base), the whole recipe is one line
(branch v8/france, `src/scripts/france_variants.py`):
`python src/scripts/france_variants.py s29 v8w_s29_ARt --rules "typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.995,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap`

## France cut-off test (13:45): `v8u_s28_AR9` and `v8u_s28_AR99`
Same as `v8u_s28_AR`, with France's protected cut-off (`thrpn`) raised from 0.995 to **0.9999** (`AR9`: about 25k more France pairs dropped,
2.8% of France's pairs) or **0.999** (`AR99`: about 10k more). Both pass the validator with --check-ids. Upload `v8u_s28_AR` first; then `AR9`
answers whether France's over-confidence continues above 0.995: the portal change of `AR9` against `AR` is about +0.0005 if 40% of those pairs are
wrong, 0 at 26% (break-even) and about -0.0005 if 10% are wrong. The label-free slot fit reads 0.2 to 0.55 there but is biased in the US and India,
so only the portal can tell. Keep whichever of `AR` / `AR9` scores higher as the final (`AR99` sits between them).
Measured today and not worth a slot: US/India threshold 0.80 (holdout -0.00005), a lower threshold for each S1's first pick (holdout +0.000045,
interval across 0), sub-group recalibration, namesake evidence for empty-address records (the model is already calibrated there).

## Earlier recommendation: `v8u_s27_AR` (portal 0.985578)
`s27` (team; holdout 0.99063, paired +0.00009 [+0.000004, +0.00019] over s22) + France decoding:
1. `typeswap` (team rule): drop one-word swaps into a type word (26k pairs; slot-fit decoy share 86% / 92%).
2. `thrpn 0.995` (new): drop France pairs with p < 0.995 **except** equal names after spaced legal forms, initials, and France's noise-word copies
   (a word swapped into or added from fils / groupe / services / developpement / "and associes"): 34k pairs, decoy share 58% / 62% (break-even 26%).
   The plain threshold of t2c (0.985) also dropped about 30k exact-name copies (about 19% decoys) and the noise-word copies (about 11-16% decoys).
3. `protect 0.9` (new): an S1 the threshold would leave empty keeps its best pair (549 S1). Emptied S1 score 0 when they have a true match.
4. `restore` (new): add unowned shortlisted pairs at the S1's own address that are France's noise-word copies, initials, spelled legal forms or
   glued names (7.3k pairs: noise swaps 4.6k, noise additions 1.8k, initials 0.8k). The model scores these France-only copy forms low because train
   has no such noise words; exact names are **not** restored (on the holdout such restores are 0.4% true).
Our estimate: about 0.985 to 0.986. A reading above 0.9850 confirms the direction.

## Other files
| File | What it is | Use it to read |
|---|---|---|
| `v8u_s27_A` | #1 without the restore | the restore's effect (#1 minus this) |
| `v8u_s22F12n_AR` | the same recipe on s22 with France's cross-encoder score (our two France-aware models, trained on synthetic France-style pairs from train records) inside the stack | whether the France cross-encoder helps |
| `v8u_s24_AR` | the same recipe on `s24` = s22 + the team's Qwen3-0.6B cross-encoder (scored tonight; holdout result in the log) | only if s24 beats s27 on the holdout |
| older: `v8_tp985p`, `v8_s27_tpp`, `v8_s22F12n_tpp`, `v8F1s_tpp`, `v8F12n_ts`, `v8s_s27_A/AR/AR2` | earlier stages of the same idea | superseded by the `v8u_*` files |

## What did not work (measured tonight)
Legal-form conflict + other house number as "siblings" (holdout: 99.9% true), stack blends s22+s27(+s26+s22e) (no holdout gain over s27), France
cross-encoder seeds and a restore driven by the cross-encoder score (holdout: even scores >= 0.99 among left-out pairs are only 15-32% true).
