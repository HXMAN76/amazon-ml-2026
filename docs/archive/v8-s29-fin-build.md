# Build `v8w_s29_FIN`: the s29 base with the final France recipe (context for the session that holds s29)

## Goal
One validated file, `v8w_s29_FIN`: s29's predictions for US/India (holdout 0.99088, the best base) and the final France recipe of
`v8u_s28_FIN` applied to s29. Expected portal about 0.9868–0.9873 (`v8u_s28_FIN` + about 0.0001). The team has **two slots left**; this file
is only uploaded if every check below passes.

## Code
- Branch `v8/france`, commit `62701b8` or later (`git pull`). Needed from it: `src/scripts/france_variants.py` (legalx on French legal forms
  only, alias protection, typeswap options, droplist/addlist rules), `src/scripts/france_lists.py` (MODEL argument), `src/scripts/band_kinds.py`,
  `src/scripts/namesake_street.py`, `src/scripts/word_swap.py`, `src/scripts/france_recall.py`.
- Do **not** add the `coined` post-rule or `nsaway` of `france_post.py`: `fb_ns_ref` replaces nsaway (only names on 11+ France S1 with p < 0.9999,
  or 6+ with p < 0.99, where France shows an excess over the US/India rate); plain coined names below 0.995 are 5-14x the US rate, so restoring
  them adds decoys.

## Inputs the run needs in `$BER_WORK`
`models/s29/config.json` (threshold), `output/s29/pair_p.parquet`, `parquet/test/source{1,2,3}.parquet`, `features/test/part_*.parquet`,
`official/validate_submission.py`; `$BER_DATA/test` for the validator.

## Commands (from `code/business_entity_resolution`)
```
R="typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01"
E="protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05"
python src/scripts/france_variants.py s29 v8w_s29_ALL2 --rules "$R,$E" --cap
python src/scripts/france_lists.py v8w_s29_ALL2 v8x/s29 s29
python src/scripts/france_variants.py s29 v8w_s29_FIN --rules "$R,droplist:s29/fb_ns_ref,droplist:s29/fb_nsnear_ref,$E,addlist:s29/fb_coined_hi" --cap
o=$BER_WORK/output/v8w_s29_FIN
python $BER_WORK/official/validate_submission.py --matching $o/matching_results.tsv --candidate $o/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids
```
About 15 minutes on a CPU notebook.

## Sanity checks (reference: the same recipe on s28, `v8u_s28_FIN`)
| Printed line | s28 reference | Accept if within |
|---|---|---|
| `rule typeswap:1.01: fires on` | 27,004 | ±15% |
| `rule typeswap:1.01:0.6:30:300` not fired by an earlier rule | 3,435 | ±30% |
| `rule thrpn:0.9999` not fired by an earlier rule | 41,687 | ±20% |
| `rule legalx:1.01: fires on` | 6,383 | ±30% |
| `alias protection: ... kept` | 1,353 | ±30% |
| `france_lists`: fb_ns_ref / fb_nsnear_ref / fb_coined_hi | 4,528 / 417 / 8,351 | ±25% |
| FIN `total dropped` | 80,256 (8.9% of France's pairs) | ±15% |
| `restore ... adds` / `addlist ... adds` | 6,647 / 8,347 | ±25% |
| `wrote ...`: S1 rows, with matches, mean matches | 1,732,544, 94.2%, 3.36 | exact / ±0.5 pt / ±0.05 |
| validator | `PASS` with `--check-ids` | must pass |

If any check fails, stop and report the printed lines. Do not upload.

## Deliver
Report the printed rule lines and the validator result, then put `matching_results.tsv` and `candidate_pairs.tsv` where the uploader can fetch
them (e.g. `s3://<bucket>/runs/v8w_s29_FIN/output/`). The uploader chooses between `v8w_s29_FIN` and `v8u_s28_FIN` (already validated, team
folder `runs/v8u_s28_FIN/output/`); both carry the same France rules.
