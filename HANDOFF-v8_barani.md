# Handoff v8 (barani): France push on the team's s22 line

Written 27 Sep 2026 about 02:20 IST. Window closes 27 Sep 23:59 IST. Never name this file `HANDOFF.md` (the Mac disk is case-insensitive: it would be the team's `handoff.md`).

## Where things are
- Team best on the portal: `s22t2c` **0.984502** (s22 + typeswap + France threshold 0.985 + cap 5 S2 / 6 S3). The team's AWS credits are gone; they decide and upload the 5 slots of 27 Sep (`TEAMMATE_HANDOFF.md`).
- User goal: portal above 0.990. Honest math: 0.990 needs France F0.5 about 0.987 (now about 0.955) with US/India at holdout level (0.9905). Realistic: 0.986 to 0.988.
- Plan file: `~/.claude/plans/so-i-wanna-continue-radiant-toucan.md`. User decisions: team picks the uploads (we deliver validated files + a one-line note each); notebook `barani-v5` may run through Sunday; commits on local branch `v8/france` (this worktree), **no push**; no test data for training (the official README: "using only the provided training data").

## Infrastructure (our AWS, profile `barani`, account 645311222213)
- Notebook `barani-v5` (ml.g5.16xlarge), lifecycle `barani-queue`. Runner now has **two lanes**: `jobs/` (GPU) and `jobs2/` (CPU) in bucket `sagemaker-us-east-1-645311222213`. Laptop tool: `smssh-venv/bin/python aws/sm/sm.py --profile barani {publish|enqueue <job> --queue jobs|jobs2|jobs|jlog <name> --queue ...|nb start|stop|status}` run from this worktree.
- **The notebook role cannot read the team bucket** (`ml-challenge-nooglers`, AccessDenied). The laptop user can: files are copied server-side into `s3://sagemaker-us-east-1-645311222213/ber/team_work/` (9.1 GB: parquet, sample, features/test, output+models s22/s26/s22e/s27/v7, s22t2c pair_p, xenc2/model, xenc2_v7 fit set, xenc2F_v7 pair lists and scores, xenc_v7 test scores, official validator), then synced to `/home/ec2-user/SageMaker/work_t` (`BER_WORK` of all v8 jobs; our v6 `work/` is untouched).
- Rule: a GPU-lane job must never wait on a CPU-lane output (a two-lane deadlock cost one restart).

## Results so far (label-free unless stated)
- Holdout (labelled): s22 0.990540; s22e 0.990533 (paired -0.000007, CI across 0); s26 0.990529 (-0.000012, CI across 0); s27 0.990635 (paired test queued, `v8a4_s27`).
- Our rebuild of the t2c recipe with today's code differs from the team's `s22t2c` in about 590 S1 (the team later made the swap flag spaced-legal aware; our file equals their `s22v3` recipe: 79.5k France pairs dropped). Compare our variants with that in mind.
- t2c empties **3,267 France S1** (France empty share 5.0% -> 6.3%; US/India 5.8%; training singletons 5.6%); 1,008 of them lose an exact-name best pair. Slot-fit decoy share: typeswap 0.87/0.92 (S2/S3), threshold-dropped set 0.38/0.29.
- France relations vs US (slot fit is biased; compare with the US): only swaps are clearly worse than the US.

## Running (queue order)
- GPU `jobs`: `v8b_xfr1` (France-aware cross-encoder 1: e5-base warm-started from `xenc2/model`, 1 epoch on 500k replay + 282k synthetic train-only pairs: initials, spelled legal forms, glued words = positives; one common word swapped = negatives; scores 1.41M France test pairs + 619k holdout pairs), then `v8c_xfr2` (same + 100k "other tenant" negatives).
- CPU `jobs2`: `v8a2_rules` (protect 0.9/0.95 on s22; t2c on s26/s22e), `v8a3_rules2` (typeins, thr 0.99/0.995, thrp, all with protect), `v8a4_s27`, `v8a5_emptied`, then `v8d_swap1` / `v8e_swap2` / `v8f_blend` (France xs := new cross-encoder score, s22 stack rebuilt on test with tag `_pF<X>`, predict `s22F<X>` with s22's own model; US/India must be unchanged; then t2c recipe and protect on it).

## Code (branch v8/france, local commits)
`src/ber/stages/xenc_fr.py` (+ test), `src/scripts/{france_empty,xfr_report,xs_merge,emptied_samples}.py`, `france_variants.py` (+ `xfr`, `protect`, `typeins`, per-rule decoy share), `aws/queue` two-lane runner, `aws/sm` `--queue`, jobs `aws/queue/jobs/v8*.sh`.

## Next
1. Read `v8a2..a5`, `xfr_report` (does the new score separate type swaps / sure negatives from exact / initials / noise-word swaps?), and the s22F* checks (US/India unchanged).
2. Pick 5 files for the team with one-line notes; copy them server-side to the team bucket (`handoff-nooglers-20260926/runs/<name>/output/`), both TSVs, validator PASS with `--check-ids`.
3. Freeze help: reproduce_final.sh / README / methodology for whatever wins. Stop the notebook at the end.
