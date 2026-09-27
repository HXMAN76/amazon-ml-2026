# Project Handoff

v7 France line (barani). Written **26 Sep 2026, about 18:55 IST**. Challenge window closes **Sun 27 Sep 2026 23:59 IST**.

> **UPDATE, 26 Sep about 19:10 IST: branch `v7/france-vocab` and its worktree were deleted at the user's request.**
> - The branch was removed locally and on GitHub. The user did not want it pushed.
> - This file is now an **untracked local copy** in the main checkout (`/Users/barani200/amazon-ml-2026`, branch `sai`). Nothing from v7 exists in git.
> - Everything below that mentions the branch, the worktree, commits or pushes is **superseded**. Before creating any branch, and before any commit or push, ask the user which branch to use and whether to push.
> - The analysis, decisions and plan below still hold.

> **File name.** This file is not called `HANDOFF.md`: the Mac disk is case-insensitive, so `HANDOFF.md` *is* the team's tracked
> `handoff.md` (checked: `ls HANDOFF.md` in the worktree opens the team's 34 KB file). Our handoffs are `HANDOFF-v<N>_barani.md`.
> The previous line is documented in `HANDOFF-v6_barani.md`, same folder.

---

## 1. Session Overview

- **Larger objective:** Amazon ML Challenge 2026, Business Entity Resolution, team Nooglers.
  - Push the leaderboard (portal) score as high as possible. The user asks for 0.99+.
  - Per S1 record, list its S2/S3 matches. The metric is macro F0.5 per S1.
- **Starting point (this session):**
  - Our own line v6 (`bs_*` models on our notebook `barani-v5`) peaked at **`bs_w3`**: locked holdout 0.98920, test-weighted 0.98736, validator PASS, never uploaded.
  - The team's line (branch `sai`) is ahead:
    - `s17`: holdout 0.99025, portal 0.981;
    - `s22`: holdout 0.99054;
    - `s22sx`: portal **0.982477**, the best so far (the team's France word-swap rule on `s22`).
- **Work in this session:**
  1. Built and measured the whole v6 line (see `HANDOFF-v6_barani.md`).
  2. Stopped the v6 training at the user's request.
  3. Diagnosed the portal gap label-free: it is **France**.
  4. Found the data generator's **noise vocabulary** for true copies, per country (section 5).
  5. Wrote plan v2 (`~/.claude/plans/i-want-to-setup-fuzzy-prism.md`).
  6. Read the team's newest France work (sai `24fb7c7`, `d8ae1f0`, `6335c20`).
  7. Created the worktree and branch `v7/france-vocab`.
  8. Stopped a stale local log watcher.
- **Current state:**
  - No v7 code has been written yet.
  - Branch `v7/france-vocab` is at `bf0ed33` (same commit as `v6/strong-parts`). Apart from this file, the worktree is clean.
  - Notebook `barani-v5` is **Stopped**.
- **Where it stopped:** just before copying the team's France scripts into the worktree and adding our `vswap` rule. The user ran `/checkpoint` at that point.

## 2. Current Objective

- **Goal:** raise the team's portal score above `s22sx` 0.982477.
  - The remaining gap is almost entirely France: France F0.5 is about 0.934, against about 0.99 for US and India.
  - Team calibration: **portal delta = 0.149 x France F0.5 delta**.
  - Per 1% of France's predicted pairs dropped (about 9k pairs), France F0.5 moves -0.0022 if the pairs were true and +0.0063 if they were wrong. So **dropping a category pays only if more than 26% of it is wrong**.
- **Three tracks, in priority order (plan v2):**
  1. **`vswap` rule:** the team's `swap_exact` rule, except that swaps into France's noise words are kept.
     - Why: `s22sx` dropped 36.7k French pairs; the portal says about 47% were wrong (about 17k), so about 19k *true* pairs were dropped too.
     - Our vocabulary finding says which half is true: pool names that swap in {fils, groupe, services, developpement} are real noisy copies; swaps into type words (club, ecole, amicale, comite, france, centre, ...) are look-alike siblings.
     - Expected: portal +0.0007 to +0.001 over `s22sx` (0.983 to 0.9835).
  2. **Open-source LLM judge**, zero-shot, on France's suspicious predicted pairs.
     - Target: mainly the team's "weak name + strong address" bucket (excess about 42k pairs: different businesses in multi-tenant buildings).
     - Model: `Qwen3-4B-Instruct-2507` (Apache-2.0, 4B).
     - Expected: +0.001 to +0.0026 more, if it separates well.
  3. **France-aware retrain of our cross-encoder**, with synthetic hard negatives built from **training** records only: another business moved to the S1's address, and type-word siblings at the same address. Restack, then a paired holdout check.
- **Honest expectation:** about 0.984 to 0.987 at best. 0.99 would need France F0.5 about 0.99, i.e. every wrong French pair removed and no true pair lost. Never promise it.
- **Acceptance for any new file:**
  - holdout (US/India, labelled) not lower on the paired bootstrap;
  - `check_submission.py` and the official validator PASS;
  - the candidate file is the parent model's (matches must stay a subset of it).
  - The portal decides; uploads are human-only.
- **Constraints:** see section 13 (no test data for training, commit style, security, licences, compute).

## 3. Project / System Context

**Pipeline** (`code/business_entity_resolution/`, package `src/ber`, stages in `src/ber/stages`, targets in `Makefile`, parameters in `configs/params.yaml`):

```
prepare -> token blocking + pruner -> dense channels (multilingual-e5-small: dense, dense_all, dense_all2)
-> first-stage XGBoost (p1) -> shortlist (best 10 per S1, p1 >= 0.005)
-> cross-encoder xenc (ours: gte-multilingual-reranker-base with sibling context "[SIB]"; team: e5 / Qwen3-0.6B)
-> stack XGBoost (xs, decoy, consensus features) -> one pool record -> one S1 (assign_exclusive), threshold, cap 5 S2 + 6 S3
-> output/<name>/matching_results.tsv + candidate_pairs.tsv
```

**WORK layout used by the France scripts** (`ber.config.paths()`: `P["work"]`, `P["parquet"]`):
- `P["parquet"]/{train,test}/source{1,2,3}.parquet`: columns `rid`, `core1` (normalised core name), `addr`, `ctry` (lower-case country, e.g. `france`), `legal`, ...
- `P["parquet"]/train/labels.parquet`: `s1_rid`, ...
- `P["work"]/features/{train,test}/part_*.parquet`: `q`, `pid`, `name_tset`, `addr_tset`, `house_eq`, `legal_conflict`, ...
- `P["work"]/models/<name>/`:
  - `config.json` (`threshold`, `exclusive`), `holdout.json` (`stack_threshold`);
  - `holdout_pred.parquet` (`q`, `pid`, `p`, `label`);
  - `oof_tune.parquet`, `final_pred.parquet`, `xgb.json`.
- `P["work"]/output/<name>/`: `pair_p.parquet` (`q`, `pid`, `p`), `matching_results.tsv`, `candidate_pairs.tsv`.
- Ids: `q` = S1 `rid`; `pid` = pool `rid` + s x 10,000,000 (s = 2 or 3).

**AWS (profile `barani`, region us-east-1):**
- **Notebook `barani-v5`:**
  - ml.g5.16xlarge (one A10G 24 GB, 64 vCPU, 256 GB RAM), 200 GB volume;
  - lifecycle `barani-queue`, which auto-stops after 1 h with no job;
  - service quotas: 16xlarge 1, 8xlarge 1, 4xlarge 2.
- **Job queue** (bucket name in `aws/queue/jobs/_header.sh`):
  - code at `ber/code`, tools at `ber/tools`;
  - jobs move `jobs/pending/*.sh` -> `jobs/live/<name>.log` -> `jobs/done/<name>.log` (last line `exit=<code>`).
  - `aws/queue/jobrunner.sh` runs every job still in `pending/` **as soon as the notebook boots**, alphabetically, one at a time. A job interrupted by a stop is not re-run: it has already left `pending/`.
- **Checkpoints** of each stage: `s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/bs/NN-<stage>-<time>/`.
  - The latest is `05-bs_errors-20260926-0828`.
  - `bs_w3` is in `02-bs_final_stack-20260926-0821/`: models + `candidate_pairs.tsv` + `matching_results.tsv`, **no `pair_p.parquet`**.
- **Laptop tool:** run from the worktree as `/Users/barani200/amazon-ml-2026/smssh-venv/bin/python aws/sm/sm.py --profile barani <cmd>`.
  The venv lives only in the main checkout. `sm.py` resolves the repo from its own path, so `publish` uploads the checkout it is run from.
  Commands:
  - `nb start|stop|status`;
  - `publish` (code to the queue bucket);
  - `enqueue aws/queue/jobs/<job>.sh --name <name>`;
  - `jobs`;
  - `jlog <name> [--no-follow]`;
  - `queue-setup` (notebook must be stopped).
  - `aws/sm/README.md` also documents the SageMaker-training-job mode (`check`, `submit`, `logs`, `pull`, `checkpoints`).
- The team's notebooks and files are separate. **`s22`'s `pair_p` is not in the shared bucket** (the `checkpoints/` prefix holds only `barani/`).

## 4. Current Implementation State

- **v7 code: none yet.** The needed team scripts exist only on `origin/sai` under `code/business_entity_resolution/src/scripts/`:
  - `word_swap.py`: `tok_df`, `flag`; `DF_MIN = 100`. A swap means exactly one core-name token differs on each side, both with df >= 100 and length > 2. `flag` currently **drops** the swapped tokens `xa`/`xb`.
  - `france_variants.py`: rule framework on a model's test `pair_p`. Rules: `swap_exact`, `swap_all`, `weak:T:pmax[:shared]`, `legal:pmax`, `thr:t`, `tiny:pmax`. It writes `WORK/output/NEWNAME` through `predict.emit`.
  - `rule_on_holdout.py`: the same swap rules on `models/<name>/holdout_pred.parquet` with labels; prints the F0.5 delta.
  - Also on sai: `swap_mixture.py`, `excess_mass.py`, `decode_rules.py`.
- **Compatibility, checked by grep** against v6's `ber`, which is what v7 has:
  - `decision.assign_exclusive` (`src/ber/decision.py:14`);
  - `predict.emit(name, P, df, cfg, t0)` (`src/ber/stages/predict.py:52`);
  - `split.holdout_q` (`src/ber/split.py:11`);
  - feature columns `house_eq`, `legal_conflict` (`stages/pairs.py:106,119`), `name_tset` and `addr_tset` (`stages/stack.py:38`).
  - Not yet run end to end on our WORK.
- **`bs_w3` files:**
  - S3 checkpoint as above;
  - local `/Users/barani200/amazon-ml-2026/output/v6_bs_w3/{matching_results.tsv, candidate_pairs.tsv}` (untracked);
  - `pair_p.parquet` is expected on the notebook disk at `/home/ec2-user/SageMaker/work/output/bs_w3/pair_p.parquet`. **Unverified.**
- **What works:**
  - the v6 pipeline and all its jobs;
  - `bs_w3` passes the official validator.
- **Incomplete:** all three v7 tracks.
- **Broken:** nothing known. See section 8 for the unverified pieces.

## 5. Detailed Session Changes (chronological)

1. **v6 line** (branch `v6/strong-parts`, pushed; details and per-job results in `HANDOFF-v6_barani.md`):
   - `bs_s6` 0.98299 (reproduces the team's `s6`);
   - first stage `bs_w1` 0.97926;
   - `bs_w2` 0.98699;
   - `bs_final` 0.98718;
   - **`bs_w3` 0.98920** (cross-encoder v2, stack on 650k S1).
   - Commits include `83cb63a fix(setmodel)`, `d55a84a refac(thin)`, `d7a956a feat(stack)`, `5ebfb60 feat(scripts)`, `af69073`/`6eb73dd feat(xenc)`, `289ea09 feat(stack)`, `0af7d3c feat(make)`, `bf0ed33 fix(aws)` and several `docs(handoff)`.
2. **`f2`/`f2b_xenc3`** (cross-encoder v3: 900k S1, 2 epochs): started, then all training was stopped at the user's request.
   - No checkpoint after 26 Sep 08:28. Its uncommitted code is in `stash@{0}` ("barani v6": `Makefile`, `configs/params.yaml`, `stages/stack.py` (`--seeds`), `stages/xenc.py` (`xenc_prev` compare)).
   - The job file is untracked at `/Users/barani200/amazon-ml-2026/aws/queue/jobs/f2_xenc3.sh`. **Untested; do not rely on it.**
3. **France diagnosis (label-free, test TSVs + `bs_w3` output, scratch scripts):**
   - France has more candidates per S1 (5.32 vs 4.53 in the US and India) and claims more pool records (62.1% vs 59.1% US, 58.1% India).
   - `bs_w3` matches per S1: France 3.436, US 3.401 (team `s17` France 3.53).
   - The extra test records are look-alikes, not orphans.
   - French look-alikes swap or inject real words and move the house number or street. Addresses are crowded: 46% of accepted French pairs sit at addresses with 5+ pool records.
4. **Noise-vocabulary finding.**
   - **Method:** "lone swaps" at the S1's exact address key: house number + street words, abbreviations expanded, region words removed (`tenant_check.key`). A lone swap is a pool record that differs by one swapped word from the S1 and is the only record with that core name at that key. The rate of a word = uses as the pool-side replacement word per 1,000 S1 core names containing it.
     - The analysis ran inline (not saved). It used the sai `word_swap.flag`, `ber.text.parse_name(...).core1`, and a 150k-S1 sample per country.
     - Session log: `~/.claude/projects/-Users-barani200-amazon-ml-2026/a171f37e-f6b3-422e-b462-bba17956ca84.jsonl`; search "per_1000_name_uses".
   - **Train (labelled):** replacement words of true lone swaps are almost only center 3,010 (2,990 true), services 2,255 (2,229), service 1,103 (1,099), centre 37 (36). Every other word has 21 cases or fewer and is mostly false.
   - **Train rates are low:** US center 63.6, services 218.0, service 172.5 per 1,000; India center 251.8, services 25.5, service 433.1. In train, noise words stand out by **count**, not by rate, because train look-alikes are letter edits, not word swaps.
   - **Test France rates:**
     - noise words: fils 967.6, groupe 1,327.2, services 2,191.9, developpement 3,136.2;
     - type words sit flat at 62 to 117: france 110.0, club 84.2, ecole 76.6, amicale 82.1, comite 78.2, sportive 103.0, amis 110.6, centre 62.2, parents 116.0, union 63.6, college 73.4, primaire 75.3, societe 117.0, fetes 114.6.
     - **"centre" is a type word in France**, unlike in the US and India.
   - **`bs_w3` acceptance of French lone swaps** (150k France S1 sample, 35,550 lone swaps):
     - noise words: fils 0.921, groupe 0.685, services 0.982, developpement **0.065**;
     - type words: france 0.976, club 0.664, amicale 0.773, ecole 0.689, comite 0.803, sportive 0.578, amis 0.783, centre 0.877, parents 0.525, union 0.586.
     - US lone swaps: center 0.999, services 0.998.
     - Records with 2+ twins of their own: 0.5% accepted.
   - **Estimates:**
     - Accepted noise-word swaps about 9.1k vs accepted type-word swaps of at least 8.5k (ten largest type words alone; 11.7k more lone swaps in smaller words were not tallied) per 150k France S1.
     - Extrapolated to all French S1 (earlier estimate): about 17k likely false positives, and about 4k missed "developpement" copies.
     - Matches the portal's reading of `s22sx` in direction (about 47% wrong, range 35 to 58%). It is not an exact test: lone swaps are not the `swap_exact` set.
5. **Portal arithmetic** (earlier):
   - A country-only file scores w x F + (1 - w) x s.
   - Probes `s17pf` 0.187 and `s17pu` 0.453 imply the public subset over-weights the US (about 0.424); the population mix is infeasible.
   - France F0.5 about 0.93 before `s22sx`, about 0.934 after (team).
6. **Team update read this session:**
   - `research.md` sections 25.1 and 25.2; `EXPERIMENTS.md` section 10.1.
   - `s22sx` 0.982477.
   - Legal-form conflicts: 6.8% of French swap pairs, about 3.9k, are certainly decoys.
   - Counting argument: excess swaps about 16k.
   - Weak-name excess about 42k.
   - Variants for the five slots of 27 Sep:
     - `s22a` weak name < 60, 17.8k extra drops;
     - `s22b` < 80, 21.9k;
     - `s22c` legal conflict, 13.3k;
     - `s22d` France threshold 0.985, 47.6k;
     - `s22e` tiny names.
   - The team's own estimate: +0.001 to +0.003 (0.9835 to 0.9855).
7. **Plan v2** written: `~/.claude/plans/i-want-to-setup-fuzzy-prism.md`.
8. **Worktree:** `git worktree add ../amazon-ml-2026-v7 -b v7/france-vocab v6/strong-parts`, with result `bf0ed33`.
9. **Stale watcher** (`watch.sh f2_xenc3`, a background shell) stopped.
10. **This file** added.

## 6. Technical Decisions and Reasoning

- **v7 is based on `v6/strong-parts`, not `origin/sai`.**
  - Our notebook WORK (`bs_w3` features, `pair_p`, models) was built by v6 code, and our queue tooling (`aws/sm`, `aws/queue`) and our cross-encoder (`stages/xenc.py`, gte + `[SIB]`) live there. The team's `xenc.py` on sai is a different file with the same path.
  - Only the team's rule scripts are copied in (`git checkout origin/sai -- <paths>`).
  - Final.
- **Hard-code France's noise vocabulary** in the rule: `{"france": {"fils", "groupe", "services", "developpement"}}`, with the evidence in the docstring.
  - Don't learn it with "rate >= 500": that threshold works in France only, and the train check in plan v2 cannot pass (train noise words have rates 25 to 433).
  - US/India vocabulary for the holdout check: {center, services, service, centre} (labelled).
  - This supersedes the plan's "learned, rate >= 500, train must give {center, services, service}" wording.
- **`vswap` = `swap_exact` minus pairs whose pool-side swapped word (`xb`) is in the country's noise vocabulary.**
  - Why: it is directly comparable with `s22sx` on the portal (one slot; the difference reads the kept pairs).
  - The pool side is used because S2/S3 records are the noisy copies.
  - Needs `word_swap.flag` to keep `xa`/`xb` (a one-line change: drop only `ta`, `tb`, `da`, `db`).
- **No "restore" at first.** Accepting "developpement" swaps the model rejected is worth about +0.00015 on the portal, and the rule's re-assignment adds complexity. Add it only if `vswap` wins on the portal.
- **LLM judge model: `Qwen3-4B-Instruct-2507`** (Apache-2.0, 4.0B), not Qwen2.5-7B-Instruct (7.6B).
  - Why: if the 8B limit counts all models together, 7.6B + three e5-small (~0.12B each) + gte reranker (~0.3B) exceeds it.
  - Excluded: Qwen2.5-3B (research licence); Llama-based models such as Jellyfish-8B (not MIT/Apache).
  - Verify the licence on the model card before use.
- **Zero-shot judge, no fine-tuning.** France is absent from training (train has only US and India). Fine-tuning on train teaches train's look-alike style (letter edits); the value here is the model's general knowledge that a club and a college at the same address are different businesses.
  - Prompt idea: "Same business? A: name | address. B: name | address. Answer Yes or No."
  - Score = P(Yes) from the next-token logits.
- **One GPU is enough for the judge:** about 200k pairs at roughly 30 to 60 pairs/s, so 1 to 2 h on one A10G.
  - The user asked for two notebooks (16x plus 8x). Use the 8xlarge when the judge and the retrain run in parallel, or when judging all ~900k French pairs.
  - Two notebooks need **per-notebook queues**. Today the runner polls one shared `jobs/pending/`, and two runners would race for the same job.
- **Retrain risk:** synthetic "type-word sibling" negatives could teach the model to reject France's real noise words (fils, groupe, developpement are never positives in train).
  - So: build negatives only from train vocabulary, never inject France's words as positives (that would be test-derived), and check France's noise-word swaps stay accepted.
  - The existing synthetic look-alike generator is `src/ber/decoys.py` (v6), which the cross-encoder already uses. Extend it.
- **Test data:** used only for label-free statistics and decode-time rules, never for training (user decision).

## 7. Failed Approaches (whole session)

- **Test-density thinning:** the density check showed the extra test records are decoys (orphan share 0). Code removed (`d55a84a`).
- **Set-model blend:** +0.000005 on top of `bs_w3`, so no gain; the output is the stack's.
- **France hypotheses rejected:**
  - "alias excess" (artifact: glued domains and initials counted as aliases);
  - neighbour records;
  - generic names (French names are not more generic);
  - address incompleteness (same as the US).
- **Team's swap rules on the holdout** cost 0.00425 / 0.00775 (in train, swaps are 99.7% true). France differs because of the vocabulary.
- **Team's `s20`** (e5-large cross-encoder): no gain. Bigger encoders are not the lever.
- **Cross-encoder v3 (`f2`)**: never finished (stopped). Its effect is unknown.
- **Infrastructure:**
  - g5.4xlarge capacity failure, so we moved to the 16xlarge;
  - `a1_base` failed on a missing `WORK/dense` (fixed in `dense.py`);
  - `s3 sync` skipped same-size edits, so `f2` ran old parameters (fixed with `--exact-timestamps`, `bf0ed33`).
- **Git:**
  - zsh `$S` word-splitting mis-staged blocks, so commits were redone;
  - the user then asked to undo all commits and recommit with one-line messages.
- **Advice error:** I recommended submitting `bs_w3`, then withdrew it after seeing the team's `s17` 0.981. The user will not upload `bs_w3` as is.

## 8. Current Bugs / Unresolved Issues

1. **Notebook queue state unchecked.**
   - Expected: `jobs/pending/` empty before starting `barani-v5`.
   - Actual: unknown. The check (`aws s3 ls` of the queue) was declined during the checkpoint.
   - Risk: any leftover `*.sh` in `pending/` (for example an old cross-encoder job) starts automatically on boot and can overwrite `work/xenc`.
   - Next: `sm.py jobs` before `sm.py nb start`.
2. **`bs_w3` `pair_p.parquet` location unverified.** Expected at `work/output/bs_w3/pair_p.parquet` on the notebook volume (not in the S3 checkpoint).
   - If it is missing, regenerate it with the predict step of the `bs_final_stack` target: `python -m ber.stages.stack predict --name bs_w3 --tag w3`.
     `bs_w3` was built with `STK=bs_w3 STK_TAG=w3`; see `aws/queue/jobs/f1_xenc2.sh` and `Makefile` lines 114 to 117.
3. **The team's scripts have not run on our WORK.** Columns are checked by grep only; the first job may reveal naming differences.
4. **`s22` not accessible to us.** Our rules can be measured only on `bs_w3`. Applying them to `s22` needs whoever has `s22`'s WORK to run one command (section 14).
5. **Portal slots:** five per day, shared by the team. The team has planned all five for 27 Sep (`s22a`, `s22c`, `s22d`, a combination, the final). Any slot for `vswap` is the user's or the team's decision.
6. **Mac quirk:** torch + XGBoost in one process segfault unless `OMP_NUM_THREADS=1` (tests, local smoke).
7. **Untracked leftovers** in the main checkout: `src/tests/test_v5.py`, `stages/{keys,expand,measure,v2_eval}.py`, `ber/channels.py`, older docs. Not part of v6 or v7; leave them.

## 9. Git / Test / Build State

- **Main checkout** `/Users/barani200/amazon-ml-2026`:
  - branch `sai` at `6335c20` = `origin/sai`; the team's branch: **never commit or push here**;
  - many untracked files (`aws/queue/jobs/{f2_xenc3,run_stages}.sh`, `aws/sm/notebook/`, `output/`, `v5.md`, `ARCHITECTURE_v4/v5.md`, `archive/`, ...).
- **Worktree** `/Users/barani200/amazon-ml-2026-v7`: branch `v7/france-vocab` at `bf0ed33`. Before this file it had no changes and no upstream.
- **`v6/strong-parts`** = `origin/v6/strong-parts` = `bf0ed33`.
- **Other local branches:** `0a`, `er`, `main`, `rm`, `v1/record-centric`, `v2/decision-layer`, `v5/multichannel-consensus`.
- **Stashes:**
  - `stash@{0}` "barani v6" (untested v6 extras, section 5.2);
  - `stash@{1}` "barani v5";
  - `stash@{2}` "er WIP";
  - `stash@{3}` "barani v0".
- **Tests:** the last known run is `pytest` 34 pass on v6 (Mac, `OMP_NUM_THREADS=1`). Nothing was run for v7.
- **No build, lint or type-check step exists** beyond pytest and the validators.

## 10. Important Files and Code Locations

| Path | Why it matters |
|---|---|
| `~/.claude/plans/i-want-to-setup-fuzzy-prism.md` | Plan v2 (tracks, expected scores). Section 6 above corrects its vocabulary-learner wording |
| `HANDOFF-v6_barani.md` (worktree) | v6 architecture, per-job results, AWS details |
| `origin/sai:code/business_entity_resolution/src/scripts/{word_swap,france_variants,rule_on_holdout}.py` | Team rule framework to copy and extend (`flag`, `tok_df`, rule loop, `emit`) |
| `origin/sai:EXPERIMENTS.md` section 10.1, `origin/sai:research.md` sections 25.1-25.2 | Team's France calibration, `s22sx` reading, planned variants |
| `code/business_entity_resolution/src/ber/decision.py` | `assign_exclusive`, `macro_f05`, `cap_per_source` |
| `code/business_entity_resolution/src/ber/stages/predict.py` | `emit(name, P, df, cfg, t0)`: writes both output files |
| `code/business_entity_resolution/src/ber/stages/xenc.py`, `src/ber/decoys.py` | Our cross-encoder and its synthetic look-alikes (track 3) |
| `code/business_entity_resolution/Makefile` | `bs_*` targets; job variables `STK`, `STK_TAG`, `SETM`, `FINAL` |
| `aws/queue/{jobrunner.sh,bootstrap.sh,onstart.sh}`, `aws/queue/jobs/_header.sh`, `aws/queue/jobs/f1_xenc2.sh` | Notebook queue and a job template |
| `aws/sm/sm.py`, `aws/sm/README.md` | Laptop commands (section 3) |
| `/Users/barani200/amazon-ml-2026/output/v6_bs_w3/` | Our best test files (untracked) |
| Scratchpad `/private/tmp/claude-501/-Users-barani200-amazon-ml-2026/a171f37e-f6b3-422e-b462-bba17956ca84/scratchpad/` | **Ephemeral:** `data/{train,test}/*.tsv` (full dataset), `venv-v5` (py3.12: polars, xgboost, torch, transformers), France analysis scripts (`france_probe.py`, `diff_types.py`, `tenant_check.py` with `key()`, `profile_match.py`, `addr_types.py`), `watch.sh`, `stage_blocks.py` |

## 11. Configuration

- **Job environment** (set in `_header.sh` / job files):
  - `BER_DATA`, `BER_WORK`, `PYTHONPATH` (dataset, WORK and code on the notebook);
  - `BER_ML_ROOT`, `BER_CODE`, `BER_SKIP_INSTALL`;
  - `BER_STAGES` (Makefile targets to run);
  - `BER_CKPT_S3` / `BER_CKPT_FALLBACK` (checkpoint destinations), `BER_WORK_S3` (WORK mirror), `BER_JOB_NAME`;
  - Makefile variables `STK`, `STK_TAG`, `SETM`, `FINAL`, `PREV`;
  - `STAGES` (per-job default list).
- **Mac:** `OMP_NUM_THREADS=1` for tests; use the scratchpad `venv-v5` Python for local analysis.
- **AWS:** profile `barani`, us-east-1; notebook `barani-v5`; lifecycle `barani-queue`; conda env `ber` under `/home/ec2-user/SageMaker/envs/ber`.
- **Credentials:** the AWS login expires. The user runs `aws login --profile barani` in their own terminal.
  - Git pushes need the user's SSH key unlocked: the user runs `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` themselves.
  - No secrets belong in this file, in chat or in git.

## 12. Testing and Verification

- **Done this session:**
  - grep-level compatibility of the team scripts with v6 `ber`;
  - notebook status via `describe-notebook-instance` (Stopped);
  - S3 listing of our checkpoints (no `s22` files in the shared bucket; no `bs_w3` `pair_p` in the checkpoint).
- **Not done:** any v7 run; the queue check; `pair_p` presence.
- **To do per track:**
  - **`vswap`:**
    - small `assert` test: a type-word swap is dropped, a noise-word swap is kept, the other country is unchanged;
    - `rule_on_holdout.py bs_w3` with `vswap` and the US/India vocabulary: F0.5 must not drop (the team's plain swap rules lost 0.004 to 0.008);
    - `france_variants.py bs_w3 bs_w3sx --rules swap_exact` and `... bs_w3v --rules vswap`, reading the printed counts: `vswap` should drop clearly fewer pairs than `swap_exact`, with the difference made of fils/groupe/services/developpement swaps;
    - `check_submission.py` + official validator (with `--check-ids` on the notebook).
  - **Judge:**
    - calibrate on labelled holdout pairs: false "No" rate on true weak-name/swap pairs, detection of false positives;
    - label-free sanity on France: noise-word swaps mostly "Yes", type-word swaps mostly "No";
    - only then a `judge:t` rule, with t where the estimated wrong share exceeds 26%.
  - **Retrain:** paired bootstrap against `bs_w3` on the locked holdout; France label-free acceptance of type-word vs noise-word swaps.

## 13. Conversation-Only Context

- **Target:** the user wants the overall score at 0.99+ and asked to keep researching improvements (fine-tuning techniques, open-source models or architectures) while GPU jobs run. Be honest that 0.99 is unlikely.
- **Instructions and preferences:**
  - Latest instruction: "once you lay the plan.... start implementing the plan in the feature branch, and start training in the gpu", which is an explicit go for plan v2.
  - Run on GPU notebooks, not the laptop. The user suggested one 16xlarge plus one 8xlarge.
  - Work independently on our own notebook and files; don't wait on teammates' files or compute.
  - The user does not want to upload `bs_w3` as is. Uploads are done by humans. The user reported the team's uploaded file (holdout about 0.99) scored 0.982 on the portal.
  - Open-source models must be MIT or Apache-2.0 and at most 8B parameters. No external data lookups.
  - **No test data for training** (no pseudo-labels, no test adaptation); label-free test statistics for analysis and decode rules are fine.
- **Commits:**
  - one commit per change;
  - one line: `type(scope): what changed`, types feat/fix/refac/docs, tests as `feat(tests)`;
  - **no body, no Co-Authored-By, no Claude-Session trailer, no emoji**. This overrides the tool's default attribution;
  - push to the feature branch only (`v7/france-vocab`), never to `main` or `sai`;
  - commit only useful changes and needed fixes; no failed or redundant ones;
  - keep a handoff file updated and pushed.
- **Security and cost:**
  - never ask for the SSH passphrase; never paste credentials, presigned URLs or Jupyter links;
  - deleting a teammate's AWS resource needs explicit confirmation;
  - stop notebooks when idle; earlier budget about 40 box-hours; only the barani notebook(s).
- **Communication:** the user prefers simple explanations (analogies helped) and action over long plans.

## 14. Immediate Next Action

In `/Users/barani200/amazon-ml-2026-v7`:
1. Make sure this handoff is committed and pushed:
   - `git log -1 -- HANDOFF-v7_barani.md`;
   - if the branch has no upstream yet, `git push -u origin v7/france-vocab`. This needs the user's SSH key unlocked.
2. Bring in the team scripts and commit, e.g. `feat(scripts): bring the team's word-swap and France variant rules from sai`:
   `git checkout origin/sai -- code/business_entity_resolution/src/scripts/word_swap.py code/business_entity_resolution/src/scripts/france_variants.py code/business_entity_resolution/src/scripts/rule_on_holdout.py`
3. Make the rule change:
   - in `word_swap.flag`, keep `xa`/`xb`;
   - in `france_variants.py`, add `NOISE = {"france": {"fils", "groupe", "services", "developpement"}}` and the rule `vswap[:pmax]`: `swap & n_exact >= 1 & p < pmax & ~xb.is_in(NOISE[country])`;
   - add `vswap` to `rule_on_holdout.py` with {center, services, service, centre};
   - one small test.
4. `sm.py jobs` (queue must be empty), then `sm.py nb start` and `sm.py publish` from the worktree.
5. Enqueue one job:
   - check that `work/output/bs_w3/pair_p.parquet` exists;
   - run `rule_on_holdout.py bs_w3`, then `france_variants.py bs_w3 bs_w3sx --rules swap_exact` and `france_variants.py bs_w3 bs_w3v --rules vswap`, then the validator.
   - **Expected:** holdout delta of `vswap` about 0; on test, `vswap` drops roughly half of what `swap_exact` drops.
6. For `s22`, whoever holds its WORK takes the v7 versions of `word_swap.py` and `france_variants.py`, then runs
   `python src/scripts/france_variants.py s22 s22v --rules vswap`. The portal difference to `s22sx` reads the kept pairs.

## 15. Remaining Plan

1. `vswap` (sections 14.2 to 14.6).
2. LLM judge, `src/scripts/llm_judge.py`:
   - export the pairs (holdout calibration set, France suspicious categories of `bs_w3`), keyed by original entity ids so the scores join onto any model;
   - `Qwen3-4B-Instruct-2507`, bf16, batched next-token P(Yes);
   - calibrate, sanity-check, then add a `judge:t` rule to `france_variants.py`;
   - if it runs in parallel with the retrain: per-notebook queue (queue prefix from the notebook name in `bootstrap.sh`/`jobrunner.sh`, `--queue` in `sm.py`) and an `ml.g5.8xlarge` notebook with the same lifecycle.
3. France-aware cross-encoder retrain (16xlarge, overnight):
   - new decoy kinds in `decoys.py` (another business at the S1's address; type-word sibling);
   - `bs_xenc` + `bs_final_stack` as a new `STK` (e.g. `bs_w5`), paired against `bs_w3`;
   - then rules on top.
4. Research while jobs run (not yet researched; ideas only):
   - listwise LLM matching (ComEM "select": judge an S1 against all its same-address candidates at once);
   - label-shift EM recalibration of France's prior (Saerens et al. 2002);
   - other French noise mechanisms found with the replacement-frequency method (added words, house-number shifts);
   - collective matching across S2/S3 twins (GraLMatch-style).
5. At the end:
   - validator with `--check-ids` on the chosen files;
   - update this handoff and memory;
   - stop the notebooks.

## 16. Things the Next Session Must NOT Do

- **Files and git:**
  - Don't create `HANDOFF.md` in either checkout: it overwrites the team's `handoff.md`.
  - Don't commit or push to `sai` or `main`. Don't edit the team's files on their branch.
  - Don't apply `stash@{0}` or re-run the cross-encoder v3 job without a deliberate decision: it is untested, costs hours of GPU, and overwrites `work/xenc`.
- **Notebooks:**
  - Don't start `barani-v5` before checking `jobs/pending/`.
  - Don't run two notebooks on the single shared queue.
  - Don't leave notebooks running idle.
  - Don't touch teammates' AWS resources.
- **Data and models:**
  - Don't train on test data or build training positives from France's test-derived noise words.
  - Don't validate the vocabulary with the "rate >= 500" learner on train (section 6).
  - Don't pick Qwen2.5-7B (8B budget) or non-MIT/Apache models.
- **Portal and claims:**
  - Don't upload to the portal or plan uploads as if they were ours: slots are shared and human-only.
  - Don't claim or promise 0.99. Don't fabricate portal numbers.
- **Security and style:**
  - Don't paste credentials, presigned URLs or Jupyter links. Don't ask for the SSH passphrase.
  - Don't add commit bodies or trailers.

## 17. Resume Instructions

1. Read this file, then plan v2 (`~/.claude/plans/i-want-to-setup-fuzzy-prism.md`), then `HANDOFF-v6_barani.md`, then the team's `EXPERIMENTS.md` section 10.1 and `research.md` section 25 on `origin/sai`.
2. Run `git fetch`, then:
   - `git -C /Users/barani200/amazon-ml-2026-v7 status -sb`;
   - `git log --oneline -3 origin/sai` (the team may have moved on; re-read their newest France notes before duplicating work);
   - `git stash list`.
3. Check AWS (the user may need to re-run `aws login --profile barani`): `sm.py nb status`, `sm.py jobs`.
4. Continue at section 14. Keep each change to its own one-line commit on `v7/france-vocab` and push.
5. Report results to the user in plain words with measured numbers. Update this file after each finished track.
