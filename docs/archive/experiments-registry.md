# Experiments: how we got the scores, what is running on every GPU, and what is still open

Team Nooglers. Last updated 26 Sep 2026 about 13:00 IST. Read this to see the approach end to end, pick an experiment nobody has run, and record your own result in section 6. Companion documents: `ARCHITECTURE.md` (the pipeline and every version), `TEAM_GUIDE.md` (how to run jobs on AWS), `handoff.md` (infrastructure and latest status), `research.md` (measurements and literature).

## 1. Ground rules for every experiment
- **One yardstick:** macro F0.5 on the locked holdout of 150,000 training S1 (`ber.split.holdout_q`, seed 2026). Nothing trains on it and no threshold is tuned on it. A change ships only if the paired bootstrap interval against the current best excludes 0: `python src/scripts/paired_models.py MODEL_A MODEL_B` (needs both models' `holdout_pred.parquet`).
- **Portal is the second yardstick** (public subset of the test set). It has read below the holdout every time (gap 0.0178 for `s4`, 0.0131 for `s12`, 0.0093 for `s17`), so a holdout gain is only "real" once a portal score confirms the direction. Portal uploads are the human's; 5 per day.
- **Leakage guards:** S1 used to fit an encoder or the cross-encoder are excluded from stage-two training and from the holdout; first-stage probabilities used by the stack always come from models that never saw those S1.
- **Names:** use unique model names and job names (your initials plus a number); the models `v*`, `s*` and folders `xenc*`, `stack_*` belong to the main line. Never sync your branch into `ber/code` (the shared prefix); use `ber/code_<initials>`.
- **Where to run:** see `TEAM_GUIDE.md` section 3 (job queue) and section 7 below (which lane and GPU are free). Long GPU jobs: tell the pipeline owner first.

## 2. How we got the scores (the ladder)

| Step | Model | What changed | Holdout F0.5 | Portal |
|---|---|---|---|---|
| 1 | `v0` | 42 features, XGBoost on token-blocking candidates (30 per S1) | 0.9377 (out-of-fold) | not uploaded |
| 2 | `v1` | 63 features: name rarity, glued names, digit alignment, romanised names, skeletons | 0.9551 (out-of-fold) | not uploaded |
| 3 | `v2` | cascade blocking (100 raw candidates pruned to 30 by a learned ranker) | 0.9565 | **0.944** |
| 4 | `s1`, `s2`, `s3all` | second stage: consensus features (competition between S1 for a record, digit and house-number agreement, TF-IDF) | 0.9617, 0.9671, 0.9677 | 0.949 (`s3all`) |
| 5 | `v3`, `s4` | name-only dense channel (multilingual e5-small fine-tuned on true pairs) for non-Latin names | 0.9598, 0.9708 | **0.953** |
| 6 | `v5`, `s5`, `s6` | dense channel over "name | address" of every record (hard negatives, pool to nearest S1); candidate shortlist (4.9 per S1 instead of 31) | 0.9757, 0.9832, 0.9831 | not uploaded |
| 7 | `s8`, `s12` | decoy edit features, more carried columns, 1.5M S1 for stage two, depth 9 | 0.98362, 0.98505 | **0.972** (`s12`) |
| 8 | `s11`, `s13` | **cross-encoder score** on the uncertain band (e5-small, digit tags, hard negatives) | 0.98887, 0.98917 | not uploaded |
| 9 | `s14` | stronger cross-encoder (e5-base, 700k fit S1, wider band) | 0.98986 | not uploaded |
| 10 | `v7`, `s15`, `s16` | first stage on 850k S1 with depth 9; competition features on the refined probability | 0.97796, 0.98935, 0.98946 | not uploaded |
| 11 | `s17` | base `v7` + e5-base cross-encoder rescored on `v7` bands + e5-small as a second feature + refined competition | **0.99025** | **0.981** |

| 12 | **`s22`** | cross-encoder on every shortlisted pair (catches confident look-alikes) | **0.99054** (+0.00029 [0.00018, 0.00039]) | not uploaded (next) |
| probes | `s17pf`, `s17pu` | France-only and US-only versions of `s17` (26 Sep 15:56 and 15:57) | n/a | **France only 0.187, US only 0.453**: France F0.5 is about 0.92, US and India about 0.99 |

What the ladder teaches: relative evidence (competition between S1 for the same record) was the first big gain; candidate recall (the two dense channels) was the second; **a model that reads both records jointly (the cross-encoder) was the third and the one that transfers best to the test set** (portal +0.009 for a holdout +0.005 from `s12` to `s17`). Small feature ideas, more data, depth and calibration each moved the holdout by at most 0.0005.

## 3. What did not help (do not repeat)
Extra token types in the blocker; higher document-frequency cap; calibration plus expected-F0.5 selection (+0.0001); name-only dense top-10 or top-5 with a cosine cut-off; name+address dense top-3 or top-5 (18 to 38M extra pairs, almost no recall); extra dense neighbours for empty-address pool records (`v6`, `s7`: 0.9756 and 0.9831, no change); a second consensus round on the stack's probabilities (`s9`, +0.00009); English-only encoders (cannot read Indic scripts).

## 4. What we know about the remaining error (`s15`, holdout)
Oracle on our candidates 0.9957; blocking loss 0.0043 and matcher loss 0.0064; precision 0.998; recall against all true pairs 0.972. 74% of the still-missed true pairs are pool records with an EMPTY address (4.4% of true pairs; 75% reach the candidates, 53% matched): name-only records, mostly ambiguous. Other misses: alias-only names 1.7k, injected or dropped words 2.7k, typos 2.2k. False positives 1,033, 58% look-alikes with no owner.

## 5. The holdout to portal gap: findings so far
- **France is the gap (26 Sep 16:00).** Country probes on the portal: France only 0.187, US only 0.453 (all other S1 left empty). Solved with France's population share 0.15 and the singleton share 0.057: France F0.5 about 0.92 (range 0.90 to 0.95), US plus India about 0.991, which equals their holdout (0.990) and post-stratified (0.9905) estimates. France costs 0.15 * (0.99 - 0.92) = 0.010, the whole 0.0093 gap. Details and formulas: `research.md` sections 23 and 24.
- **Why the model does not see it:** its own France estimate is 0.987. On the test it claims 63.8% of France's pool records (US 59.1%, India 58.0%), 3.53 matches per S1 (3.40, 3.37), has 14 times as many S1 with more than 5 S2 matches (impossible in training), and 5.9% of its predicted pairs sit below p 0.99 (US 2.0%). If France's real ownership matched the US, about 7% of the predicted France pairs are wrong (precision about 0.93). The errors are confident and spread over strata; shared addresses do not concentrate them.
- **Ruled out as the main cause:** S1-side attributes (post-stratified test estimate 0.9905), distractor density (`s23`, the holdout inside a test-like universe loses only 0.0007), name ambiguity (test US and India are less ambiguous than train), row order (no signal).
- **Adversarial validation (26 Sep):** AUC 0.8716 for US and India pairs, driven by set-size artefacts (blocking scores, competition counts, name counts); `s21` (features dropped) costs 0.00024 on the holdout; its portal effect is untested and now less relevant because US and India already transfer.

## 6. Experiment registry (add yours at the bottom)

### 6.1 Finished (holdout F0.5, paired against the row's reference)
| Id | Question | Result | Where |
|---|---|---|---|
| `s10` | more stage-two training data (1.5M S1 against 400k) | 0.98343, +0.00036 | `runs/s10` |
| `s8`, `s12` | decoy edit features, carried columns, depth 9 | +0.00054, +0.00198 over `s6` | `runs/s8`, `runs/s12` |
| `s11` | cross-encoder score (e5-small) | +0.00525 over `s8` | `runs/s11` |
| `s14` | e5-base cross-encoder | +0.00069 over `s13` | `runs/s14` |
| `v7`, `s15` | first stage on 850k S1, depth 9 | 0.97796 (`v7`), +0.00018 (`s15`) | `runs/v7`, `runs/s15` |
| `s16` | competition on the refined probability | +0.00011 | `runs/s16` |
| `s17` | everything above, both cross-encoders | 0.99025 (portal 0.981) | `runs/s17` |
| `s22` | cross-encoder on every shortlisted pair | **0.99054**, +0.00029 [0.00018, 0.00039] over `s17` | `runs/s22/output/` |
| `s18` | small cross-encoder on the base model's data | 0.99010, -0.00014 against `s17` (no gain) | `runs/s18` |
| `s21` (`v9`) | first stage without blocking-score and competition features | 0.99000, -0.00024 against `s17`; portal effect unknown | `runs/s21` |
| `v10`, `s23` | test-like universe (21% of train S1 dropped, 41% of the pool unowned as on the test) | first stage 0.97721 (`v7` 0.97796), stack **0.98955** (-0.0007 against `s17`); the holdout itself sits in the test-like universe, so extra orphan distractors cost at most about 0.0007: distractor density alone is not the portal gap; portal effect unknown | `runs/s23` (`work10` on the second notebook) |
| `s19` (`v8`) | joint name counts over train and test in the first stage | 0.99026, +0.00001 against `s17` (no gain) | second notebook |
| `s25` | `s17` plus address multiplicity features | 0.99028, +0.00004 (no gain); France over-claiming unchanged (64.2% of the pool claimed, 3.53 matches per S1, 205 S1 above 5 S2): address sharing is not the France cause | `runs/s25` |
| `s20` | e5-large (560M) cross-encoder as `xs`, e5-base as `xs2`, band pairs | 0.99030, +0.00005 against `s17` (no gain: a bigger cross-encoder is not the lever) | second notebook `work/output/s20` |
| `s22t2c` (France rules on `s22`) | type-word swap rule + France threshold 0.985 + cap | portal **0.984502** (`s22sx` 0.982477, `s17` 0.980502) | `runs/s22t2c` |
| analysis `ra1` to `ra5` | recall attrition, name ambiguity, ambiguity profile, post-stratification | see `research.md` section 23 | `jobs2/done/ra*.log` |

### 6.2 Running or queued now (26 Sep 13:00)
| Id | Question | Notebook, lane, GPU | Expected |
|---|---|---|---|
| `v9`, `s21` | shift-robust first stage (joint counts, blocking-score and competition features dropped) | main, lane 1, GPU 0 | portal gap closes; holdout about equal |
| `s18` | small cross-encoder on the base model's 966k pairs: size or data? | main, lane 2, GPU 0 (`ru2b`, then `ru3`) | decides whether to go bigger |
| `s17f85`, `f90`, `f95`, `g85` | France and global thresholds on `s17` | main, lane 2 (`rv9`); `s17f85` ready | France behaviour on the portal |
| `v8`, `s19` | joint name counts only | second notebook, lane B, GPU 0 | shift check |
| `s22` | cross-encoder on EVERY shortlisted pair (catch confident look-alike false positives) | second notebook, lane B3, GPUs 2 and 3 | +0.001 to +0.002 |
| `v10`, `s23` | **test-like universe**: drop 21% of the train S1 (outside the sample and holdout) before the competition and count features so the training data has the test's S1 count and distractor rate (`pairs.drop_frac`); separate work directory `work10` | second notebook, lane B4, GPU 0 (`rq0` to `rq4`) | calibration for test conditions; portal gap |
| `s24` | **third cross-encoder from another family**: `Qwen/Qwen3-0.6B` (Apache-2.0, 0.6B) with a one-logit head, fitted on the same 966k pairs as the e5-base model (one epoch, bf16, gradient checkpointing, prompt "Same business? A: ... B: ..." closed by `<|im_end|>`), scored on the v7 band pairs as feature `xs3`; stack `s24` = `s22` + `xs3` | second notebook, lane B3, GPU 2 (train, about 4.6 h from 15:20 IST), then GPUs 2 and 3 (scoring, about 2.5 h), then the stack build (`rn1`, `rn2`, `rn3`) | +0.0003 to +0.0008 over `s22`; ready about 00:00 IST |
| free | lane 2 (main) | idle | your experiment |

### 6.3 Ideas nobody has run (pick one)
1. **Two-fold cross-fitted cross-encoder** (fit on half of the non-holdout S1, score the other half, average both for test): every S1 gets an honest `xs` and the stack keeps all S1. Needs a `--fold` option in `stages/xenc.py`. Expected +0.0005 to +0.001.
2. **Density-ratio weights** for the stack: weight training pairs by p(test | features)/p(train | features) from the adversarial classifier so the stack trains on test-like data. Files: `src/scripts/adv_validation.py`, `stack.train`.
3. **Learned per-S1 shortlist size** (predict how many candidates to keep from the sorted first-stage scores) to shrink `candidate_pairs.tsv` further at no score cost (the organisers rank smaller candidate sets higher).
4. **Sibling support features** for empty-address pool records: similarity to all the S1's confident records (not only the best other one), pool-pool name links across S2 and S3.
5. **State-name normalisation** (`OK` and `Oklahoma`, Indic state names) in `text.py`; needs a re-prepare and rebuild.
6. **Cross-encoder ensembles** (different seeds, fit sets, bases such as bge-m3) as extra stack features `xs2`, `xs3` (`stack build --xenc-dir2` supports one extra; extend for more).
7. **Larger fit set or more epochs** for the cross-encoder (`--set fit_more=... epochs=...`), or a wider training band.
8. **A different second-stage learner** (LightGBM, CatBoost, a small neural net) or a seed ensemble of the stack; expected small (+0.0002).
9. **Cap of 5 S2 and 6 S3 matches per S1** in `predict.emit` (worth at most +0.0002).
10. **Per-country probes and thresholds** (section 8), and threshold re-tuning for the test's higher distractor rate.

### 6.4 Teammate experiments (add a row: id, what you tried, holdout F0.5 with paired interval, portal score if any, where the files are)
| Id | Owner | What | Holdout F0.5 (paired CI) | Portal | Files |
|---|---|---|---|---|---|
| | | | | | |

## 7. Compute map (26 Sep 13:30)
Progress 13:30: `ru2b` finished, `ru3` (`s18`) scoring; `rz4`, `rx1`, `ry0`, `rw2`, `rq1` running, all logs fresh, no failures. Results are added to section 6 as they arrive.

| Machine | Type | GPUs | Job lanes (queue prefixes) |
|---|---|---|---|
| `test-notebook` | `ml.g5.16xlarge`, 64 vCPU, 256 GB | 1 x A10G | `jobs/`, `jobs2/` (`jobs3/`, `jobs4/` after the next restart) |
| `test-notebook-2` | `ml.g5.24xlarge`, 96 vCPU, 384 GB | 4 x A10G | `jobsB/` (GPU 0), `jobsB2/` (GPU 1), `jobsB3/` (GPUs 2 and 3), `jobsB4/` (free) |

Set `CUDA_VISIBLE_DEVICES` explicitly in every GPU job script (the lanes share the machine's GPUs). The second notebook's working files come from `s3://sagemaker-us-east-1-567503593043/state/` (exported from the main notebook; results you produce there stay on its disk unless the job uploads them to `runs/<name>/`). Heartbeats: `diag/<notebook>.txt`.

## 8. Backlog for tomorrow (more submissions may be allowed)
- **Per-country probe files:** take a good matching file and empty the predictions of one country's S1; the drop in portal score gives that country's F0.5 on the test set (France, India, US). Uses the allowed submissions, is diagnostic only, never final.
- **France threshold sweep** on the portal with `reemit.py` variants; a stricter France threshold favours precision, which F0.5 rewards.
- **Final choice** by holdout and portal together, then the freeze: code zip, one reproduction run (`reproduce_final.sh`), the official validator with `--check-ids`, the methodology document.

## 9. Research notes: what could still be missing for the gap (26 Sep 13:00)
Literature (abstract level): importance weighting with a train-versus-test classifier is the standard covariate-shift remedy, also for tree ensembles (arXiv 2410.20978, 2007.04043); gradient-boosting discriminators can localise and correct the features that cause a shift (DataFix, 2312.04546); an entity-resolution case study reports the same training-to-production gap and closes it with data-centric fixes (2111.10497); collective or group-level decoding helps sparse records (GraLMatch 2406.15015, label propagation 2605.25814).

Ideas ranked by expected effect on the portal gap:
1. **Test-like training universe** (`v10`, `pairs.drop_frac`): reproduces the test's S1 count and unowned pool share from labelled data; all competition and count features then have test-like distributions and the models see the test's distractor rate. Running.
2. **Shift-robust first stage** (`v8`, `v9`): joint name counts; drop of blocking-score and competition features; running.
3. **Cross-encoder on every shortlisted pair** (`s22`): lets the cross-encoder veto confident look-alikes. Running.
4. **Density-ratio weights** for the stack (adversarial classifier output as sample weights): not built.
5. **Cross-encoder test-time augmentation** (score each pair with the two records swapped, or with the romanised name, and average): not built, cheap on the free GPU lanes.
6. **Group-level decoding** (link pool records of one entity across S2 and S3 and assign the group to one S1): partly captured by the stack's similarity to the S1's best other record; a full version is not built.
7. **Transductive use of the unlabelled test set** (pseudo-labels): not used; the rules say the model is built from the provided training data, so confirm with the organisers before trying it.

## 10. Ranked plan with gains (26 Sep 16:30, after the country probes; supersedes the 15:00 ranking)
Facts: the portal gap is France (about 0.010 of score). US and India transfer from the holdout without loss. The holdout still has +0.001 to +0.002 of realistic headroom (about two thirds of the missed recall is name-ambiguous and unrecoverable, `research.md` section 23). So France work is worth five times any holdout work. Score arithmetic: overall = 0.15 * F(France) + 0.85 * F(US, India); every +0.01 of France F0.5 is +0.0015 overall; France 0.92 to 0.99 would be +0.0105 (portal 0.9915), which is the ceiling.

| Rank | Idea | France F0.5 effect | Overall gain (portal) | Cost | Evidence and risk | State |
|---|---|---|---|---|---|---|
| 0 | **Word-swap decoy rule (France)**: drop predicted pairs whose core names differ by one swapped common word (`s22sx`: only when the S1 has a confident exact-name copy, 36.7k pairs; `s22sw`: all swaps below p 0.9999, 41.6k) | +0.026 if the dropped pairs are wrong, -0.016 if true (France F0.5) | +0.0039 to -0.0024 | 1 slot | tests the France decoy mechanism; if it wins, extend to legal-form and house-number siblings | **submitted: 0.982477 (+0.0011); about 47% of the dropped pairs were wrong, break-even is 26%** |
| 1 | **France threshold sweep on `s22`** (`s22f97`, `s22f985`, `s22f995`; portal reads the France curve, one slot each) | 0.92 to 0.93 to 0.95 if the wrong pairs sit in the low-probability tail | +0.001 to +0.003 | 1 to 3 slots | break-even correctness is about 0.74, France's model probabilities are over-confident, so the best model-p cut-off is 0.975 to 0.99; if the errors are spread over high p the gain is smaller | files ready, `runs/s22f97`, `runs/s22f985`, `runs/s22f995` |
| 2 | **Cap of 5 S2 and 6 S3 per S1** (impossible extras: 186 France S1 against 35 US) | small | +0.0002 | 1 h | pure rule, safe | not built |
| 3 | **France calibration by distribution matching**: choose France's cut-off (or a monotone remap of p) so its predicted ownership (63.8% of the pool) and matches per S1 (3.53) match the US and India (59% and 3.40) | up to 0.95 to 0.96 | +0.004 to +0.006 | 2 h, portal-checked | assumption: France's truth is as US-like as the generator suggests; use the sweep of rank 1 to confirm | after rank 1 |
| 4 | **France-aware features and training**: pool-side evidence the model is over-trusting (see `france_profile.py`); `s25` (address multiplicity, running), name-length and generic-name features, France legal forms | 0.95 to 0.97 if a real cause is found | +0.004 to +0.008 | 4 to 8 h | cannot be validated on France without labels; only portal probes | `s25` running; more ideas need the sweep first |
| 5 | Ship the best holdout model as the base for all France work: `s22` now, `s24` (Qwen3 third cross-encoder) or `s25` if they beat it | 0 | +0.0003 to +0.001 | running | paired test decides | `s24` about 02:00 IST, `s25` about 16:45, `s20` about 20:00 |
| 6 | Small bundle: stack seeds, refit with the holdout S1 | 0 | +0.0002 to +0.0004 | 2 h | no holdout number after a refit | not built |
| 7 | Extra channel for the 1,600 never-proposed pairs with an address | 0 | +0.0002 to +0.0004 | 6 h+ | new index | not built |
| 8 | Density-ratio weights, test pseudo-labels | unknown | unknown | 3 to 6 h | pseudo-labels: ask the organisers first | not recommended |
| Not worth doing | more neighbours for empty-address records, expected-F0.5 decisions, second consensus round, holdout threshold tuning, S1-attribute reweighting, small cross-encoder alone, further distractor-density work (`s23`) | | | | measured | |

Expected portal outcome if ranks 1 to 5 land: 0.984 (threshold only) to 0.988 (calibration works) to 0.990 (a real France cause found). The 0.988 people on the leaderboard are consistent with a France fix.

Portal probes of the day: `s17pf` 0.187 and `s17pu` 0.453 done; one slot left on 26 Sep, planned `s22f985`. Tomorrow: `s22f97`, `s22f995`, `s22` alone, then `s24` or the France-aware model.

### 10.1 Variants ready for the five slots of 27 Sep (all on `s22`, all include the `swap_exact` rule of `s22sx`, scored 0.982477)
Files `runs/<name>/output/matching_results.tsv`; a portal difference to `s22sx` is a direct reading of the extra pairs dropped (break-even 26% wrong; per 1% of France's pairs: -0.0022 France F0.5 if true, +0.0063 if wrong, i.e. -0.00033 or +0.00094 overall).
| File | Extra rule on top of `s22sx` | Extra pairs dropped |
|---|---|---|
| `s22a` | name similarity < 60, strong address, same house number, p < 0.999 | 17.8k (total 54.6k) |
| `s22b` | the same with name similarity < 80 | 21.9k (total 58.6k) |
| `s22c` | legal-form conflict, p < 0.9999 | 13.3k (total 50.1k) |
| `s22d` | France threshold 0.985 | 47.6k (total 84.4k) |
| `s22e` | tiny pool names that are not the initials of the S1 name, p < 0.9999 | see `runs/s22e` log |
Order suggestion: `s22a`, `s22c`, `s22d`, then the combination of what won, then the final file. Realistic total gain of these categories: +0.001 to +0.003 (portal 0.9835 to 0.9855); +0.005 needs categories with more than 60% wrong pairs.

### 10.2 Plan for the five slots of 27 Sep (revised after `research.md` section 26; portal `s17` 0.980502, `s22sx` 0.982477)
Files (all on `s22`, `runs/<name>/output/matching_results.tsv`): `s22t1` = rule `typeswap` only (24.5k pairs, decoy share about 80%, expected about +0.0004 over `sx`), `s22t2` = `typeswap` + France threshold 0.985, `s22f1` = all swaps (53.5k), `s22f2`/`s22f3`/`s22f4` = all swaps + threshold 0.985 / 0.97 / 0.995. Shift-model tests (whole files, no France rule): `s23` (test-like universe), `s21` (shift-robust first stage).
1. `s22t1`. 2. `s22t2` (or `s22f1` if `t1` disappoints). 3. `s23` alone: separates world A from world B (see section 26): compare with `s17` 0.980502; if it gains, rebuild the France rules on `s23` (`france_variants.py` needs `output/s23/pair_p.parquet` in `work10`). 4. Best France rule on the best base. 5. Final validated file (the last upload must be the best one if the portal counts the last).
Update 20:00: the test-like evaluation of `v7` (`research.md` section 26.2) shows that shift costs US and India at most 0.001 to 0.0015: drop slot 3 (`s23`). Use it for the other France question instead (`s22t2`/`f`-variants, or a global threshold check `s17g85`).

### 10.3 Review of the teammate's plan (26 Sep 20:30) and what we took from it
- **Adopted:** symmetric cross-encoder (random A/B order in training, both orders averaged as logits when scoring, `xs_asym` = |logit(A,B) - logit(B,A)| as an uncertainty feature; the teammate's pilot: hard-band AP 0.9146 to 0.9604). Implemented as `xenc ... --set symmetric=1`; job chain `rc1` (train, e5-base recipe of `s17`, GPU 0), `rc2` (score every shortlisted pair on GPUs 0 and 1), `rc3` (stack `s26` = `s22` with the symmetric score as main feature and the old e5-base score as `xs3`, paired test against `s22`). Ready about 03:00 to 04:00 IST. Hard capacities 5 S2 and 6 S3 after exclusivity: `france_variants.py --cap` (variants `s22t1c`, `s22t2c`, `s22cap`).
- **Not adopted, with reasons:** two-fold cross-fitting (two more cross-encoder fits plus stack changes for about +0.0002 to +0.0005, leakage risk; revisit only if `s26` wins); sibling candidate channel (the closest experiment, extra empty-address neighbours `v6`/`s7`, gave nothing; two thirds of the misses are name-ambiguous and the stack already has the sibling name evidence; the candidate oracle 0.9957 is not the binding limit, the matcher is); France stack features learned from labels (US and India labels contain no France-type decoys: swap pairs are true 99.7% there, so `typeswap_risk` would get no weight; the structural rule is the correct place); density-ratio weights (the test-like evaluation of `v7` shows a mismatch penalty of only 0.0006, below the plan's own 0.001 gate); "no portal probes" (France has no labels; the portal gave `s22sx` +0.0015 over `s22`; we keep it but only for at most five predeclared structural variants).
- **Kept as optional after `s26`:** three stack seeds (+0.0001 to +0.0003), a learned per-S1 candidate budget (the organisers rank smaller candidate sets higher; weight unknown).

### 10.4 France variants ready for 27 Sep (all on `s22`, all with the caps 5 S2 / 6 S3; portal `s22t2c` 0.984502 is the base to beat)
| File | Rules | Pairs dropped in France |
|---|---|---|
| `s22u3` | type-word swaps + threshold 0.995 | 104k |
| `s22u4` | type-word swaps + threshold 0.97 | 66k |
| `s22u1` | type-word swaps + threshold 0.985 on non-exact-name pairs only | 56k |
| `s22u2` | type-word swaps + threshold 0.995 on non-exact-name pairs only | 68k |
| `s22v3` | as `s22t2c` but the swap flag ignores spaced legal forms (`e u r l`) | 79.5k |
| `s22v1` | swaps (spaced-legal aware) + threshold 0.985 that spares equal names after legal spacing and initials pairs | 49.5k |
| `s22v2` | the same with 0.995 | 58.2k |
Suggested order: `s22u3` (cut-off direction), `s22v1` (do the protected pairs help), then the combination; `s26` versions after 03:00.

### 10.5 Compute map, 26 Sep 21:10
| GPU | Job | Ready |
|---|---|---|
| second notebook GPU 0 | `rc1` symmetric e5-base cross-encoder (seed 0), then `rc2` scoring (train split), `rc3` stack `s26` | 21:15, 23:15, 00:45 |
| second notebook GPU 1 | idle now; `rc2` scoring (test split) from 21:15; then `rd2` scoring of seed 1 (test split) | 23:15 to 01:15 |
| second notebook GPU 2 | `rn1` Qwen3-0.6B cross-encoder training, then `rn2` scoring (with GPU 3), `rn3` stack `s24` | 22:05, 00:35, 02:05 |
| second notebook GPU 3 | idle until `rn2` | |
| main notebook GPU | `rd1`: seed 1 of the symmetric cross-encoder (23:30), then `rd2` (second notebook, lane B2): score both splits with seed 1, average the logits of the two seeds (`xenc2SymE_v7`, columns `xs`, `xs_asym`, `xs_seed_gap`), stack `s27`, paired tests against `s22` and `s26` | about 03:30 |
Update 21:25: second-notebook GPU 3 now runs `rg10` (lane B4): two more XGBoost seeds of the `s22` stack (`s22s1`, `s22s2`, `seed=1,2`, same chunk files), then `blend_stacks.py` (mean of logits, threshold re-tuned on the blended out-of-fold probabilities) gives `s22e`, paired test against `s22`, France rules can then be run on `s22e` (`france_variants.py s22e ...`). Expected +0.0001 to +0.0003. `blend_stacks.py` also blends `s26`/`s27`/`s24` variants later.
Update 22:45: Qwen3-0.6B scoring (`rn2`) and its stack (`s24`) are cancelled to save credits (the fitted model is `work/xenc3Q/model`); exports of the whole work directory to the shared bucket (`re1`, `re2`) are queued; the team continues on its own AWS (`TEAMMATE_HANDOFF.md`).


## 11. Continuing on our own AWS after the handoff (27 Sep, 04:00 to 13:00 IST)

Credits topped up (+100 USD) after the handoff; resumed on `test-notebook-2` (account 567503593043), resized as needed (`g5.8xlarge` for decoding, `g5.12xlarge`/`g5.24xlarge` for the Qwen and mDeBERTa jobs; `g5.12xlarge` was twice reported "temporarily unavailable" by AWS, `g5.24xlarge` always available). Barani continued in parallel on their own account (767397931665, notebook `barani-v5`), pushed branch `v8/france` (65 files, `france_variants.py` rules `thrpn`, `protect`, `restore`, `typeins`, `xfr`, `legalhouse`; `xenc_fr.py` for a France-aware cross-encoder; scripts `france_recall.py`, `france_empty.py`, `xfr_report.py`, `xfr_slots.py`, `xs_merge.py`, `stack_predict_merge.py`). We deployed their branch read-only to a separate S3 prefix (`ber/code_v8`) to run their rules on our own bases without merging (their `stack.py`/`france_variants.py` edits conflict with ours).

### 11.1 New bases confirmed overnight (holdout, paired against `s27` 0.99063 unless noted)
| Id | What | Holdout F0.5 | Paired delta | Ship |
|---|---|---|---|---|
| `s26` | symmetric e5-base cross-encoder (single seed) | 0.99053 | -0.00001 vs `s22` [-0.00010, +0.00008] | no |
| `s22e` | 3 XGBoost seeds of the `s22` stack, blended | 0.99053 | -0.00001 vs `s22` [-0.00010, +0.00008] | no |
| **`s27`** | symmetric e5-base cross-encoder, 2 seeds averaged (`xs`, `xs_asym`), stack | **0.99063** | +0.00009 vs `s22` [+0.000004, +0.00019] | **yes, new base** |
| `s27r` | `s27` refit with the holdout S1 folded into training (`--set refit_all=1`, added to `stack.train`) | in-sample only (99.2% of S1 lists identical to `s27`; OOF unchanged 0.9904) | n/a (not paired-testable) | no, reverted |
| **`s29`** | `s27` + Qwen3-0.6B score in the `xs2` slot (band pairs only, as scored by barani) | **0.99088** | **+0.00025 vs `s27` [+0.00015, +0.00034]** (+0.00034 vs `s22`) | **yes, best base** |
| `s31` | `s29` + Qwen extended to the near-confident (0.98-0.999) and very-low (0.005-0.02) probability zones (+1.2M pairs each split, scored by us) | 0.99089 | +0.00001 vs `s29` [-0.00005, +0.00009] | no gain |
| `s31b` | `s31` + `e5-small` as a 4th cross-encoder feature (`xenc_v7`) | 0.99091 | +0.00003 vs `s29` [-0.00004, +0.00010] | no gain |
| `s28a/b/c` | XGBoost depth/eta/rounds sweep on `s27`'s features | 0.99054/0.99062/0.99061 | -0.00009/-0.00001/-0.00002 vs `s27` | no gain |
| `s28d` | `s27` + singleton-weighted loss (`w_singleton=3`, added to `stack.train`) | 0.99051 | -0.00013 vs `s27` | no gain |
| `s28e` | `s27` + LightGBM learner | failed (`lightgbm` not installed on the notebook; not retried, no further gain expected given the XGBoost sweep) | | |
| per-country thresholds | separate threshold for US and India (`country_thr.py`, two-fold honest tuning) | | -0.00001 vs the global threshold | no gain |
| `s30` | `s29` + `microsoft/mdeberta-v3-base` (MIT) as a 4th cross-encoder | **failed**: AdamW's first optimizer step makes ~200 of ~202 DeBERTa-v2 parameters non-finite in this torch/transformers build (`torch 2.10.0+cu128`, `transformers 5.17.0`); confirmed with a parameter-by-parameter gradient probe (`bf16`, `fp32` and `eager` attention all fail the same way; SGD does not). About 1.9 GPU-hours lost before the kill reached the notebook. Not retried (would need an older torch/transformers pin or a different optimizer, not worth the remaining time). | | |

**Conclusion: the Qwen line is exhausted.** Its only real gain is `s29`'s original band-only addition (+0.00025, CI clear of 0); every extension (wider coverage, adding a 4th cross-encoder) landed inside noise. Stack-side tuning (`s28*`, country thresholds) found nothing. `s29` is the best base.

### 11.2 Qwen coverage analysis (`qwen_bins.py`, `qwen_france.py`)
- Qwen (as delivered by barani) scores only the uncertain band (`band_lo=0.02` to `band_hi=0.98` at first-stage `p1`, plus some easy examples), about 30% of test pairs and 23% of train pairs.
- Extending it to the near-confident zone (`p1` 0.98 to 0.999, where the false-pair count is tiny — 577 to 1,485 false pairs per bin in the labelled train data — but each one is a confident false positive, the kind that hurt France) and the very-low zone (0.005 to 0.02) added 1.2M pairs per split; `s31`'s result above shows this did not help.
- `qwen_france.py`: in the France pairs Qwen has scored, its bins do not cleanly separate decoys from true pairs the way the slot-limit fit already does by relation (`swap`/`other`) — e.g. `exact` name pairs below the France threshold with a high Qwen score are still mostly decoy (0.66/0.53) but so are ones with a mid Qwen score (0.57/0.43): no clean cut-off. No new France rule found from Qwen.

### 11.3 France rule research (label-free, slot-limit fit; scripts `pool_support_scan.py`, `mate_scan.py`, `typeswap_calib.py`, `clone_probe.py`)
All ruled out as a decoy mechanism (no new rule):
- **Pool-support / clustering:** 98% of France swap pairs have no same-core-name mate among the S1's other candidates (`pool_support_scan.py`); a pool record closer to an unclaimed record at its own address than to the S1's list is under 1% of pairs (`mate_scan.py`). Decoys are lone records, not small clusters.
- **Exact address cloning:** decoy-type swap pairs and generic-word swap pairs have almost identical address-equality rates (17.5% vs 17.7%, `clone_probe.py`); decoys are not literal clones of the S1 record.
- **Per-word calibration** (`typeswap_calib.py`): confirms the current `typeswap` word list (ratio >= 0.75) is well separated from generic noise words (`france`, `services`, `fils`, `groupe`, `developpement`, all under 0.22 fitted decoy share); lowering the ratio cut-off to 0.6 only adds about 3k pairs and about +0.0003 estimated overall gain, not tested on the portal (superseded by barani's `thrpn`).
- **Sure-negatives classifier** (an S1 already at its slot cap: any further pair from that source is certainly wrong): only 22 such pairs exist in France, too few to train a classifier on.

### 11.4 Portal readings today (base to beat from the handoff: `s22t2c` 0.984502)
| File | What | Portal | Reading |
|---|---|---|---|
| `v8u_s27_AR` | `s27` + `typeswap` + `thrpn:0.995` + `protect:0.9` + `restore` (barani's recipe) | **0.985578** | +0.001076 over `s22t2c`; confirms the recipe |
| `v8u_s22F12n_AR` | same recipe on `s22`, France-aware cross-encoder score inside the stack | **0.984136** | -0.0014 vs `v8u_s27_AR`, -0.0004 vs `s22t2c`: **the France-aware cross-encoder is a net loss** (rejects too many true copies); do not submit further `F12n`/`xfr` files without a different design |
| `v8w_s29_AR` | same recipe on `s29` (built by us, reproduces `v8u_s27_AR`'s recipe byte-identically on `s27` as a control) | not yet submitted | estimate **~0.9858** (0.9852 to 0.9865): `s27`->`s29` holdout gain (+0.00025, ~85% US/India weight) scaled by the portal's historical ~3x multiplier for cross-encoder changes |
| `v8w_s31_A/AR`, `v8w_s31b_AR` | same recipe on `s31`/`s31b` | not submitted | expected within noise of `v8w_s29_AR` (same France drop rates, same restore counts); not worth a slot |

Files in `s3://sagemaker-us-east-1-567503593043/runs/<name>/output/` and mirrored to the shared bucket for `v8u_*`. Local copies of `v8u_s27_AR`, `v8u_s22F12n_AR`, `v8w_s29_AR` in `~/Downloads/`.

### 11.5 Organiser rulings that resolve earlier open questions (public Q&A sheet, checked 27 Sep)
- **Unsupervised statistics, self-training and pseudo-labels on the unlabeled test files are explicitly allowed** ("Computing unsupervised stats ... on the provided test files uses no external data or labels, so it is allowed. Self-training and synthetic pairs from the provided records are also fine.", asked and answered at least 8 times). This resolves the `EXPERIMENTS.md` section 9 caution about `typeswap`/`thrp`/`thrpn`/`xenc_fr.py` using test-derived statistics: they are within the rules. Disclose the mechanism in the methodology document.
- **The private leaderboard scores the team's best PUBLIC submission**, not the last upload. No need to reserve the final slot for the best file.
- **Candidate-set size is judged as total pairs** (a separate criterion from the F0.5 score); a pruned later stage is fine as long as `candidate_pairs.tsv` is the exact set fed to the model.
- No model above 8B parameters is allowed even as a distillation teacher.

### 11.6 Code changes this session (uncommitted at write time, see git log for the commit)
- `stack.py`: `refit_all` (fold the holdout into training, in-sample only, not shippable), a 4th cross-encoder slot (`--xenc-dir4`, feature `xs4`), a `--learner {xgb,lgb}` switch, singleton-weighted training (`w_singleton`). `predict()` handles both learners.
- `xenc.py`: a `bf16`/`fp32` override for `train`/`score` (DeBERTa families default to bf16), a non-finite-loss guard that raises after printing the first 5 steps (stops a broken run instead of burning the full training budget — this is what should have caught `s30` in seconds instead of after 1.9 hours; the guard was added mid-`s30` and did catch it on the retry).
- `france_variants.py`: added an s27-local reimplementation of barani's `protect` semantics (kept separate from their `thrpn`/`restore`, which we deploy from their branch instead of merging).
- New analysis scripts (`src/scripts/`): `pool_support_scan.py`, `mate_scan.py`, `typeswap_calib.py`, `clone_probe.py`, `qwen_bins.py`, `qwen_france.py`, `country_thr.py`.


Note for readers of this branch: section 11 was written on branch `sai` (commit `bb0b6b3`). The `stack.py` / `xenc.py` / `france_variants.py` changes it lists live on `sai` only (they conflict with this branch's versions); the analysis scripts it names are copied here.

## 12. France error classes read from raw records (27 Sep 2026, 14:00 to 16:00 IST, sai side)

Portal: **`v8w_s29_AR` 0.985875** (best; +0.000297 over `v8u_s27_AR`, which confirms the portal moving about 3x the holdout for cross-encoder gains). With US/India at the holdout (0.9909, world A), France is about **0.957**; the top of the leaderboard (0.9906 on 26 Sep) implies France about 0.988 there. France's remaining deficit (about 0.031 France F0.5, about 0.0047 overall) is the whole gap to the top.

Method: raw samples of real records rather than the slot fit, then counts against the US/India test and the labelled holdout. Scripts (all `src/scripts/`, jobs `aws/queue/jobs/sx_*.sh`, run on `test-notebook-2` of account 567503593043 with `BER_WORK=/home/ec2-user/SageMaker/work`):

| Script | Question | Finding |
|---|---|---|
| `sibling_fingerprint.py` | do two copies of one S1 share raw-text noise, so namesake records with an empty address can be told apart? | weak: raw name equal 4.9% between copies of one S1 vs 2.1% between namesakes; the owner's copies are the closest 34% of the time vs 29% by chance. Worth about +0.0002 at most; dropped |
| `france_explore.py` | how does the generator fill a shared building (loose address key, raw samples, train truth vs France predictions)? | **train distractors are hidden siblings: same brand with a changed type/suffix word AND a changed house number** (`Classic Ram Infrastructure` @28/1B vs `Classic Ram Food` @28/4B, `Kolkata Products Holdings` @28/5G). France siblings often sit in the same building instead |
| `france_errors.py` | what does the `s29` recipe keep, drop and restore; what do over-cap S1 look like? | kept decoys are mostly **the same name at another street** (`Bordeaux Parents` @25 rue Renault vs @25 R du Mirail); about 15% of the high-p drops of `thrpn` are **coined aliases at the S1's exact address** (`Kelojax`, `Syndelta`), which are true in train |
| `street_cluster.py` | first street-mismatch instrument | contaminated: department names (`gironde`, `nord`) counted as street words; superseded by `namesake_street.py` |
| `namesake_street.py` | street words rare in BOTH S1 and pool; street same/mismatch, house same/diff, namesake count per name | **exact-name pairs in another street with 6+ namesake S1: France 0.87% of predicted pairs (mean p 0.95), US 0.11%, holdout 99.5% true (US/India).** Unique names show no excess (France 0.11%, US 0.29%). Generic city-brand names (`bordeaux club sarl`, 530 S1) have namesakes in the same city and small house numbers collide; US/India namesakes sit in other cities. About 6.9k of about 7.9k such France pairs are decoys (about 87%); the recipe keeps 99.5% of them because `thrpn` spares exact names. **The slot fit cannot see exact-name decoys** (they are counted among the "sure" exact copies k), which is why "exact names carry no decoys" was concluded earlier |
| `single_match.py` | are France's extra one-match S1 singletons that caught a namesake? | France (after the recipe) 5.39% empty and 7.08% one match vs US/India about 5.8% and 6.0%. Holdout: single-match S1 are singletons only 0.2-0.3% of the time even with an away match. France has 3.7x the US rate of single matches in another street: lever about +0.0003, not built |
| `alias_drops.py` | does the recipe drop `X Co formerly known as <S1 name>` records? | only 465 in `ARtL` (245 in `AR`); `legalx` fires on about 220 of them (it reads the alias's `Co` as a legal form) |
| `france_post.py` | post-rules on a recipe run | `nsaway:N` (drop exact-name pairs with a street mismatch when N+ France S1 share the name), `nsaway_all:N` (non-exact too), `coined` (restore raw-predicted pairs whose pool core is one coined word, 6+ letters, outside France's S1 name vocabulary, at the S1's street and house number, slot caps kept) |

### 12.1 Files built (all `check_submission.py` OK; `s3://sagemaker-us-east-1-567503593043/runs/<name>/output/`), not uploaded
| File | Rules on `s29` | France pairs changed | Estimated overall gain over `v8w_s29_AR` |
|---|---|---|---|
| `v8w_s29_ARt` | `AR` + extended typeswap (`typeswap:1.01:0.6:30:300`, 45 words) | +3.9k drops (fitted decoy 0.76-0.81) | +0.0003 |
| `v8w_s29_ARtL` | + `legalx:1.01` | +8.7k drops (fitted decoy 0.30-0.46, biased) | 0 to +0.0002 more |
| `v8w_s29_ARtL99`, `v8w_s29_ARtL9` | as `ARtL` with `thrpn` 0.999 / 0.9999 | +9.2k / more | unknown; test only after the two rules above |
| **`v8w_s29_ARtLNC`** | `ARtL` + `nsaway:6` + `coined` | -6,177, +5,862 | **+0.001 to +0.0013 in total (about 0.9870)**; best guess |
| `v8w_s29_ARtLN` | `ARtL` + `nsaway:6` | -6,177 | isolates `nsaway` (about +0.0005) |
| `v8w_s29_ARtLNaC` | `nsaway_all:6` + `coined` | -7,468, +5,862 | |
| `v8w_s29_ARtLN2C` | `nsaway:2` + `coined` | -6,944, +5,862 | |
Suggested order if slots exist: `ARtLNC`, then `ARtL` (isolates the two new rules), then the `thrpn` direction.

### 12.2 What still hides about 0.02 of France F0.5 (not measured)
1. France recall inside the `thrpn` 0.995 cut: about 34k non-exact pairs dropped at about 45% decoys, i.e. about 19k true pairs lost. A per-class restore (like `coined`: typo'd type words `sport`/`sportif`, `& Associes`, `Cie` additions at the exact address) could win about +0.0005 to +0.0007.
2. Same street, other house number, exact name: 14.7k France pairs at p about 0.9 (`Calais Federation` @295 vs @524); this is the train distractor pattern. Needs the holdout rate of the same cell before a rule.
3. France blocking recall has never been measured (US/India candidate oracle 0.9957).

## 13. Final evening (27 Sep 16:00 to 23:59 IST) and the freeze

**Final file: `v8w_s29_FIN`, public leaderboard 0.987745** (the private leaderboard uses the best public submission). `s29` + barani's final France
recipe (`S29_FIN_BUILD_barani.md`; all sanity checks within tolerance except `legalx` firing on 9,589 pairs against 6,383 on `s28`, a base-model
difference, not a bug). Rebuild: `reproduce_final.sh` steps 8c and 9; package: `submission/make_package.sh v8w_s29_FIN`.

| File | What | Leaderboard |
|---|---|---|
| `v8w_s29_AR` | `s29` + the version-8 France recipe | 0.985875 |
| **`v8w_s29_FIN`** | `s29` + final recipe (thrpn 0.9999, French legalx, namesake / near-namesake drop lists, coined re-adds) | **0.987745** |
| `v8w_s29_HYB` | FIN + our French-aware restores (FRZ) + barani's `v9_xF2_FIN` adds and drops, `s29` base | 0.987143 |

What was tried after FIN (all on `s29`; scripts in `src/scripts/`, jobs `aws/queue/jobs/sx_*.sh`, Sai-account runner in `aws/queue/sai_notebook/`):
- **`s32`** = `s29` + `jhu-clsp/mmBERT-base` (MIT) cross-encoder as xs4 (trained and scored on Sai's account): holdout +0.000037 [-0.000034, +0.000109]. No gain.
  `gte-multilingual-reranker-base` could not run (its remote code crashes in transformers 5.17); `xenc.py` gained `remote_code=1` and drops segment ids
  for single-segment encoders.
- **France pseudo-label student** (`france_student.py`): AUC 1.000 on its own pseudo-labels, but it dropped about 56k mostly true copies and restored
  sibling decoys (every positive label sits at the S1's own address, so everything else looks negative). Not used.
- **Per-cell excess over the US/India rate** (`excess_cells.py`): invalid for drops (France makes far more domains and initials, and its probabilities
  sit lower, so rates differ for reasons that are not decoys).
- **Typo restores** (`typo_restore.py`, 1,090 pairs, all samples true) and **multi-word type changes** (`glued_typeswap.py`, 153 kept, mostly decoys;
  glued/domain type swaps are brand words, not decoys): too small to matter.
- **French-aware cross-encoder** (`xfz.py`: training pairs rewritten into French form with the same labels; 90k original + 210k French-ized pairs,
  warm start from the e5-base model, one epoch). On 60k labelled pairs: French-ized AP 0.9970 to 0.9998 (errors 7,749 to 1,333), original unchanged.
  `xfz_decide2.py` re-added FIN-dropped France pairs with its score >= 0.98 (posterior from per-bin likelihood ratios at France's prior), 366 pairs below
  the stack threshold with score >= 0.995, and dropped 142 type swaps into another real word (`v8w_s29_FRZ`). Its drop side contradicted the measured
  noise-word evidence and was not used. Combined with barani's `v9_xF2_FIN` changes, the leaderboard fell to 0.987143 (-0.0006): **rewriting US/India
  pairs into French teaches French vocabulary but not France's decoy semantics** (in the training countries a swapped category word is noise, in
  France it marks a sibling business), so the re-added pairs include siblings. A French-aware model needs France-specific negatives, not only
  translated training pairs.

Compute at the end: every notebook of account 567503593043 stopped (28 Sep); `sai-xenc` (Sai's account) stopped at 23:31. Nothing deleted.
