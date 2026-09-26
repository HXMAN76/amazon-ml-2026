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
| 11 | **`s17`** | base `v7` + e5-base cross-encoder rescored on `v7` bands + e5-small as a second feature + refined competition | **0.99025** | **0.981** |

What the ladder teaches: relative evidence (competition between S1 for the same record) was the first big gain; candidate recall (the two dense channels) was the second; **a model that reads both records jointly (the cross-encoder) was the third and the one that transfers best to the test set** (portal +0.009 for a holdout +0.005 from `s12` to `s17`). Small feature ideas, more data, depth and calibration each moved the holdout by at most 0.0005.

## 3. What did not help (do not repeat)
Extra token types in the blocker; higher document-frequency cap; calibration plus expected-F0.5 selection (+0.0001); name-only dense top-10 or top-5 with a cosine cut-off; name+address dense top-3 or top-5 (18 to 38M extra pairs, almost no recall); extra dense neighbours for empty-address pool records (`v6`, `s7`: 0.9756 and 0.9831, no change); a second consensus round on the stack's probabilities (`s9`, +0.00009); English-only encoders (cannot read Indic scripts).

## 4. What we know about the remaining error (`s15`, holdout)
Oracle on our candidates 0.9957; blocking loss 0.0043 and matcher loss 0.0064; precision 0.998; recall against all true pairs 0.972. 74% of the still-missed true pairs are pool records with an EMPTY address (4.4% of true pairs; 75% reach the candidates, 53% matched): name-only records, mostly ambiguous. Other misses: alias-only names 1.7k, injected or dropped words 2.7k, typos 2.2k. False positives 1,033, 58% look-alikes with no owner.

## 5. The holdout to portal gap: findings so far
- The test is harder than the training data: 5.8 pool records per S1 against 4.7, about 40% of pool records unowned against 26%, 47% India against 40%, 15% France (no labels). The model-based estimate for `s15` gives US 0.9940, India 0.9951, France 0.9866 (France: 2.5 times more uncertain, 3.60 mean matches per S1 against 3.4).
- **Adversarial validation (26 Sep):** a classifier separates holdout pairs from test pairs (US and India) with AUC 0.8716 using first-stage features. The drivers are the blocking scores (IDF sums computed inside each split's own pool), the competition features (`n_q_for_p` 12.0 against 9.9, `n_cand_q` 35.8 against 39.8) and name-rarity counts (`log_cnt_s1_b` 0.82 against 0.64, because the test has 21% fewer S1). These are set-size artifacts the first stage relies on.
- Remedies being tested: joint name counts over train and test (`pairs.joint_counts`, model `v8`), a first stage that uses pair-level evidence only (`v9`: drops the blocking-score and competition features; the stack recomputes competition from the probabilities), cross-encoder coverage of every shortlisted pair (`s22`). France and per-country thresholds: `reemit.py` variants, and per-country probe files (see section 8).

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
| analysis `ra1` to `ra5` | recall attrition, name ambiguity, ambiguity profile, post-stratification | see `research.md` section 23 | `jobs2/done/ra*.log` |

### 6.2 Running or queued now (26 Sep 13:00)
| Id | Question | Notebook, lane, GPU | Expected |
|---|---|---|---|
| `v9`, `s21` | shift-robust first stage (joint counts, blocking-score and competition features dropped) | main, lane 1, GPU 0 | portal gap closes; holdout about equal |
| `s18` | small cross-encoder on the base model's 966k pairs: size or data? | main, lane 2, GPU 0 (`ru2b`, then `ru3`) | decides whether to go bigger |
| `s17f85`, `f90`, `f95`, `g85` | France and global thresholds on `s17` | main, lane 2 (`rv9`); `s17f85` ready | France behaviour on the portal |
| `v8`, `s19` | joint name counts only | second notebook, lane B, GPU 0 | shift check |
| `s20` | `multilingual-e5-large` (560M) cross-encoder, 1 epoch | second notebook, lane B2, GPU 1 | +0.0005 to +0.001 |
| `s22` | cross-encoder on EVERY shortlisted pair (catch confident look-alike false positives) | second notebook, lane B3, GPUs 2 and 3 | +0.001 to +0.002 |
| `v10`, `s23` | **test-like universe**: drop 21% of the train S1 (outside the sample and holdout) before the competition and count features so the training data has the test's S1 count and distractor rate (`pairs.drop_frac`); separate work directory `work10` | second notebook, lane B4, GPU 0 (`rq0` to `rq4`) | calibration for test conditions; portal gap |
| `s24` | **third cross-encoder from another family**: `Qwen/Qwen3-0.6B` (Apache-2.0, 0.6B) with a one-logit head, fitted on the same 966k pairs as the e5-base model (one epoch, bf16, gradient checkpointing, prompt "Same business? A: ... B: ..." closed by `<|im_end|>`), scored on the v7 band pairs as feature `xs3`; stack `s24` = `s22` + `xs3` | second notebook, lane B3, GPU 2 (train, about 4.6 h from 15:20 IST), then GPUs 2 and 3 (scoring, about 2.5 h), then the stack build (`rn1`, `rn2`, `rn3`) | +0.0003 to +0.0008 over `s22`; ready about 00:00 IST |
| `s25` | `s17` plus address multiplicity features (`stack build --addrmult`: S1 and pool records sharing the exact address, exact address equality) for France's shared addresses | main, lane 1 (`rs1`, tag `_m`) | +0.0003 holdout; more on France |
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

## 10. Ranked plan (26 Sep 15:00, after the recall analysis in `research.md` section 23)
Facts behind the ranking: 76.5% of the missed pairs have an empty pool address and about two thirds of all misses are name-ambiguous (several S1 share the name, the pool record has no address): not recoverable. The reducible recall is about 4,700 pairs (about +0.0028 at most, realistically a quarter of it). Precision loss is 0.0016. The threshold is flat. S1-side attributes explain none of the portal gap (post-stratified estimate 0.9905). Total realistic holdout headroom is +0.001 to +0.002.

| Rank | Idea | Expected holdout gain | Portal hope | Cost | Risk | State |
|---|---|---|---|---|---|---|
| 1 | **Per-country portal probes** (`s17pf` France only, `s17pu` US only, `s17pi` India only; `runs/<name>/output/matching_results.tsv`). Country score F = (portal - (1 - w) * 0.056) / w with w = 0.15, 0.383, 0.468 (public-subset mix is approximate) | 0 | steers the last day | 3 submission slots | none | files ready |
| 2 | Ship **`s22`** (+0.00029, CI excludes 0) | +0.0003 | +0.0003 to +0.0006 | 1 slot | none | ready, `runs/s22/output/` |
| 3 | **`v10`/`s23` test-like universe** and its holdout drop as a measure of the distractor-density effect | -0.001 to +0.0005 | +0.001 to +0.004 if the distractor rate is the cause | running | may hurt the holdout | `rq3` to `rq4` |
| 4 | **More cross-encoder diversity**: `s20` e5-large (running), a Qwen3-0.6B or 1.7B (Apache-2.0) cross-encoder as a third score, order-swap averaging of scores, second seed | +0.0004 to +0.001 | same | 6 to 12 GPU hours | low | `s20` running; Qwen3 not built |
| 5 | **France address-multiplicity features** (S1 sharing the exact address, pool records sharing the address, name margin inside an address group), retrain first stage and stack | +0.0003 (shared-address strata are 5% of holdout S1) | +0.001 or more if France is the gap | 4 to 6 h | moderate, cannot be validated on France | not built |
| 6 | Small bundle: stack seed ensemble, cap 5 S2 and 6 S3, refit the stack with the holdout S1 after the model is chosen | +0.0002 to +0.0004 | same | 2 h | refit has no holdout number | not built |
| 7 | Character n-gram or second dense channel for the 1,600 never-proposed pairs with an address (typos, glued names, non-Latin generic names) | +0.0002 to +0.0004 | same | 6 h+, new index | low | not built |
| 8 | Density-ratio weights for the stack | -0.0002 to +0.0003 | unknown | 3 h | moderate | not built |
| 9 | Pseudo-labels on the test set (France) | unknown | unknown | 6 h | confirmation bias; rules unclear | not recommended before asking the organisers |
| 10 | `v8` joint name counts | about 0 | small | running | low | `rx3` running |
| Not worth doing | more neighbours for empty-address records (`v6`, `s7`), expected-F0.5 decisions (`e1`), second consensus round (`s9`), threshold retuning, S1-attribute reweighting, small cross-encoder alone (`s18`) | | | | | measured, no gain |

