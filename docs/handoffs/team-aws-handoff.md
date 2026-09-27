# Handoff to the team: continue on your own AWS (written 26 Sep 2026, about 22:45 IST)

> **27 Sep 16:00 IST (sai side):** portal best `v8w_s29_AR` 0.985875; new France rules `nsaway` / `coined` and the ready file `v8w_s29_ARtLNC` (estimate about 0.9870). Read `HANDOFF-v8_barani.md` top block and `EXPERIMENTS.md` section 12.

Read this first, then `handoff.md` (infrastructure details, pitfalls), `research.md` sections 23 to 27 (what we measured), `EXPERIMENTS.md` (registry, plans), `submission_checklist.md` (rules, freeze). The window closes **Sun 27 Sep 2026 23:59 IST**; 5 portal submissions per day; one login at a time.

## 1. Where we are

Portal (public subset), all on the same test set:

| File | Portal | What it is |
|---|---|---|
| `s17` | 0.980502 | stack with two cross-encoders on the uncertain band (holdout 0.99025) |
| `s22sx` | 0.982477 | `s22` + France decoy rule (swap with exact-copy support) |
| **`s22t2c`** | **0.984502** | `s22` + type-word swap rule + France threshold 0.985 + caps 5 S2 / 6 S3 (**best so far**) |

`s22` = stack on the first stage `v7` with the e5-base cross-encoder scoring EVERY shortlisted pair (holdout 0.99054). The whole portal gap between holdout (0.990) and portal was **France** (no labels there; US and India transfer, France was about 0.93, now about 0.957). All France work is a decoding step after the model (`src/scripts/france_variants.py`), not a retrained model.

**Our AWS credits are used up** (300 USD; billing lags about 9 hours). Do not restart our notebooks (`test-notebook`, `test-notebook-2`, account 567503593043); a watcher stops them when their queues are empty. Everything you need is copied to the shared bucket (section 3).

## 2. What to do next (in this order)

1. **Submit the France variants** (files in `handoff-nooglers-20260926/runs/<name>/output/matching_results.tsv`; the portal takes only that file; every variant reuses the candidate file of its base, `candidate_pairs.tsv` next to it). Base to beat: `s22t2c` 0.984502. Suggested order for the five slots:
   1. `s22u3` (type swaps + threshold 0.995): is the cut-off higher than 0.985?
   2. `s22v1` (type swaps, spaced-legal aware, + threshold 0.985 that spares equal names after legal spacing and initials pairs): do the protected pairs help?
   3. The combination of what won (build it with the recipe in section 4: e.g. `thrp:0.995`, `thrx`, other thresholds).
   4. The same rules on the best new base if one exists (`s26`, `s22e`, `s27`; see section 5).
   5. The final file; if the portal counts the last upload, make the last slot your best file.
   Other prepared files: `s22u4` (threshold 0.97), `s22u1` (0.985 on non-exact-name pairs only), `s22u2` (0.995 on non-exact-name pairs only), `s22v2` (`thrp` 0.995), `s22v3` (`t2c` with the spaced-legal swap fix), `s22t1c` (type swaps + cap, no threshold), `s22cap` (cap only). Portal differences between two variants on the same public subset are exact (six digits): a category is worth dropping when more than about 26% of its pairs are wrong (`research.md` 25.1).
2. **New bases** (finished overnight, check `runs/` and `work/models/*/holdout.json`): `s26` (symmetric e5-base cross-encoder, both orders, plus `xs_asym`), `s22e` (three XGBoost seeds of the `s22` stack, blended by logit mean), `s27` (two seeds of the symmetric cross-encoder averaged, stack). Accept a new base only if the paired holdout interval against `s22` is positive (`python src/scripts/paired_models.py s22 <name>`); then rerun the France rules on it (section 4) and submit the best rule set once.
3. **Freeze** (leave the last 6 hours): final code zip, official validator with `--check-ids`, methodology document (`submission/Documentation_template.md` still describes older numbers), `reproduce_final.sh` and `README.md` (already describe `s22` + the France decoding; if the final model is `s26`/`s27`, add the symmetric cross-encoder steps: `xenc train ... --set symmetric=1`, scoring of every shortlisted pair with `symmetric=1`, the stack with `--xenc-dir xenc2Fsym_v7 --xenc-dir2 xenc_v7 --xenc-dir3 xenc2F_v7`). Package rules: `submission_checklist.md` sections A, B, D.

## 3. What is where

Shared bucket (you own it): `s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/` (created by us with server-side copies; `EXPORTED_re1.txt` / `EXPORTED_re2.txt` appear when the notebook exports are complete):
- `runs/<name>/output/{matching_results,candidate_pairs}.tsv` for every model and variant (`runs/README` is `EXPERIMENTS.md`), plus `runs/<name>/model` for some.
- `work/`: the complete working directory of the second notebook (prepared parquet, features, first-stage and stack models, cross-encoder models and scores, pair probabilities `output/<model>/pair_p.parquet`, stack chunk folders `stack_p`, `stack_s`, `stack_e`). `work10/`: the test-like-universe experiment.
- `state/`, `ber/`: the exports we used to feed the second notebook and the code sync folder; `set_bucket.sh`, `params_v10.yaml`.
- Raw data and validator: `s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1/` (`dataset/`, `utils/validate_submission.py`).
Code: GitHub `HXMAN76/amazon-ml-2026`, branch `sai` (`git pull`); everything in `code/business_entity_resolution/`.

## 4. Running things on your AWS

- **Machine:** one SageMaker notebook instance in your account, `ml.g5.4xlarge` (1 A10G, 16 vCPU, 64 GB) is enough for decoding and small stacks; for stack training (about 7M pairs x 124 features) use 128 GB RAM or more (`ml.g5.8xlarge`/`12xlarge`); a cross-encoder fit needs one GPU (about 2 hours for the e5-base recipe). Volume at least 250 GB (the work directory is about 60 GB; `blocks` indexes were deleted).
- **Set up:** create the notebook with the lifecycle configuration `aws/notebook/onstart.sh` (job-aware idle auto-stop after 1 hour), copy code to `s3://<your-bucket>/ber/code`, run `bash aws/notebook/set_bucket.sh <your-bucket>` first (it replaces our bucket name in the three scripts), then follow `handoff.md` section 4 (the S3 job queue: put a script in `jobs/pending/`). You can also just open a terminal in Jupyter and run the commands.
- **Get the work directory:** `aws s3 sync s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/work /home/ec2-user/SageMaker/work` and the data from `raw/v1/dataset` into `/home/ec2-user/SageMaker/dataset`; `export BER_DATA=/home/ec2-user/SageMaker/dataset BER_WORK=/home/ec2-user/SageMaker/work PYTHONPATH=/home/ec2-user/SageMaker/ber/src` (`ber` conda env: Python 3.12, `requirements.txt`; the `pytorch` env needs `pip install transformers sentencepiece polars==1.44.2 duckdb==1.5.5 pyyaml scikit-learn xgboost` after every restart). Copy the official validator to `$BER_WORK/official/validate_submission.py` so that `emit` validates every file it writes.
- **Build France variants on a base** (about 4 minutes each, CPU only, needs `work/parquet/test`, `work/output/<base>/pair_p.parquet`, `work/models/<base>/{config,holdout}.json`):
  `python src/scripts/france_variants.py <base> <newname> --rules "typeswap:1.01,thrp:0.985" --cap` writes `work/output/<newname>/matching_results.tsv` and `candidate_pairs.tsv` (validated). Rules (see the file header): `typeswap:pmax`, `swap_exact:pmax`, `swap_all:pmax`, `thr:t`, `thrx:t` (spares exact-name pairs), `thrp:t` (also spares initials pairs and equal names after legal spacing), `weak:T:pmax[:shared]`, `legal:pmax`, `tiny:pmax`; `--cap` keeps at most 5 S2 and 6 S3 per S1 in every country.
- **Blend stacks** (seeds, learners): `python src/scripts/blend_stacks.py <new> <a> <b> [<c>]`.
- **Test-like holdout evaluation, per-country checks:** `src/scripts/` (every script has a docstring with usage): `paired_models.py`, `attrition.py`, `decoy_by_category.py`, `relations.py`, `france_profile*.py`, `poststrat.py`, ...

## 5. What we learned (details in `research.md` 23 to 27)

- Holdout headroom is small (about +0.001 to +0.002): two thirds of the missed recall are name-ambiguous records with an empty pool address; exclusivity, shortlist and threshold lose nothing.
- The portal gap is France: France decoys are pairs whose names differ by one swapped **type word** (`club`, `ecole`, `comite`, ...); they have a constant rate per S1 (about 0.1) independent of how full the S1 is (slot limit 5 S2 / 6 S3), whereas true copies compete for slots. Type words are learned without labels from the slot fit (`france_variants.py`, rule `typeswap`); generic suffix swaps (`services`, `groupe`, `france`) are true noise.
- France probabilities are over-confident up to about 0.985: 46% of the pairs between the model threshold and 0.985 were wrong. The France threshold gave +0.0016.
- Shift for US and India is small (the test-like holdout costs the first stage 0.0014, mismatch part 0.0006): shift-robust models (`s21`, `s23`) and density-ratio weights are not worth it.
- No gain (do not repeat): extra empty-address neighbours (`v6`/`s7`), iterated consensus (`s9`), expected-F0.5 decisions (`e1`), small cross-encoder alone (`s18`), e5-large cross-encoder (`s20`), address multiplicity features (`s25`), joint name counts (`s19`), address/street-mismatch and legal-conflict drop rules.
- Remaining France deficit about 0.03 of France F0.5 (0.0045 overall) is unexplained: hypotheses in `research.md` 26.2.

## 6. Pitfalls that cost us time

`aws s3 sync` skips a same-size changed file: use `--exact-timestamps`; check every job script for a syntax error with `bash -n` and a Python file with `py_compile`; never copy a job script from an old `/tmp` file (we queued an old one by mistake); the AWS CLI login expires after about 5 hours (`aws login`); polars regex has no look-around; `assign_exclusive` breaks ties by lower `q`; the portal accepts only `matching_results.tsv` and shows six digits; keep `--exclude 'work/*'` when syncing code with `--delete`; in job headers set `CUDA_VISIBLE_DEVICES` explicitly.

## 7. Rules to remember

Only the provided data; models MIT/Apache and at most 8B parameters (ours: XGBoost, `multilingual-e5-small` 118M, `-base` 278M, and a saved but unused `Qwen3-0.6B` Apache-2.0); no external lookups; `candidate_pairs.tsv` and its code count in the ranking (smaller sets rank higher: ours has about 4.74 candidates per S1); every S1 exactly one row; matches must be a subset of the candidates.

## 8. Status of the overnight jobs (26 Sep 22:45)

Running or queued on our second notebook, all stopping it when done (about 03:30): `rc1` (symmetric e5-base cross-encoder training), `rc2` (scoring every shortlisted pair, both orders), `rc3` (stack `s26`, paired test against `s22`), `rg10` (stack seeds, `s22e`), `rd1` on our main notebook (second cross-encoder seed, done about 23:30), `rd2` (scores with it, averages the two seeds, stack `s27`), exports `re1`/`re2` to the shared bucket. **Cancelled to save money:** Qwen3-0.6B scoring and its stack `s24` (the fitted model is saved in `work/xenc3Q/model`; scoring takes about 2.5 hours per pair list on one A10G).

## 9. Continuing Qwen on another AWS account (planned)
The fitted Qwen3-0.6B cross-encoder is `work/xenc3Q/model` (in the export). Templates: `aws/jobs/qwen_score.sh` (scores the band pairs of both splits, about 200 pairs/s per A10G, 5 to 8 hours on one GPU) and `aws/jobs/qwen_stack.sh` (stack `s24` = `s22` + the Qwen score, paired test against `s22`; needs the full work directory and 128 GB RAM). Expected gain is small (+0.0001 to +0.0004 over `s22`); accept it only with a positive paired interval, then rebuild the France rules on `s24`.

