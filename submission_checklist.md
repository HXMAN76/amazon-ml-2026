# Submission checklist (from the problem statement and the guidelines PDFs)

Written 2026-09-25. Sources: `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` and `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`. Status is what the repo does today; "owner" says who has to act. Update this file as items close.

## A. Leaderboard file: `matching_results.tsv`

| Requirement | Status | Owner |
|---|---|---|
| Tab-separated, exact columns `source1_entity_id`, `matched_entity_ids` | Done (checked by official validator header rule) | done |
| Exactly one row per test S1 entity (1,732,544 rows), no duplicate S1 rows | Done (1,732,544 rows written from `test_source1`) | done |
| Empty `matched_entity_ids` for singletons | **Bug found and fixed**: empty lists were written as the two characters `""`, which the official validator rejects. Writer now uses `quote_style="never"`; rerun `v0e` | fixed, verify on `v0e` log |
| No duplicate IDs inside a list | Done by construction (one row per (S1, pool id) after grouping) | done |
| IDs are S2-/S3- only, exist in the test set, no self-matches | Built from the test Parquet only; **run the official validator with `--check-ids`** (job `v0f`) | in progress |
| Final matches are a subset of candidates | Verified on the v0 output: 0 violations once the `""` issue is removed | done |
| Format: UTF-8, plain TSV | Done | done |
| Official validator prints PASS before every upload | Not yet PASS (see first item) | pending `v0e` and `v0f` |
| Upload in the portal, expect status `SCORED` | Human action, from one laptop or desktop only (see D) | human |

## B. Final package: `<team_name>_submission.zip`

```
<team_name>_submission.zip
├── output/matching_results.tsv        # same file as the leaderboard upload
├── output/candidate_pairs.tsv         # the candidate set the model ran inference on
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md          # filled in (a .pdf export is also allowed)
```

| Requirement | Status | Owner |
|---|---|---|
| `candidate_pairs.tsv` is the exact set fed to the model; every matched ID appears in it | Done for v0 (all 51.9M scored pairs are the candidates) | done |
| Candidate file size | **Risk:** 691 MB raw (95 MB matching). Portal or zip size limits are unknown; job `v0f` reports the gzip sizes. If too large, lower K or keep only candidates above a low probability (must remain the exact set the model scored, so the model would also have to be rerun on that set) | ask organisers if a limit exists |
| Code self-contained and runnable; anyone can regenerate both outputs from the training and test data | **Not done:** need one entry point (`make reproduce` or a script) that runs prepare, sample, block, pairs, train, predict; README must state exact steps and expected runtime | agent |
| All source under `src/`, plus `README.md` and pinned `requirements.txt` | `src/ber` exists; `configs/`, `Makefile`, `tests/`, `scripts/` sit beside `src/` (allowed as long as the README explains it). Legacy v0 modules (`blocking.py`, `features.py`, `model.py`, `decide.py`, `run.py`, `normalize.py`, `data.py`) are unused and should be removed from the zip to avoid confusion | agent |
| Source code has proper comments describing the functions | Mostly docstrings on stages; review before packaging | agent |
| Methodology document filled from `Documentation_template.md` (template is in S3 `docs/`) | **Not done.** Sections needed: executive summary, problem analysis, solution strategy, blocking (keys, number of candidate pairs, how true matches were kept), matching model (features, model, threshold method), results and error analysis, conclusion, appendix (code structure and entry points) | agent drafts, team reviews |
| Guidelines say a 1 to 2 page document; the statement says no page limit | Keep the main body about 2 pages and move detail to the appendix | agent |
| Team name, members, date for the template and zip name | **Missing** | human |

## C. Rules that can disqualify

| Rule | How we comply | Note |
|---|---|---|
| No external lookup of business identities: no commercial ER APIs, no government registries, no geocoding APIs, no internet data augmentation | Pipeline uses only the provided TSVs. Rules (abbreviations, legal forms) are hand-written in `text.py`, not fetched | State this in the methodology document, including that no pretrained model was used in v0 |
| Final model MIT or Apache-2.0 and at most 8B parameters | v0 uses XGBoost (Apache-2.0); LightGBM (MIT) only in legacy code; no neural model yet | List every library and any pretrained model with licence in the document; later models (bge-m3 MIT, multilingual-e5 MIT, Qwen3 Apache-2.0) must be logged the same way |
| Top teams' packages are reviewed in detail; results must reproduce | Needs the reproduce entry point (section B) and pinned versions | agent |
| Multiple registrations or IDs: instant disqualification | Team members use one registration each | humans, confirm |

## D. Operating limits and portal rules

| Rule | What it means for us |
|---|---|
| Challenge window 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST | Freeze the final run well before the end; leave the last 6 hours for the package and re-validation |
| Maximum **5 submissions per day**, 3 days | Plan uploads (v0 first, then only changes that beat the holdout); do not tune on the public leaderboard, which uses a subset of test. The private leaderboard decides the final ranking |
| Maintain the version history of all submissions | `work/runs/runs.jsonl` and MLflow log every run; record the leaderboard score and file hash next to each upload |
| Desktop or laptop only, no mobile; **no simultaneous logins** (one device per participant at a time, otherwise the challenge can be terminated) | Only one person uploads at a time from one device; never keep two sessions open |
| Portal uploads are done by a human | The agent never automates portal actions; it only prepares and validates files |
| Queries go through the organisers' Google Form; technical problems go to support@unstop.com with a screenshot and registered email | Human |
| Top 100 teams later submit methodology, candidate generation strategy, model architecture and features, and possibly the final source code | Keep the code and document consistent with the final run |

## E. Open items to close now

1. Team name, team member names and submission date for the zip name and the template (human).
2. Confirm the official validator PASS on `v0e`/`v0f`, then upload `matching_results.tsv` (human, one device).
3. Reproduce entry point, README, cleanup of legacy modules, licence table (agent).
4. Draft the methodology document with real numbers: OOF macro F0.5 0.9377 on the 250k train sample (not a holdout), blocking recall 0.9416, test candidate pairs 51,892,359, matches per S1 3.24 (train truth 3.46) (agent; update after the holdout run).
5. Ask whether the portal or organisers limit upload size (matching 95 MB, candidates 691 MB raw) (human).
