# Handoff v8 (barani): France push on the team's s22 line

Updated 27 Sep 2026 about 03:45 IST. Window closes 27 Sep 23:59 IST. Never name this file `HANDOFF.md` (the Mac disk is case-insensitive: it would be the team's `handoff.md`).


## Sai-side update, 27 Sep about 16:00 IST (newest; read first)
- **Portal: `v8w_s29_AR` 0.985875** (best so far; `s29` = `s27` + Qwen3-0.6B in the `xs2` slot, your recipe `typeswap:1.01,thrpn:0.995,protect:0.9,restore:...:0.05` + `--cap`). The team cannot upload right now; the files below are ready for any slot that opens.
- New France error classes read from raw records (full account in `EXPERIMENTS.md` section 12):
  1. **Namesakes in another street.** Exact-name pairs whose streets do not match, on names shared by 6+ France S1 (`bordeaux club sarl`): France 0.87% of predicted pairs vs US 0.11%; 99.5% true on the US/India holdout, so France-only. About 87% decoys. `thrpn` spares exact names and the slot fit cannot see exact-name decoys (they sit in its "sure copies" count k), so the earlier "exact names carry no decoys" was a blind spot, not a measurement. Rule `nsaway:6` in `src/scripts/france_post.py`.
  2. **Coined aliases at the S1's exact address** (`Kelojax`, `Syndelta`; train: `Novizetaumbra`) are about 15% of `thrpn`'s high-p drops and true in train. Rule `coined` (restore, slot caps kept).
  3. `legalx` misfires on about 220 alias records (`X Co formerly known as <S1 name>`: `Co` read as a legal form); `coined` restores them.
- Files (`s3://sagemaker-us-east-1-567503593043/runs/<name>/output/`, checker OK, not uploaded): **`v8w_s29_ARtLNC`** (`s29` + extended typeswap + `legalx` + `nsaway:6` + `coined`; estimate about 0.9870), `v8w_s29_ARtL` (control without the two new rules), `v8w_s29_ARtLN`, `v8w_s29_ARtLNaC`, `v8w_s29_ARtLN2C`, `v8w_s29_ARt`, `v8w_s29_ARtL99`, `v8w_s29_ARtL9`. Run the official validator with `--check-ids` before any upload.
- Negative today: sibling typo fingerprint for namesakes with an empty address (owner closest 34% vs 29% chance); per-country thresholds; stack sweeps; LightGBM; Qwen coverage extension (`s31`, `s31b`); mDeBERTa cross-encoder (non-finite parameters after the first AdamW step in torch 2.10 / transformers 5.17).
- Still open (EXPERIMENTS 12.2): per-class restores inside the `thrpn` cut (about 19k true pairs lost), same-street other-house-number exact-name pairs (14.7k), France blocking recall.
- Infra used for this (account 567503593043, profile `hxman-26`, us-east-1): notebook `test-notebook-2` (stopped; `ml.g5.12xlarge`, 4 x A10G), S3 job lanes `jobsB`, `jobsB2`, `jobsB3`, `jobsB4` under `s3://sagemaker-us-east-1-567503593043/` (put `<job>.sh` in `<lane>/pending/`, logs in `<lane>/live/` and `<lane>/done/`, last line `exit=<code>`), work directory `/home/ec2-user/SageMaker/work` (holds `output/s29`, `models/s29`, all `v8w_*` runs), code prefixes `ber/code_x` (sai branch + these scripts) and `ber/code_v8` (this branch, deployed read-only; job scripts `aws/queue/jobs/sx_*.sh` show both). `test-notebook` (`ml.g5.16xlarge`) belongs to a teammate's Qwen run; do not stop it without asking.

## State
- Team best on the portal: `s22t2c` **0.984502**. The team decides and uploads the 5 slots of 27 Sep; we deliver validated files + a note.
- **Delivered (03:40):** five files, all official-validator PASS **with `--check-ids`**, in `s3://ml-challenge-nooglers/ml-challenge-2026/handoff-nooglers-20260926/runs/<name>/output/`, plus the note `handoff-nooglers-20260926/V8_FILES_barani.md` (same text as `V8_FILES_barani.md` here): `v8_tp985p`, `v8_s22F12n_tpp`, `v8F1s_tpp`, `v8F12n_ts`, `v8_s27_tpp` (suggested order and meaning in the note).
- No portal score for any v8 file yet. Honest outlook: +0.001 to +0.002 together (0.9855 to 0.9865); 0.990 would need France about 0.987.
- User decisions: team picks uploads; `barani-v5` may run through Sunday; commits on local branch `v8/france` (this worktree), **no push**; no test data for training (official README: "using only the provided training data").

## Key measurements (label-free unless stated)
- Holdout (labelled): s22 0.990540; s22e -0.000007, s26 -0.000012 (CIs across 0); s27 +0.000094 [+0.000004, +0.000188].
- t2c's plain threshold 0.985 drops 63k France pairs: non-exact 33k are about 50%/43% decoys (US 0%: `thrx`/`thrp` fired sets), exact-name 30k only about 19% (implied) -> `thrp` (spares equal-after-legal-spacing names and initials) beats `thr`. Samples confirm (`clinique pascal` / `clinique pascal sasu` same address dropped by t2c).
- t2c empties 3,267 France S1 (empty share 5.0% -> 6.3%; US/India 5.8%; training 5.6%); `protect:0.9` restores 1,482 (5.7%). The slot fit cannot judge protect's restored pairs (all at k = 0: the "0.0" is an artefact).
- France-aware cross-encoder 1 (`xencFR/model`, 36 min, 1 epoch, warm start from `xenc2/model`; 500k replay + 50k initials + 32k spelled legal + 50k glued positives, 150k one-word-swap negatives, train records only): France type swaps below 0.5: 90% (old 19%); initials 0.04% (old 5%); noise-word swaps 63% (old 25%: a flaw, fixed by keeping the old score for swaps into fils/groupe/services/developpement); exact names 8.5% (old 3.7%). Holdout AP equal (0.99929 vs 0.99932). `xfr_slots.py`: its extra rejections are decoy-rich only among one-word swaps (83-85%), not among "other" pairs.
- Model 2 (`xencFR2/model`) adds 100k "other tenant" negatives. Bases built by swapping France's xs and rebuilding the s22 stack for France only (US/India unchanged): `s22F1n`, `s22F2n`, `s22F12n` (mean logit), `s22F1s` (new score only on one-word swaps). On them `typeswap` fires on 3.5-6k pairs (s22: 25k) and `thrp` still drops 24-25k pairs at 44-46%/38-39% decoy share.



## Morning of 27 Sep (after the portal readings)
- Branch `v8/france` is checked out in the main folder `/Users/barani200/amazon-ml-2026` and **pushed after every commit** (user request, about 06:10).
  The `-v8` worktree was switched to `sai` by the user: never commit there.
- Portal: `v8u_s27_AR` **0.985578** (best); `v8u_s22F12n_AR` 0.984136 (France cross-encoders inside the stack cost about 0.009 of France F0.5:
  the roughly 31k France pairs they removed were essentially all true). The cross-encoder route is dropped; everything builds on s27.
- Label-free checks that came back negative or neutral (do not repeat): dropping the swaps France still predicts (88% are noise-word copies:
  fils, services, developpement, groupe, france), `exactfar` (equal names at another address: France's decoy share is below the US reference),
  `thrpk` (the extra protected kinds are about 34% decoys by the slot fit, above break-even; its `alias` option about 60%), per-country
  thresholds for US and India (the single 0.74 is already best), stack blends, a restore driven by the cross-encoder score.
- France's match-count profile per S1 equals US/India's (profile_counts.py) except about 1% more S1 with one match, mostly confident exact names
  (single_pair.py): the remaining France error is in which records are matched, not how many; no label-free lever left there.
- Running: Qwen3-0.6B scoring of 2.46M test + 2.13M train band pairs (about 200 pairs/s; done about 12:00), then `v8n3_s28` (s28 = s27's stack +
  the Qwen score as xs4, paired test against s27, France recipe, validation, upload of `v8u_s28_AR`; about 13:30 to 14:00).
- Final package: `bash submission/make_package.sh <RUN>` (zip in the organisers' layout); methodology draft updated in `submission/Documentation_template.md`.

## s28 (27 Sep 09:40)
- Tried after s28 (holdout, paired against s28, all intervals include 0; not used): s28b deeper/slower stack +0.000041, blend s28+s28b +0.000026, blend s28+s27 -0.000038. s28 is the final model.
- `s28` = s27's stack + Qwen3-0.6B score as xs4: holdout 0.990770, paired +0.000135 [+0.000047, +0.000226] over s27. `v8u_s28_AR` validated (--check-ids), in the team folder and at `output/v8u_s28_AR/` on the laptop. Next upload suggested.

## Portal readings
- 27 Sep about 06:00: **`v8u_s27_AR` 0.985578** (+0.001076 over `s22t2c` 0.984502).
- About 06:15: **`v8u_s22F12n_AR` 0.984136**: the France cross-encoders in the stack hurt (about -0.009 France F0.5). Drop that route; build on s27. Next: `v8k_s27_KA` (thrpk).

## Overnight work after 03:45 (user asleep; target 0.990 / top 50)
- Findings: (1) France's noise-word swaps (fils, groupe, services, developpement) are true copies the model scores low: `thrpn` (thrp that also
  spares them) raises the decoy share of what it drops from 48% to 54% (the spared 7.8k pairs are about 16% decoys); 8.6k unowned noise-swap records
  sit at their S1's address with p about 0.16 (restore candidates). (2) Exact-name restores are bad (holdout: 0.4% true). (3) The legal-form +
  house-number "sibling" idea is wrong (holdout: 99.9% true, France rate below the holdout's). (4) Stack blends (s22+s27, +s26+s22e) do not beat s27.
  (5) Confident non-exact France pairs look like true copies (glued, coined aliases, d b a) in full and free S1 alike: no hidden decoy pool there.
- New rules in `france_variants.py`: `thrpn:t`, `restore:KINDS:pmin` (uses `france_recall.restore_candidates`), `legalhouse` (not useful).
- Queue at 04:10 (GPU lane `jobs`, in order): `v8m3_xfr3` (cross-encoder 3: full replay + noise-swap positives), `v8m3a0_kill` (stops the CPU-lane
  job `v8z9_s24` that blocks it), `v8m3a_r3` (recipes `v8s_<base>_{A,AR,AR2}` = typeswap + thrpn 0.995 + protect [+ restore of noise swaps,
  initials, spelled legal, glued at the S1's address, p >= 0.05 / 0.2] on s27 and s22F12n), `v8m3q_qwen` (score the team's Qwen3-0.6B
  cross-encoder, both splits), `v8m3r_s24` (stack s24 = s22 + xs3, paired tests, recipes `v8r_s24_{A,AR}`), `v8m4_xfr4` (seed 2 of model 3),
  `v8m5_avg34` (mean of 3 and 4 in the stack), `v8m9_deliver` (validator --check-ids, upload to `s3://sagemaker-us-east-1-645311222213/ber/v8/runs/`).
- Candidate for the team today (if the portal confirms the direction): `v8s_s27_AR` (s27 + typeswap + thrpn 0.995 + protect + restore), or its s24
  version if s24 beats s27 on the holdout. Copy from our bucket to the team folder with the laptop (the notebook role cannot write there).

## 27 Sep 11:30 to 13:00: where s28 still loses (all measured, nothing shipped)
- US/India threshold curve (s28 holdout): 0.66 0.990709, 0.70 0.990731, **0.72 0.990770**, 0.75 0.990761, 0.80 0.990719, 0.85 0.990619, 0.90 0.990362.
  `v8u_s28_AR_t80` (US/India at 0.80) is validated in `ber/v8/runs/` but not worth a slot (world A: expected -0.00004).
- Sub-group miscalibration search (`subgroups.py s28`): one drop group, holdout -0.000010 [-0.000041, +0.000015]; nothing to restore.
- **Holdout error budget (`v8e4_budget`)**: 518,468 true pairs, 511,015 proposed (pair recall 0.9856); 505,396 kept, only 603 wrong
  (precision 0.9988); 6,222 true candidates not kept. Oracle gains: no wrong pair +0.0011; every true candidate kept +0.0039; no blocking miss
  +0.0043. US/India lose on recall, not precision.
- **Lost true pairs by kind (`misses.py s28`)**: below threshold 5.9k (exact core name 3.3k at mean p 0.36, mostly **empty pool address**:
  namesakes), blocking misses 7.5k (shared word + other/empty address 3.8k, exact core 2.3k, coined 0.9k, non-Latin 0.3k), lost to another S1 0.3k.
  Mostly inherent ambiguity (an empty-address record with a generic name could belong to any namesake).
- **France bands above the 0.995 cut (`--dry`, new rule option `thrpn:t:tmin`, rule `kind:KIND:SRC:hi[:lo]`)**: slot fit reads 0.2 to 0.55
  decoys in [0.995, 0.9999), but the same fit on the US and India is biased by up to 0.4 to 1.0 for single kinds (US S3 coined names 0.41 to 0.72,
  India S2 swaps 0.96, all about 99.9% true on the holdout). **The slot fit cannot decide a stricter French cut; no portal slot spent on it.**

- US/India loss by kind: the namesake groups (exact name, empty pool address) are calibrated (true share = mean p in every group, `v8e9_namesake`).
  Rank-dependent threshold (first pick 0.5, others 0.76): OOF +0.000042, holdout +0.000045 [-0.000048, +0.000142]; not used.
- France assignment (`v8f1_assigned`): pairs per S1 train truth 3.46; test US 3.39, India 3.37; France s28 raw 3.47, **v8u_s28_AR 3.31**. France now
  predicts fewer pairs than the US/India level with lower precision: both about 25k wrong and about 40k missing French pairs. Recovering true pairs is
  worth only +0.0022 France F0.5 per 1% of France's pairs; removing wrong ones +0.0063. Unclaimed same-address "other" candidates with p >= 0.3:
  France 35k against US 2.1k (`v8f2_recall`), mostly type-word siblings (decoys).
- Rule `typeconf:pmax[:rmin]` (a learned type word replaced, other words may change too; `--samples N` prints examples): 22k France pairs but only
  190 not already dropped by typeswap + thrpn 0.995 (rmin 0.5 pulls in the noise words: wrong). `france` swaps/additions behave like the noise words
  (typeswap ratio 0.50 like fils/groupe/services) but protecting them is worth about +0.00006. France levers are exhausted label-free.
- 13:00 to 14:15 (all measured): recall rescue of exact-name empty-address records outside the shortlist fails (holdout -0.000036: 170k such pairs
  are generic names, 1.2% true); Qwen xs4 cannot flag French decoys (scores 9% of France pairs, 2.3k of 27k typeswap pairs below 0.5); generator
  suffix words (US southside/eastgate/... are 0% owned in train; France participations/holding/distribution/international) are already left
  unclaimed (France 0.6%); country labels are clean. **Extended typeswap** (`typeswap:1.01:0.6:30:300`, 43 words): +1,382 France drops at 0.86 / 0.91
  decoy share -> `v8u_s28_ARt` (validated, team folder). The teammate's s29 (Qwen in the xs2 slot, holdout 0.99088, +0.00025 over s27) is the best base.
- **Portal test files (validated, team folder and `output/` on the laptop)**: `v8u_s28_AR9` (France thrpn 0.9999) and `v8u_s28_AR99` (0.999).

## Infrastructure (our AWS, profile `barani`, account 645311222213)
- Notebook `barani-v5` (ml.g5.16xlarge), two-lane runner: `jobs/` (GPU), `jobs2/` (CPU); lanes read their pending list once per loop (a job queued later waits until the lane's list is done). Laptop tool from this worktree: `smssh-venv/bin/python aws/sm/sm.py --profile barani {publish|enqueue <job> --queue jobs|jobs2|jobs|nb start|stop|status}`; read logs with `aws s3 cp s3://sagemaker-us-east-1-645311222213/<lane>/done/<job>.log -` (sm.py jlog output is hard to grep).
- The notebook role cannot read the team bucket; the laptop user can: team files are copied server-side into `s3://sagemaker-us-east-1-645311222213/ber/team_work/`, then synced to `/home/ec2-user/SageMaker/work_t` (`BER_WORK` of every v8 job; our v6 `work/` untouched). Delivery: `v8w_deliver.sh` template (validator --check-ids, then `ber/v8/runs/<name>/output/`), then a laptop server-side copy to the team folder.
- A GPU-lane job must never wait on a CPU-lane output (deadlock).

## Code (branch v8/france, local commits)
`src/ber/stages/xenc_fr.py` (+ `src/tests/test_xenc_fr.py`), `stack.py --ctry`, scripts `france_empty`, `emptied_samples`, `xfr_report`, `xfr_slots`, `xs_merge` (`--keep-old-noise`, `--only-swaps`), `stack_predict_merge`, `france_variants` (+ `xfr`, `protect`, `typeins`, `legalhouse`, per-rule slot-fit decoy share), `aws/queue` two-lane runner, `aws/sm --queue`, jobs `aws/queue/jobs/v8*.sh`, note `V8_FILES_barani.md`.

## Next
1. Wait for the team's portal readings; `v8u_follow` pre-builds `thrp` 0.99 / 0.995 (+protect) on s22 and s22F12n. Deliver more files only on request (template `v8w_deliver.sh`).
2. Freeze help: `reproduce_final.sh` / README / methodology for the chosen file (for a cross-encoder file: `xenc_fr`, `xenc train/score`, `xs_merge`, `stack build --ctry france`, `stack_predict_merge`, `france_variants`).
3. Stop `barani-v5` when done (it also auto-stops after 1 h idle).
