# Handoff v8 (barani): France push on the team's s22 line

Updated 27 Sep 2026 about 03:45 IST. Window closes 27 Sep 23:59 IST. Never name this file `HANDOFF.md` (the Mac disk is case-insensitive: it would be the team's `handoff.md`).

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
