# Handoff: v4 architecture work (Roshan's branch)

Written 2026-09-25, ~20:10 IST, branch `roshan`, repo root `H:\mlchallenge\amazon-ml-2026`. Audience:
anyone continuing this work without the chat history. Read this first, then
[ARCHITECTURE_v3_final.md](ARCHITECTURE_v3_final.md) (what's shipped), [ARCHITECTURE_v4.md](ARCHITECTURE_v4.md)
(what's being built), [ARCHITECTURE_v3_research.md](ARCHITECTURE_v3_research.md) (literature/reasoning
behind both). Nothing here contains secrets; never add credentials to the repo.

## 1. State in brief

- **v3 is done and shippable.** `stack0` (base 63 features + `v0base`'s p1 + fine-tuned cross-encoder's
  `xs`) beats `v0base`: OOF macro F0.5 **0.9643** vs 0.9548, paired-bootstrap 95% CI **[+0.0091, +0.0096]**
  (ship=True). Submission files generated and validated **PASS**, including `--check-ids`, at
  `roshan-notebook:/opt/ml/input/data/notebook/work/output/stack0/{matching_results.tsv,candidate_pairs.tsv}`
  — copy these to your machine and upload; nothing else needed for a valid submission today.
- **v4 is code-complete, execution in progress, currently blocked on infra (SSM connectivity), not on
  code or ideas.** Two independent jobs were launched on `roshan-notebook` (not the separate
  `roshan-v4-notebook` — see §6) around 2026-09-25 13:23-13:25 UTC (18:53-18:55 IST):
  1. `prepare → block --all-train → audit_recall` (v4 build-order step 1: blocking-recall audit)
  2. `train_xenc2 → score_xenc → train_stack` (v4 build-order step 3: cross-encoder v2, independent
     of the audit, gated against v0base/stack0)
  Both were confirmed running (`ps aux` showed live processes) shortly after launch. Since ~18:59:57
  IST the SSM session to the container (`mi-0e95744a56a8c9ac2`, `roshan-notebook`'s bootstrap
  container) has shown `ConnectionLost` and has not recovered on repeated retries — **this affects
  every SSM-managed instance in the account simultaneously** (checked via `aws ssm
  describe-instance-information`), so it is not specific to this container; general AWS API access
  (S3) works fine throughout. **No destructive action (stop/restart/rebootstrap) has been taken** —
  the jobs may still be alive on the notebook, killing the SSM link would not kill them, but I cannot
  currently see their output to confirm progress or completion.
- **Next person's first job**: get SSH access back (see §5), check
  `/opt/ml/input/data/notebook/v4_audit_chain.log` and `/opt/ml/input/data/notebook/v4_xenc2_chain.log`
  on `roshan-notebook`. If either finished, read the result and act per the gate in
  [ARCHITECTURE_v4.md](ARCHITECTURE_v4.md) (ship only if paired-bootstrap 95% CI lower bound > 0 vs.
  the current best). If a job died, just relaunch it — nothing is lost except GPU time, both are
  idempotent (`--name` outputs go to fresh `WORK/models/<name>/`).

## 2. Access and identity

| Item | Value |
|---|---|
| AWS account | `323170157217` (own account, separate from teammates') |
| Region | `us-east-1` |
| CLI profiles | `default` and `amazon-ml` (both work; commands in this session used `--profile amazon-ml`). Verify: `aws sts get-caller-identity --profile amazon-ml` |
| SSH tooling | `smssh-venv/` (Python venv at repo root) has `sagemaker-ssh-helper` installed; `smssh-venv/Scripts/sm-local-ssh-notebook` and friends are the CLI wrappers |
| SSH key | `~/.ssh/sagemaker-ssh-gw` (auto-generated/rotated by the helper on each connect) |
| IAM role | `sagemaker-competition-notebook-execution-role` (same role used for all notebooks this session) |

Notebooks (check current state — this changes):

| Name | Instance type | Volume | Status as of writing | Purpose |
|---|---|---|---|---|
| `roshan-notebook` | `ml.g6.xlarge` (L4, 24GB VRAM) | 100GB | `InService`, SSM container `ConnectionLost` | **Primary compute.** Has v3's full `work/` (all models), v4's isolated `work_v4/`, code, dataset already synced |
| `roshan-v4-notebook` | `ml.g5.xlarge` (A10G, 23GB VRAM) | 200GB (resized this session, was 100) | `InService`, SSM container `ConnectionLost` | **Abandoned mid-setup** — bootstrapping this as a separate v4 instance hit repeated infra issues (see §7); v4 work was moved to `roshan-notebook` instead with an isolated `BER_WORK` dir. This notebook has no code/dataset synced and is currently just idle and billing — **stop it if not resuming this approach** (`aws sagemaker stop-notebook-instance --notebook-instance-name roshan-v4-notebook --profile amazon-ml --region us-east-1`) |
| `barani-notebook` | was `ml.g5.xlarge` | — | **deleted this session** (teammate's, was Stopped, user confirmed deletion) | n/a |
| `summa` | `ml.t3.medium` | — | Stopped, untouched, not ours | n/a |

Connect to `roshan-notebook` (the one that matters):

```bash
cd H:\mlchallenge\amazon-ml-2026
source smssh-venv/Scripts/activate    # or smssh-venv/Scripts/activate.bat on plain cmd
export AWS_PROFILE=amazon-ml
# Use `yes |` to keep stdin open — a plain call gets EOF (no tty) and the whole SSH
# session (and its -L port forward) dies within seconds. This is not optional.
yes | sm-local-ssh-notebook connect-notebook roshan-notebook   # backgrounds a tunnel on localhost:17022
# wait ~10-20s, then:
ssh -p 17022 -i ~/.ssh/sagemaker-ssh-gw -o StrictHostKeyChecking=accept-new root@localhost 'echo alive'
```

If `sm-connect-ssh-proxy` reports `Instance status: ConnectionLost` / `Error: Instance is offline`,
the SSM session itself is down (see §7 for what this looked like this session) — retry after a few
minutes; check `aws ssm describe-instance-information --profile amazon-ml --region us-east-1` for
`PingStatus` across ALL listed instances to see if it's a broader issue (it was, this session) before
assuming the specific notebook died.

Notebook control:

```bash
export AWS_PROFILE=amazon-ml
aws sagemaker describe-notebook-instance --notebook-instance-name roshan-notebook --region us-east-1 --query NotebookInstanceStatus --output text
aws sagemaker stop-notebook-instance  --notebook-instance-name roshan-notebook --region us-east-1
aws sagemaker start-notebook-instance --notebook-instance-name roshan-notebook --region us-east-1
```

Cost: roughly similar per-hour-while-InService billing as any g6/g5.xlarge notebook; both
`roshan-notebook` and `roshan-v4-notebook` are currently `InService` and billing — stop whichever
isn't in active use.

## 3. Directory map

On `roshan-notebook` (`/opt/ml/input/data/notebook`, the persistent data mount inside the bootstrap
container):

| Path | Content |
|---|---|
| `dataset/{train,test}/*.tsv` | raw competition data |
| (code lives at `/root/ber/code` inside the container, synced from S3 — see §4) | |
| `work/` | **v3's** WORK dir: `parquet/`, `blocks/`, `features/`, `models/{v0base,xenc0,stack0}/`, `output/{v0base,stack0}/`, `v2/xenc0/` |
| `work_v4/` | **v4's** isolated WORK dir (fresh this session, only has whatever the audit chain produced before the SSM link died — check `parquet/`, `blocks/` there) |
| conda env `ber` | Python (matches `requirements.txt`, includes torch 2.7.1, sentence-transformers 5.1.1, faiss-cpu 1.9.0, xgboost, polars, duckdb, etc.) |

In the repo (`H:\mlchallenge\amazon-ml-2026`):

| Path | Content |
|---|---|
| `code/business_entity_resolution/` | package `ber` (`src/ber`), `requirements.txt`, `README.md` |
| `code/business_entity_resolution/src/ber/stages/` | all pipeline stages, including v4's new ones: `audit_recall.py`, `train_xenc2.py`, `train_blockenc.py`, `score_blockenc.py` |
| `ARCHITECTURE_v3_research.md` | literature research behind both v3 and v4 (Ditto, Sudowoodo, SC-Block, CoSiNES, Siamese-GCN, compliant model shortlist) |
| `ARCHITECTURE_v3_final.md` | v3's shipped pipeline, decided (not options), with reasoning for what was cut |
| `ARCHITECTURE_v4.md` | v4's plan — **read this in full before continuing**, it has the exact build order, gates, and commands for every stage |
| `aws/notebook/bootstrap_ssh.py` | script that bootstraps SSH on a fresh notebook (also at `s3://sagemaker-us-east-1-323170157217/ber/aws-notebook/bootstrap_ssh.py`) |
| `smssh-venv/` | local Python venv with `sagemaker-ssh-helper` for SSH access |

S3 layout (`s3://sagemaker-us-east-1-323170157217/`):

| Path | Content |
|---|---|
| `ber/code/` | synced copy of `code/business_entity_resolution/` — **always sync after committing** (see §4) |
| `ber/aws-notebook/bootstrap_ssh.py` | for bootstrapping a fresh notebook's SSH (used for `roshan-v4-notebook`, see §7) |
| `ber/aws-notebook/v4-*.log` | diagnostic logs from `roshan-v4-notebook`'s lifecycle-config bootstrap attempts (stale now, that approach was abandoned) |
| `ssh-authorized-keys/` | managed by sagemaker-ssh-helper, don't touch |

## 4. Deploy workflow (code change → running on the notebook)

No code zip needed for submission (per latest competition instructions — only
`matching_results.tsv` + `candidate_pairs.tsv` are required). Deploy flow is simple, no
versioning/archiving step:

```bash
# 1. edit code locally
# 2. commit as "Roshan" (never a Claude/agent git identity)
git add <files> && git commit -m "..."
# 3. sync to S3
cd H:\mlchallenge\amazon-ml-2026
aws s3 sync code/business_entity_resolution s3://sagemaker-us-east-1-323170157217/ber/code --exclude "*.pyc" --exclude "__pycache__/*" --profile amazon-ml --only-show-errors
# 4. pull onto the container (over the SSH tunnel from §2)
ssh -p 17022 -i ~/.ssh/sagemaker-ssh-gw root@localhost 'aws s3 sync s3://sagemaker-us-east-1-323170157217/ber/code /root/ber/code --exclude "*.pyc" --exclude "__pycache__/*" --only-show-errors'
```

**Do not skip step 4** — a stage script edited locally and pushed to S3 does nothing until pulled
onto the container; this caused at least one wasted job run this session (`ModuleNotFoundError` for
a stage that existed locally/on S3 but not yet on the container).

Launching a job (always via `setsid ... & disown` over SSH so it survives the SSH session ending,
and always set both `BER_WORK` and `BER_DATA` explicitly — the default for `BER_DATA` is a relative
`"dataset"` path that silently resolves wrong depending on cwd):

```bash
ssh -p 17022 -i ~/.ssh/sagemaker-ssh-gw root@localhost '
unset PYTHONPATH; export PYTHONPATH=/root/ber/code/src
export BER_WORK=/opt/ml/input/data/notebook/work_v4    # or .../work for v3-namespace runs
export BER_DATA=/opt/ml/input/data/notebook/dataset
source /opt/conda/etc/profile.d/conda.sh; conda activate ber
cd /root/ber/code
setsid python -m ber.stages.<stage> --name <name> ... > /opt/ml/input/data/notebook/<logname>.log 2>&1 < /dev/null &
disown
'
```

Chain multiple stages in one launch with `setsid bash -c "cmd1 && cmd2 && cmd3"` (used for both v4
jobs this session) so a later stage doesn't start until the earlier one succeeds.

## 5. v4 pipeline: exact commands and current state

Full design, reasoning, and gates: [ARCHITECTURE_v4.md](ARCHITECTURE_v4.md). Summary of the build
order and what's been run:

| Step | Command | Status |
|---|---|---|
| 1. Blocking-recall audit | `prepare --split train && block --split train --all-train && audit_recall` (BER_WORK=`work_v4`) | **Launched ~13:23 UTC, result unknown** — SSM link died before completion could be confirmed. Log: `/opt/ml/input/data/notebook/v4_audit_chain.log` |
| 2. Dense blocking (`blockenc0`) | `train_blockenc.py` → `score_blockenc.py` (k-sweep) | **Not started** — strictly conditional on step 1 showing a material recall gap. Do not build until step 1's result is known and reviewed |
| 3. Cross-encoder v2 (`xenc2`) | `train_xenc2 --baseline v0base && score_xenc --name xenc2 --split train --pair-p work/models/v0base/oof.parquet && train_stack --name stack_xenc2 --baseline v0base --xenc-name xenc2` (BER_WORK=`work`, v3's namespace) | **Launched ~13:25 UTC, result unknown** — same SSM issue. Log: `/opt/ml/input/data/notebook/v4_xenc2_chain.log`. This one is independent of step 1/2 — check it regardless of the audit outcome |
| 4. Rebuild base model on unioned candidates (only if step 2 ships) | `pairs.py` (both splits) → `train_gpu --name v4base` → `train_xenc2 --name xenc2b --baseline v4base` → ... → `train_stack --name stack_v4` | Not started, conditional |

**First thing to do**: reconnect SSH (§2), then:
```bash
ssh -p 17022 -i ~/.ssh/sagemaker-ssh-gw root@localhost 'cat /opt/ml/input/data/notebook/v4_audit_chain.log; echo ---; cat /opt/ml/input/data/notebook/v4_xenc2_chain.log; echo ---; ps aux | grep -E "stages\.(prepare|block|audit|train_xenc2|score_xenc|train_stack)" | grep -v grep'
```
If a job's log shows it finished, read the result. If a job died partway (no matching process, log
ends mid-stage with no error), just relaunch that stage — everything is idempotent by `--name`.

`audit_recall.py`'s output (once you have it) directly decides whether step 2 (dense blocking) is
worth building — see the gate wording in [ARCHITECTURE_v4.md](ARCHITECTURE_v4.md) §"New piece 1".

## 6. Why v4 isn't on its own notebook

Originally tried standing up `roshan-v4-notebook` as a separate g5.xlarge instance so v4 could run
in parallel with anything on `roshan-notebook`. This hit a chain of infra problems (detailed in §7)
that consumed a lot of session time without a working result, so the decision was made (with user
confirmation) to abandon that and run v4 on `roshan-notebook` instead, using a separate `BER_WORK`
(`work_v4/`) so it doesn't touch v3's already-shipped `work/`. If someone wants to revive the
separate-notebook approach, the lifecycle config `roshan-v4-ssh-bootstrap` is still attached and
mostly working (it gets as far as pulling the training image and starting the SSH container) — the
last known failure was `ModuleNotFoundError: sagemaker_ssh_helper` from a flaky conda-env-detection
line, which was fixed by pinning to a dedicated `sshboot` conda env in the on-start script (see the
lifecycle config content via `aws sagemaker describe-notebook-instance-lifecycle-config
--notebook-instance-lifecycle-config-name roshan-v4-ssh-bootstrap`), but this was never verified
end-to-end before the SSM outage hit and the approach was dropped.

## 7. Known pitfalls this session (each cost real time)

1. **`sm-connect-ssh-proxy` can't take a trailing remote command.** Its arg-passing puts any extra
   args *before* the hostname in the final `ssh` invocation, so `connect-notebook <name> sleep 999999`
   gets parsed as `ssh -L ... sleep 999999 <instance-id>` — `sleep` becomes the SSH *hostname*, and
   the instance ID becomes a remote command argument. Symptom: `Warning: Permanently added 'sleep'`
   then `bash: 999999: command not found`. **Fix**: don't pass a trailing command at all — use
   `yes | sm-local-ssh-notebook connect-notebook <name>` to keep stdin open instead (see §2).
2. **A plain (no-`yes`) interactive SSH session dies in seconds when launched non-interactively** (no
   tty, stdin is `/dev/null`): the remote login shell gets immediate EOF on stdin and exits, tearing
   down the `-L` port forward with it. This looks like "the tunnel keeps dropping" but is actually
   "the tunnel was never going to survive this launch method." Always use `yes |`.
3. **`BER_DATA` must be set explicitly alongside `BER_WORK`.** It defaults to a relative `"dataset"`
   path, which resolves against whatever the current working directory happens to be at call time —
   this caused a validator run to fail with `Test source1 file not found: dataset/test/...` even
   though the submission files themselves were correct; only the validator's own subprocess call had
   the wrong cwd.
4. **`predict.py` didn't originally support stacked models** (those needing `p1`/`xs` joined in
   before scoring) — it only handled plain pointwise models. Extended it to detect `"baseline"` in
   the model's `config.json` and join `p1`/`xs` the same way `train_stack.py` does, rather than
   writing a separate predict script.
5. **XGBoost stage-2c stacking got SIGKILL'd (exit 137) reproducibly right after fold 0**, three runs
   in a row, on the notebook's memory-constrained container. Root cause: DMatrix/booster host-side
   buffers weren't released between folds. Fixed with explicit `del dtr, dva, m; gc.collect()` at the
   end of each fold iteration in `train_stack.py`.
6. **sentence-transformers 5.x's legacy `CrossEncoder.fit(train_dataloader=...)` API is unreliable**:
   one run silently didn't persist weights to `output_path` at all (empty directory afterward
   despite a clean "done" log line), and a retry with an explicit `model.save()` afterward hung with
   zero progress for 110+ minutes. Root-caused to the deprecated `DataLoader`/`InputExample` path;
   fixed by migrating to `CrossEncoderTrainer` + `datasets.Dataset` + `BinaryCrossEntropyLoss` with
   `dataloader_num_workers=0` (this is what `train_xenc.py`/`train_xenc2.py` use now, and it's been
   validated end-to-end — weights genuinely save, training completes in the expected time).
7. **A SageMaker notebook's local-mode SSH bootstrap container only registers in SSM *after*
   `bootstrap_ssh.py` actually runs** — normally a manual step in the notebook's own Jupyter
   terminal. To avoid ever touching the browser (explicit user requirement this session), this was
   automated via a **notebook lifecycle configuration** (`create-notebook-instance-lifecycle-config`
   with an on-start script), which runs as root on the actual host and can shell out as `ec2-user` to
   pull the bootstrap script from S3 and launch it with `nohup ... &`. Each lifecycle-config change
   requires a full stop → update → start cycle to take effect (~5-8 min each) — this was iterated on
   5+ times this session (see the config name `roshan-v4-ssh-bootstrap` and its revision history in
   AWS if useful) chasing: (a) a disk-full condition on the container's root overlay (97% used, only
   3.3GB free — `docker system prune` found nothing to reclaim, meaning it wasn't stale Docker
   images; root cause never fully identified, though root disk usage came back to a healthier 18GB
   free on a later attempt without clear explanation), (b) `--volume-size-in-gb` on
   `update-notebook-instance` **only resizes `/home/ec2-user/SageMaker`, not the OS root volume**
   where Docker/conda actually live — resizing to 200GB did not fix the root disk issue, (c) a flaky
   conda-env auto-detection (`grep -m1 -i python3` against `conda env list`) that worked once then
   picked a different env on a later restart, missing the `sagemaker_ssh_helper` package that had
   only been pip-installed into the *previous* env — fixed by pinning to one dedicated env name
   (`sshboot`) created deterministically every time.
8. **SSM session loss can affect every managed instance in the account simultaneously**, not just
   one notebook — seen directly this session (`roshan-notebook`'s and `roshan-v4-notebook`'s
   containers both went `ConnectionLost` around the same time and neither recovered on repeated
   retries over ~40+ minutes of checking). General AWS API calls (e.g. `aws s3 ls`) kept working
   throughout, so it's specifically the SSM Session Manager WebSocket layer, not full connectivity
   loss. No fix found within this session; only mitigation is patience + periodic retry, and *not*
   taking destructive action (stop/restart) on the assumption that a running job would be lost for
   nothing if the underlying container is actually fine and just the SSM link is stuck.

## 8. Rules established this session

- Git commits under username **"Roshan"** only — never a Claude/agent git identity. (Co-authorship
  trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` is added per this session's
  attribution convention; adjust if that convention doesn't apply to your session.)
- No code-versioning/archiving step before pushing to SageMaker — edit locally, commit, sync to S3,
  pull onto the notebook, done (§4).
- No code zip needed for submission — only `matching_results.tsv` + `candidate_pairs.tsv`.
- Conserve tool/token usage where possible: batch SSH calls, avoid tight polling loops, prefer longer
  wait intervals when checking on long-running jobs.
- **Never automate anything through the browser for AWS/notebook access** — this was an explicit,
  repeated instruction this session; all notebook bootstrapping had to go through lifecycle configs
  or direct SSH/SSM instead (§7 point 7 is the direct consequence of this constraint).
- Deleting a teammate's AWS resource (this session: `barani-notebook`) requires explicit user
  confirmation first, every time, even if technically permitted by IAM.
- Every new architecture idea is evaluated the same way, no exceptions: **ship only if the paired-
  bootstrap 95% CI lower bound is > 0** versus whichever of {v0base, v3 stack0, ...} is currently the
  best gate-passing model. Nothing is assumed to help until it clears that bar — this discipline is
  why `train_rank` (listwise LTR) and `consensus_prune` (relational post-hoc rule) were both tried,
  found to lose the gate, and correctly *not* shipped, rather than shipped on the strength of their
  reasoning alone.
