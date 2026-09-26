#!/usr/bin/env python3
"""Run BER pipeline stages as SageMaker training jobs on a GPU (no notebook), and watch them live from the terminal.

  python aws/sm/sm.py check                               read-only: identity, bucket, role, dataset access, GPU quota
  python aws/sm/sm.py setup                               one time: bucket, execution role, dataset copy (idempotent)
  python aws/sm/sm.py submit --stages smoke               start a job and stream its log live (Ctrl-C detaches; the job keeps running)
  python aws/sm/sm.py submit --run v5 --stages v5_candidates
  python aws/sm/sm.py logs <job>                          attach to a job's live log from any terminal
  python aws/sm/sm.py status [<job>]                      recent jobs (status, runtime, cost) or one job in detail
  python aws/sm/sm.py stop <job>
  python aws/sm/sm.py pull --run v5 [--model s5]          reports (+ the model's matching_results.tsv) -> output/sm/<run>/
  python aws/sm/sm.py checkpoints --run v5                stage checkpoints of a run (time, files, holdout scores)
  python aws/sm/sm.py queue-setup [--instance ml.g5.4xlarge]   notebook job queue (team workflow): scripts, lifecycle, type
  python aws/sm/sm.py nb start|stop|status                start / stop the notebook barani-v5
  python aws/sm/sm.py publish                             upload code/business_entity_resolution for the queued jobs
  python aws/sm/sm.py enqueue aws/queue/jobs/smoke.sh     queue a job (one at a time on the notebook GPU)
  python aws/sm/sm.py jobs                                runner beacon + pending / live / done jobs
  python aws/sm/sm.py jlog <job>                          follow a job's log until exit=<code>
  python aws/sm/sm.py nbstatus                            notebook barani-v5: state, SSH container, pipeline stages so far
  python aws/sm/sm.py nblog                               follow the notebook pipeline log live (from S3, no SSH needed)
  python aws/sm/sm.py pull --run v5 --checkpoint <id>     download one stage checkpoint

Every job of a run shares WORK on S3 (s3://<bucket>/ber/work/<run>/): a job restores it, runs its stages, saves after every stage.
After every successful stage the files it produced (model weights, configs, holdout reports, measurements, the submission file) are
also frozen as a checkpoint in the team's common bucket, s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/<owner>/<run>/
<NN>-<stage>-<time>/ with a manifest.json; never overwritten (fallback: s3://<own bucket>/ber/checkpoints/<run>/).
Run with the smssh-venv Python (boto3 + botocore[crt] reads `aws login` sessions). Profile: --profile, else AWS_PROFILE, else barani.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "code" / "business_entity_resolution"
ENTRY = Path(__file__).resolve().parent / "entry.py"
REGION = "us-east-1"
IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312"   # AWS PyTorch DLC, Python 3.12, CUDA
ROLE_NAME = "ber-sagemaker-training"
DATA_SRC = "s3://ml-challenge-nooglers/ml-challenge-2026/raw/v1"
COMMON = "s3://ml-challenge-nooglers/ml-challenge-2026"             # the team's common bucket: checkpoints go to checkpoints/<owner>/<run>/
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
TERMINAL = {"Completed", "Failed", "Stopped"}
PRICE_PER_HOUR = {"ml.g5.xlarge": 1.41, "ml.g5.2xlarge": 1.52, "ml.g5.4xlarge": 2.03, "ml.g5.8xlarge": 3.06,  # approximate on-demand
                  "ml.g5.12xlarge": 7.09, "ml.g4dn.xlarge": 0.74, "ml.g4dn.4xlarge": 1.51}                   # training, us-east-1
PKG_SKIP = ["*/__pycache__/*", "*.pyc", "*/.pytest_cache/*", "*/.DS_Store", "*/.venv*"]


# ------------------------------------------------------------------ pure helpers (unit-tested without AWS)
def job_name(run: str, stages: list[str], now: datetime | None = None) -> str:
    """Valid SageMaker job name (<= 63 chars of [a-zA-Z0-9-]): ber-<run>-<stages>-<yymmdd-hhmmss>."""
    ts = (now or datetime.now()).strftime("%y%m%d-%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", "-".join(stages)).strip("-")
    head = re.sub(r"[^a-zA-Z0-9]+", "-", f"ber-{run}").strip("-")
    room = 63 - len(head) - len(ts) - 2
    return f"{head}-{slug[:max(room, 1)].strip('-')}-{ts}"


def build_tarball(pkg: Path = PKG, entry: Path = ENTRY) -> bytes:
    """The code shipped to the job: the package directory (without caches) plus entry.py, as a gzipped tar."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(pkg.rglob("*")):
            rel = f"{pkg.name}/{p.relative_to(pkg).as_posix()}"
            if p.is_file() and not any(fnmatch.fnmatch("/" + rel, pat) for pat in PKG_SKIP):
                tar.add(p, arcname=rel)
        tar.add(entry, arcname="entry.py")
    return buf.getvalue()


def job_request(name: str, role_arn: str, bucket: str, run: str, stages: list[str], instance: str, max_hours: float,
                image: str = IMAGE, git_sha: str = "", heartbeat: int = 120, volume_gb: int = 200, ckpt_base: int = 0,
                owner: str = "barani") -> dict:
    """CreateTrainingJob request: the DLC image runs entry.py from the `code` channel; `dataset` holds train/ test/ utils/."""
    cmd = "set -e; mkdir -p /opt/code; tar xzf /opt/ml/input/data/code/source.tar.gz -C /opt/code; exec python /opt/code/entry.py"
    chan = lambda name_, uri: {"ChannelName": name_, "DataSource": {"S3DataSource": {  # noqa: E731
        "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}, "InputMode": "File"}
    return {
        "TrainingJobName": name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File",
                                   "ContainerEntrypoint": ["bash", "-c"], "ContainerArguments": [cmd]},
        "InputDataConfig": [chan("code", f"s3://{bucket}/ber/code/{name}/"), chan("dataset", f"s3://{bucket}/ber/dataset/")],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/ber/output/"},
        "ResourceConfig": {"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": volume_gb},  # g5 uses its NVMe instead
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_hours * 3600)},
        "Environment": {"BER_STAGES": " ".join(stages), "BER_WORK_S3": f"s3://{bucket}/ber/work/{run}/",
                        "BER_CKPT_S3": ckpt_uri(owner, run), "BER_CKPT_FALLBACK": f"s3://{bucket}/ber/checkpoints/{run}/",
                        "BER_CKPT_BASE": str(ckpt_base),
                        "BER_JOB_NAME": name, "BER_CODE": "/opt/code/business_entity_resolution", "BER_GIT_SHA": git_sha or "nogit",
                        "BER_HEARTBEAT": str(heartbeat), "PYTHONUNBUFFERED": "1"},
        "Tags": [{"Key": "project", "Value": "ber"}, {"Key": "run", "Value": run}],
    }


def ckpt_uri(owner: str, run: str) -> str:
    """Where stage checkpoints go: the common team bucket, one folder per person and run (so runs of teammates never collide)."""
    return f"{COMMON}/checkpoints/{owner}/{run}/"


def cost(instance: str, seconds: float) -> float:
    return PRICE_PER_HOUR.get(instance, 0.0) * seconds / 3600


def follow(job: str, sm, logs, poll: float = 10.0, out=sys.stdout, max_polls: int | None = None) -> str:
    """Stream a job's status transitions and CloudWatch log lines until it ends (each line printed once); returns the status."""
    seen: set = set()
    tokens: dict[str, str | None] = {}
    polls = 0
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        for t in d.get("SecondaryStatusTransitions", []):
            key = (t["Status"], str(t.get("StartTime")))
            if key not in seen:
                seen.add(key)
                print(f"[status] {t['Status']}: {t.get('StatusMessage', '')}", file=out, flush=True)
        try:
            for s in logs.describe_log_streams(logGroupName=LOG_GROUP, logStreamNamePrefix=job + "/").get("logStreams", []):
                tokens.setdefault(s["logStreamName"], None)
        except Exception as e:  # noqa: BLE001 - the log group/stream appears only once the container starts
            if "ResourceNotFound" not in type(e).__name__ + str(e):
                raise
        for stream in list(tokens):
            while True:
                kw = {"logGroupName": LOG_GROUP, "logStreamName": stream, "startFromHead": True}
                if tokens[stream]:
                    kw["nextToken"] = tokens[stream]
                r = logs.get_log_events(**kw)
                for e in r.get("events", []):
                    print(e["message"], file=out, flush=True)
                nxt = r.get("nextForwardToken")
                if not r.get("events") or nxt == tokens[stream]:
                    tokens[stream] = nxt
                    break
                tokens[stream] = nxt
        status = d["TrainingJobStatus"]
        if status in TERMINAL:
            secs = d.get("BillableTimeInSeconds") or d.get("TrainingTimeInSeconds") or 0
            inst = d["ResourceConfig"]["InstanceType"]
            print(f"[done] {job}: {status}" + (f" | {d['FailureReason']}" if d.get("FailureReason") else "") +
                  f" | billable {secs / 60:.1f} min, about ${cost(inst, secs):.2f} on {inst}", file=out, flush=True)
            return status
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return status
        time.sleep(poll)


# ------------------------------------------------------------------ AWS side
def session(profile: str):
    import boto3

    return boto3.Session(profile_name=profile, region_name=REGION)


def _ids(sess) -> tuple[str, str, str]:
    acct = sess.client("sts").get_caller_identity()["Account"]
    return acct, f"sagemaker-{REGION}-{acct}", f"arn:aws:iam::{acct}:role/{ROLE_NAME}"


def _split(uri: str) -> tuple[str, str]:
    b, _, k = uri.removeprefix("s3://").partition("/")
    return b, k


def check(sess, quota: bool) -> None:
    ident = sess.client("sts").get_caller_identity()
    acct, bucket, role_arn = _ids(sess)
    print(f"identity: {ident['Arn']}")
    s3, iam, sm = sess.client("s3"), sess.client("iam"), sess.client("sagemaker")
    probes = {
        "bucket " + bucket: lambda: s3.head_bucket(Bucket=bucket),
        "role " + ROLE_NAME: lambda: iam.get_role(RoleName=ROLE_NAME),
        "sagemaker list-training-jobs": lambda: sm.list_training_jobs(MaxResults=1),
        "dataset source " + DATA_SRC: lambda: s3.list_objects_v2(Bucket=_split(DATA_SRC)[0], Prefix=_split(DATA_SRC)[1] + "/dataset/", MaxKeys=5)["KeyCount"],
        "dataset copy s3://" + bucket + "/ber/dataset/": lambda: s3.list_objects_v2(Bucket=bucket, Prefix="ber/dataset/", MaxKeys=20)["KeyCount"],
    }
    for label, fn in probes.items():
        try:
            r = fn()
            print(f"  ok   {label}" + (f" ({r} objects)" if isinstance(r, int) else ""))
        except Exception as e:  # noqa: BLE001 - report every probe
            print(f"  --   {label}: {type(e).__name__}: {str(e)[:160]}")
    if quota:
        sq = sess.client("service-quotas")
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page["Quotas"]:
                if re.fullmatch(r"ml\.g5\.(x|2x|4x|8x)large for training job usage", q["QuotaName"]):
                    print(f"  quota {q['QuotaName']}: {q['Value']:.0f}")


def setup(sess) -> None:
    """Idempotent: S3 bucket, execution role (SageMaker + this bucket), dataset + official validator copied into the bucket."""
    acct, bucket, role_arn = _ids(sess)
    s3, iam = sess.client("s3"), sess.client("iam")
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket exists: {bucket}")
    except Exception:  # noqa: BLE001 - missing
        s3.create_bucket(Bucket=bucket)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={k: True for k in (
            "BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")})
        print(f"bucket created: {bucket}")
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"role exists: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
                                                          "Action": "sts:AssumeRole"}]}
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="BER pipeline SageMaker training jobs", Tags=[{"Key": "project", "Value": "ber"}])
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
        print(f"role created: {role_arn}")
    cb, cp = _split(COMMON)
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
         "Resource": f"arn:aws:s3:::{bucket}/*"},
        {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{cb}/{cp}/checkpoints/*"},
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{cb}",
         "Condition": {"StringLike": {"s3:prefix": [f"{cp}/checkpoints", f"{cp}/checkpoints/*"]}}}]}
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="ber-work-bucket", PolicyDocument=json.dumps(policy))
    print(f"role policy: own bucket read/write; checkpoints in {COMMON}/checkpoints/")
    sb, sk = _split(DATA_SRC)
    copied = skipped = 0
    for sub in ("dataset/train/", "dataset/test/", "utils/validate_submission.py"):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=sb, Prefix=f"{sk}/{sub}"):
            for o in page.get("Contents", []):
                rel = o["Key"][len(sk) + 1:].removeprefix("dataset/")
                dst = f"ber/dataset/{rel}"
                try:
                    if s3.head_object(Bucket=bucket, Key=dst)["ContentLength"] == o["Size"]:
                        skipped += 1
                        continue
                except Exception:  # noqa: BLE001 - not copied yet
                    pass
                print(f"copying s3://{sb}/{o['Key']} -> s3://{bucket}/{dst} ({o['Size'] / 1e6:.0f} MB)", flush=True)
                s3.copy({"Bucket": sb, "Key": o["Key"]}, bucket, dst)  # server-side (multipart when large); no laptop transfer
                copied += 1
    print(f"dataset: {copied} copied, {skipped} already present in s3://{bucket}/ber/dataset/")


def submit(sess, a) -> None:
    import subprocess

    if a.dry_run:  # no AWS call at all: placeholders for the account-specific names
        bucket, role_arn = f"sagemaker-{REGION}-<account>", f"arn:aws:iam::<account>:role/{ROLE_NAME}"
        base = 0
    else:
        _, bucket, role_arn = _ids(sess)
        base = len(checkpoint_ids(sess, a.owner, a.run))  # checkpoint numbers continue across the jobs of a run
    stages = a.stages.split()
    name = job_name(a.run, stages)
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
        sha += "-dirty" if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet"]).returncode else ""
    except Exception:  # noqa: BLE001
        sha = "nogit"
    req = job_request(name, role_arn, bucket, a.run, stages, a.instance, a.max_hours, image=a.image, git_sha=sha, heartbeat=a.heartbeat,
                      ckpt_base=base, owner=a.owner)
    tar = build_tarball()
    if a.dry_run:
        print(json.dumps(req, indent=2))
        print(f"code tarball: {len(tar) / 1e6:.2f} MB (not uploaded: dry run)")
        return
    sess.client("s3").put_object(Bucket=bucket, Key=f"ber/code/{name}/source.tar.gz", Body=tar)
    sess.client("sagemaker").create_training_job(**req)
    print(f"submitted {name} on {a.instance} (run {a.run}, stages {stages}, max {a.max_hours} h)")
    print(f"console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{name}")
    if not a.detach:
        watch(sess, name)


def watch(sess, job: str) -> None:
    print(f"following {job} (Ctrl-C detaches; the job keeps running; reattach: python aws/sm/sm.py logs {job})", flush=True)
    try:
        follow(job, sess.client("sagemaker"), sess.client("logs"))
    except KeyboardInterrupt:
        print(f"\ndetached. reattach: python aws/sm/sm.py logs {job}   stop: python aws/sm/sm.py stop {job}")


def status(sess, job: str | None) -> None:
    sm = sess.client("sagemaker")
    if job:
        d = sm.describe_training_job(TrainingJobName=job)
        secs = d.get("BillableTimeInSeconds") or 0
        print(json.dumps({"job": job, "status": d["TrainingJobStatus"], "secondary": d.get("SecondaryStatus"),
                          "failure": d.get("FailureReason"), "instance": d["ResourceConfig"]["InstanceType"],
                          "stages": d.get("Environment", {}).get("BER_STAGES"), "work": d.get("Environment", {}).get("BER_WORK_S3"),
                          "billable_min": round(secs / 60, 1), "cost_usd": round(cost(d["ResourceConfig"]["InstanceType"], secs), 2)}, indent=2))
        return
    now = datetime.now(timezone.utc)
    for j in sm.list_training_jobs(NameContains="ber-", SortBy="CreationTime", SortOrder="Descending", MaxResults=15)["TrainingJobSummaries"]:
        end = j.get("TrainingEndTime") or now
        mins = (end - j["CreationTime"]).total_seconds() / 60
        print(f"{j['TrainingJobName']:<63} {j['TrainingJobStatus']:<11} {mins:6.1f} min")


def _ckpt_locations(sess, owner: str, run: str) -> list[tuple[str, str]]:
    """(bucket, prefix) of the checkpoint folders of a run: the common bucket first, then fallback copies in the own bucket."""
    _, own, _ = _ids(sess)
    cb, cp = _split(ckpt_uri(owner, run))
    return [(cb, cp), (own, f"ber/checkpoints/{run}/")]


def checkpoint_ids(sess, owner: str, run: str) -> list[str]:
    """Checkpoint folders of a run (NN-stage-time), oldest first, from both locations."""
    out = set()
    for b, prefix in _ckpt_locations(sess, owner, run):
        try:
            r = sess.client("s3").list_objects_v2(Bucket=b, Prefix=prefix, Delimiter="/")
        except Exception:  # noqa: BLE001 - own bucket or common prefix not there yet
            continue
        out |= {p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in r.get("CommonPrefixes", [])}
    return sorted(out)


def _find_ckpt(sess, owner: str, run: str, cid: str) -> tuple[str, str] | None:
    for b, prefix in _ckpt_locations(sess, owner, run):
        try:
            if sess.client("s3").list_objects_v2(Bucket=b, Prefix=f"{prefix}{cid}/", MaxKeys=1).get("KeyCount"):
                return b, f"{prefix}{cid}/"
        except Exception:  # noqa: BLE001
            continue
    return None


def headline(manifest: dict) -> str:
    """One line per checkpoint: the scores its reports carry (final / stacked / first-stage holdout, test-mix estimate)."""
    bits = []
    for path, m in manifest.get("metrics", {}).items():
        if not isinstance(m, dict):
            continue
        name = path.split("/")[1] if path.startswith("models/") else path
        for key in ("final_holdout_f05", "stack_holdout_f05", "macro_f05", "holdout_f05", "recall"):
            if isinstance(m.get(key), (int, float)):
                bits.append(f"{name} {key}={m[key]:.4f}")
                break
        est = (m.get("test_mix_estimate") or m.get("test_mix") or {}).get("estimate_pooled_proxy")
        if isinstance(est, (int, float)):
            bits.append(f"{name} test-mix={est:.4f}")
    return "; ".join(bits) or "-"


def checkpoints(sess, owner: str, run: str) -> None:
    s3 = sess.client("s3")
    ids = checkpoint_ids(sess, owner, run)
    if not ids:
        print(f"no checkpoints yet under {ckpt_uri(owner, run)}")
    for cid in ids:
        loc = _find_ckpt(sess, owner, run, cid)
        try:
            m = json.loads(s3.get_object(Bucket=loc[0], Key=f"{loc[1]}manifest.json")["Body"].read())
        except Exception:  # noqa: BLE001 - an upload that did not finish
            print(f"{cid:<40} (no manifest)")
            continue
        size = sum(f["bytes"] for f in m.get("files", [])) / 1e6
        print(f"{cid:<40} {m.get('minutes', 0):6.1f} min {len(m.get('files', [])):4d} files {size:8.0f} MB  git {m.get('git', '')}  {headline(m)}")
    print(f"location: {ckpt_uri(owner, run)}   download one: sm.py pull --run {run} --checkpoint <id>")


def pull_checkpoint(sess, owner: str, run: str, cid: str) -> None:
    s3 = sess.client("s3")
    loc = _find_ckpt(sess, owner, run, cid)
    if not loc:
        raise SystemExit(f"checkpoint {cid} not found for run {run}")
    bucket, prefix = loc
    dest = REPO / "output" / "sm" / run / "checkpoints" / cid
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, o["Key"], str(dest / rel))
            n += 1
    print(f"{n} files -> {dest}")


def nblog(sess, run: str, follow: bool, poll: float = 15.0, out=sys.stdout) -> None:
    """Stream the notebook pipeline log (copied to S3 every 30 s by run_v5.sh): prints only new bytes, like tail -f."""
    _, bucket, _ = _ids(sess)
    s3 = sess.client("s3")
    key = f"ber/notebook/run/{run}/pipeline.log"
    pos = 0
    while True:
        try:
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size > pos:
                body = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={pos}-")["Body"].read().decode("utf-8", "replace")
                print(body, end="", file=out, flush=True)
                pos = size
                if "RUNNER EXIT" in body:
                    return
        except Exception as e:  # noqa: BLE001 - log not written yet
            if pos == 0 and "404" not in str(e) and "Not Found" not in str(e):
                raise
        if not follow:
            return
        time.sleep(poll)


def nbstatus(sess, run: str) -> None:
    """Notebook state, SSH container registration, and the stage lines of the pipeline log so far."""
    _, bucket, _ = _ids(sess)
    d = sess.client("sagemaker").describe_notebook_instance(NotebookInstanceName="barani-v5")
    print(f"notebook barani-v5: {d['NotebookInstanceStatus']} ({d['InstanceType']})")
    inst = sess.client("ssm").describe_instance_information().get("InstanceInformationList", [])
    print("ssh container: " + (", ".join(f"{i['InstanceId']} {i['PingStatus']}" for i in inst) or "not registered"))
    try:
        log = sess.client("s3").get_object(Bucket=bucket, Key=f"ber/notebook/run/{run}/pipeline.log")["Body"].read().decode("utf-8", "replace")
        lines = [ln for ln in log.splitlines() if any(k in ln for k in ("STAGE", "CHECKPOINT", "HEARTBEAT", "FAILED", "RUNNER EXIT", "ALL STAGES"))]
        print("\n".join(lines[-15:]) or "pipeline log has no stage lines yet")
    except Exception:  # noqa: BLE001
        print("pipeline not started yet")


# ------------------------------------------------------------------ notebook + S3 job queue (the team's workflow, TEAM_GUIDE §3)
NOTEBOOK = "barani-v5"
LIFECYCLE = "barani-queue"
QUEUE_DIR = REPO / "aws" / "queue"


def _aws(profile: str, *args: str) -> int:
    import subprocess

    return subprocess.run(["aws", *args, "--profile", profile, "--region", REGION]).returncode


def queue_setup(sess, profile: str, instance: str) -> None:
    """Upload the notebook scripts, create/refresh the lifecycle config, and point the (stopped) notebook at it."""
    import base64

    _, bucket, _ = _ids(sess)
    s3, sm = sess.client("s3"), sess.client("sagemaker")
    for f in [*QUEUE_DIR.glob("*.sh"), *(QUEUE_DIR / "jobs").glob("*.sh")]:
        key = f"ber/queue/{f.relative_to(QUEUE_DIR).as_posix()}"
        s3.upload_file(str(f), bucket, key)
        print(f"uploaded s3://{bucket}/{key}")
    s3.upload_file(str(ENTRY), bucket, "ber/tools/entry.py")
    content = [{"Content": base64.b64encode((QUEUE_DIR / "onstart.sh").read_bytes()).decode()}]
    try:
        sm.create_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=LIFECYCLE, OnStart=content)
        print(f"lifecycle config {LIFECYCLE} created")
    except sm.exceptions.ClientError as e:
        if "already exists" not in str(e):
            raise
        sm.update_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=LIFECYCLE, OnStart=content)
        print(f"lifecycle config {LIFECYCLE} updated")
    try:
        st = sm.describe_notebook_instance(NotebookInstanceName=NOTEBOOK)["NotebookInstanceStatus"]
    except sm.exceptions.ClientError as e:
        if "RecordNotFound" not in str(e):
            raise
        _, _, role_arn = _ids(sess)
        sm.create_notebook_instance(NotebookInstanceName=NOTEBOOK, InstanceType=instance, RoleArn=role_arn, VolumeSizeInGB=200,
                                    PlatformIdentifier="notebook-al2023-v1", LifecycleConfigName=LIFECYCLE,
                                    Tags=[{"Key": "project", "Value": "ber"}])
        print(f"notebook {NOTEBOOK} created: {instance}, 200 GB, lifecycle {LIFECYCLE} (it starts now; bootstrap log: sm.py jlog _bootstrap)")
        publish(sess, profile)
        return
    if st != "Stopped":
        print(f"notebook {NOTEBOOK} is {st}: stop it (sm.py nb stop) and rerun queue-setup to change its type / lifecycle")
        return
    sm.update_notebook_instance(NotebookInstanceName=NOTEBOOK, InstanceType=instance, LifecycleConfigName=LIFECYCLE)
    print(f"notebook {NOTEBOOK}: type {instance}, lifecycle {LIFECYCLE} (start it with: sm.py nb start)")
    publish(sess, profile)


def publish(sess, profile: str) -> None:
    """Code the jobs run: the package -> ber/code (mirror), entry.py -> ber/tools."""
    _, bucket, _ = _ids(sess)
    rc = _aws(profile, "s3", "sync", str(PKG), f"s3://{bucket}/ber/code", "--delete", "--exclude", "*__pycache__*",
              "--exclude", ".pytest_cache/*", "--exclude", "models/*", "--exclude", "work/*", "--exclude", ".DS_Store", "--only-show-errors")
    sess.client("s3").upload_file(str(ENTRY), bucket, "ber/tools/entry.py")
    print(f"published {PKG.name} -> s3://{bucket}/ber/code" + ("" if rc == 0 else f" (sync exit {rc})"))


def enqueue(sess, path: str, name: str | None, queue: str = "jobs") -> None:
    _, bucket, _ = _ids(sess)
    s3 = sess.client("s3")
    n = name or Path(path).stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", n):
        raise SystemExit(f"bad job name {n!r}")
    for sub in ("done", "live"):
        try:
            s3.head_object(Bucket=bucket, Key=f"{queue}/{sub}/{n}.log")
            print(f"note: {queue}/{sub}/{n}.log exists; use a fresh --name to keep logs apart")
        except Exception:  # noqa: BLE001
            pass
    s3.upload_file(path, bucket, f"{queue}/pending/{n}.sh")
    print(f"queued {n} on {queue}  (watch: sm.py jlog {n} --queue {queue})")


def jobs(sess, queues: tuple[str, ...] = ("jobs", "jobs2")) -> None:
    _, bucket, _ = _ids(sess)
    s3 = sess.client("s3")
    d = sess.client("sagemaker").describe_notebook_instance(NotebookInstanceName=NOTEBOOK)
    print(f"notebook {NOTEBOOK}: {d['NotebookInstanceStatus']} ({d['InstanceType']})")
    try:
        print("runner: " + s3.get_object(Bucket=bucket, Key="jobs/runner.txt")["Body"].read().decode().strip())
    except Exception:  # noqa: BLE001
        print("runner: no beacon yet")
    for q in queues:
        for sub in ("pending", "live", "done"):
            objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"{q}/{sub}/").get("Contents", [])
            objs = sorted(objs, key=lambda o: o["LastModified"])[-12:]
            print(f"{q}/{sub}: " + (", ".join(f"{Path(o['Key']).stem} ({o['LastModified']:%m-%d %H:%M} UTC)" for o in objs) or "-"))


def jlog(sess, name: str, follow: bool, poll: float = 15.0, out=sys.stdout, queue: str = "jobs") -> None:
    """Follow a queued job: <queue>/live/<name>.log while it runs (new bytes only), then the rest of <queue>/done/<name>.log."""
    _, bucket, _ = _ids(sess)
    s3 = sess.client("s3")
    pos = 0

    def read(key: str) -> str | None:
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None

    while True:
        done = read(f"{queue}/done/{name}.log")
        text = done if done is not None else (read(f"{queue}/live/{name}.log") or "")
        if len(text) > pos:
            print(text[pos:], end="", file=out, flush=True)
            pos = len(text)
        if done is not None or not follow:
            return
        time.sleep(poll)


def notebook(sess, action: str) -> None:
    sm = sess.client("sagemaker")
    if action == "start":
        sm.start_notebook_instance(NotebookInstanceName=NOTEBOOK)
        print(f"starting {NOTEBOOK}; bootstrap log: sm.py jlog _bootstrap")
    elif action == "stop":
        sm.stop_notebook_instance(NotebookInstanceName=NOTEBOOK)
        print(f"stopping {NOTEBOOK}")
    else:
        d = sm.describe_notebook_instance(NotebookInstanceName=NOTEBOOK)
        print(f"{NOTEBOOK}: {d['NotebookInstanceStatus']} | {d['InstanceType']} | lifecycle {d.get('NotebookInstanceLifecycleConfigName')}"
              + (f" | {d['FailureReason']}" if d.get("FailureReason") else ""))


def pull(sess, run: str, model: str | None, candidates: bool) -> None:
    """Small results of a run (reports, configs, measure, runs.jsonl; the chosen model's TSVs) into output/sm/<run>/."""
    _, bucket, _ = _ids(sess)
    s3 = sess.client("s3")
    prefix = f"ber/work/{run}/"
    want = ["models/*/holdout.json", "models/*/config.json", "models/*/report.json", "measure/*.json", "runs/runs.jsonl",
            "models/prune/report.json"]
    if model:
        want += [f"output/{model}/matching_results.tsv"] + ([f"output/{model}/candidate_pairs.tsv"] if candidates else [])
    dest = REPO / "output" / "sm" / run
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            if any(fnmatch.fnmatch(rel, w) for w in want):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, o["Key"], str(dest / rel))
                n += 1
                print(f"  {rel} ({o['Size'] / 1e6:.1f} MB)")
    print(f"{n} files -> {dest}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "barani"))
    ap.add_argument("--owner", default=None, help="folder of your checkpoints in the common bucket (default: the profile name)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--quota", action="store_true", help="also list the g5 training-job quotas (slow)")
    sub.add_parser("setup")
    s = sub.add_parser("submit")
    s.add_argument("--stages", required=True, help='space-separated make targets, e.g. "v5_candidates", or "smoke"')
    s.add_argument("--run", default="v5", help="run name: jobs of one run share WORK on S3")
    s.add_argument("--instance", default="ml.g5.4xlarge")
    s.add_argument("--max-hours", type=float, default=8.0)
    s.add_argument("--image", default=IMAGE)
    s.add_argument("--heartbeat", type=int, default=120, help="seconds between GPU/RAM/disk lines in the log")
    s.add_argument("--detach", action="store_true", help="submit and return without following the log")
    s.add_argument("--dry-run", action="store_true", help="print the request and the tarball size; call nothing")
    lg = sub.add_parser("logs")
    lg.add_argument("job")
    st = sub.add_parser("status")
    st.add_argument("job", nargs="?")
    sp = sub.add_parser("stop")
    sp.add_argument("job")
    pu = sub.add_parser("pull")
    pu.add_argument("--run", default="v5")
    pu.add_argument("--model", default=None, help="also download output/<model>/matching_results.tsv")
    pu.add_argument("--candidates", action="store_true", help="with --model: also candidate_pairs.tsv (large)")
    pu.add_argument("--checkpoint", default=None, help="download this stage checkpoint (an id from `checkpoints`) instead")
    ck = sub.add_parser("checkpoints")
    ck.add_argument("--run", default="v5")
    nl = sub.add_parser("nblog", help="follow the notebook pipeline log from S3")
    nl.add_argument("--run", default="v5")
    nl.add_argument("--no-follow", action="store_true")
    ns = sub.add_parser("nbstatus", help="notebook, SSH container and pipeline stage summary")
    ns.add_argument("--run", default="v5")
    qs = sub.add_parser("queue-setup", help="notebook job queue: upload scripts, lifecycle, instance type (notebook must be stopped)")
    qs.add_argument("--instance", default="ml.g5.4xlarge")
    nbp = sub.add_parser("nb", help="notebook start | stop | status")
    nbp.add_argument("action", choices=["start", "stop", "status"])
    sub.add_parser("publish", help="upload the code the queued jobs run")
    eq = sub.add_parser("enqueue", help="queue a job script")
    eq.add_argument("path")
    eq.add_argument("--name", default=None)
    eq.add_argument("--queue", default="jobs", help="lane: jobs (GPU) or jobs2 (CPU, runs in parallel)")
    sub.add_parser("jobs", help="notebook, runner beacon, pending / live / done jobs")
    jl = sub.add_parser("jlog", help="follow a queued job's log")
    jl.add_argument("name")
    jl.add_argument("--no-follow", action="store_true")
    jl.add_argument("--queue", default="jobs")
    a = ap.parse_args(argv)
    a.owner = a.owner or a.profile
    sess = None if (a.cmd == "submit" and a.dry_run) else session(a.profile)
    if a.cmd == "check":
        check(sess, a.quota)
    elif a.cmd == "setup":
        setup(sess)
    elif a.cmd == "submit":
        submit(sess, a)
    elif a.cmd == "logs":
        watch(sess, a.job)
    elif a.cmd == "status":
        status(sess, a.job)
    elif a.cmd == "stop":
        sess.client("sagemaker").stop_training_job(TrainingJobName=a.job)
        print(f"stop requested: {a.job}")
    elif a.cmd == "queue-setup":
        queue_setup(sess, a.profile, a.instance)
    elif a.cmd == "nb":
        notebook(sess, a.action)
    elif a.cmd == "publish":
        publish(sess, a.profile)
    elif a.cmd == "enqueue":
        enqueue(sess, a.path, a.name, a.queue)
    elif a.cmd == "jobs":
        jobs(sess)
    elif a.cmd == "jlog":
        jlog(sess, a.name, not a.no_follow, queue=a.queue)
    elif a.cmd == "nblog":
        nblog(sess, a.run, not a.no_follow)
    elif a.cmd == "nbstatus":
        nbstatus(sess, a.run)
    elif a.cmd == "checkpoints":
        checkpoints(sess, a.owner, a.run)
    elif a.checkpoint:
        pull_checkpoint(sess, a.owner, a.run, a.checkpoint)
    else:
        pull(sess, a.run, a.model, a.candidates)


if __name__ == "__main__":
    main()
