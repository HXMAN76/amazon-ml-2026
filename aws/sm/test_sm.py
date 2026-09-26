"""Tests for the SageMaker job tooling (no AWS: fake clients; entry.py runs locally with test hooks).

Run: <python with pytest> -m pytest -q aws/sm/test_sm.py
"""

import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm  # noqa: E402


def test_job_name_is_valid_and_bounded():
    n = sm.job_name("v5", ["v5_candidates", "v5_pass1"], now=datetime(2026, 9, 26, 1, 2, 3))
    assert n == "ber-v5-v5-candidates-v5-pass1-260926-010203"
    long = sm.job_name("run_with_a_long_name", ["v5_candidates", "v5_pass1", "v5_expand", "v5_pass2", "v5_xenc"])
    assert len(long) <= 63 and all(c.isalnum() or c == "-" for c in long) and not long.startswith("-")


def test_tarball_has_package_and_entry_but_no_caches(tmp_path):
    pkg = tmp_path / "business_entity_resolution"
    (pkg / "src" / "ber" / "__pycache__").mkdir(parents=True)
    (pkg / "src" / "ber" / "x.py").write_text("x = 1\n")
    (pkg / "src" / "ber" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (pkg / "Makefile").write_text("all:\n")
    entry = tmp_path / "entry.py"
    entry.write_text("print('hi')\n")
    names = tarfile.open(fileobj=io.BytesIO(sm.build_tarball(pkg, entry)), mode="r:gz").getnames()
    assert set(names) == {"entry.py", "business_entity_resolution/Makefile", "business_entity_resolution/src/ber/x.py"}


def test_job_request_runs_entry_from_the_code_channel():
    r = sm.job_request("ber-v5-smoke-1", "arn:aws:iam::1:role/r", "bkt", "v5", ["smoke"], "ml.g5.4xlarge", 1.5, git_sha="abc")
    spec = r["AlgorithmSpecification"]
    assert spec["TrainingImage"].endswith("pytorch-training:2.7.1-gpu-py312") and spec["ContainerEntrypoint"] == ["bash", "-c"]
    assert "/opt/ml/input/data/code/source.tar.gz" in spec["ContainerArguments"][0] and "entry.py" in spec["ContainerArguments"][0]
    ch = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in r["InputDataConfig"]}
    assert ch == {"code": "s3://bkt/ber/code/ber-v5-smoke-1/", "dataset": "s3://bkt/ber/dataset/"}
    assert r["Environment"]["BER_WORK_S3"] == "s3://bkt/ber/work/v5/" and r["Environment"]["BER_STAGES"] == "smoke"
    assert r["ResourceConfig"]["InstanceType"] == "ml.g5.4xlarge" and r["StoppingCondition"]["MaxRuntimeInSeconds"] == 5400


class FakeSM:
    def __init__(self, script):
        self.script, self.i = script, 0

    def describe_training_job(self, TrainingJobName):
        d = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return {"TrainingJobStatus": d["status"], "SecondaryStatusTransitions": d["trans"], "ResourceConfig": {"InstanceType": "ml.g5.4xlarge"},
                "BillableTimeInSeconds": 1800, "FailureReason": d.get("fail")}


class FakeLogs:
    """Log lines appear over time; get_log_events pages like CloudWatch (same token back when nothing is new)."""

    def __init__(self, batches):
        self.batches, self.calls = batches, 0

    def describe_log_streams(self, logGroupName, logStreamNamePrefix):
        self.calls += 1
        if self.calls == 1:
            raise type("ResourceNotFoundException", (Exception,), {})("log group not found yet")
        return {"logStreams": [{"logStreamName": logStreamNamePrefix + "algo-1-1"}]}

    def get_log_events(self, logGroupName, logStreamName, startFromHead, nextToken=None):
        avail = sum(self.batches[: self.calls - 1], [])       # everything written up to this poll
        pos = int(nextToken or 0)
        new = avail[pos: pos + 2]                               # small pages force the inner paging loop
        return {"events": [{"message": m} for m in new], "nextForwardToken": str(pos + len(new))}


def test_follow_streams_every_line_once_and_reports_the_end():
    t1 = {"Status": "Starting", "StartTime": 1, "StatusMessage": "Preparing the instances"}
    t2 = {"Status": "Training", "StartTime": 2, "StatusMessage": "Training image download completed"}
    script = [{"status": "InProgress", "trans": [t1]}, {"status": "InProgress", "trans": [t1, t2]},
              {"status": "InProgress", "trans": [t1, t2]}, {"status": "Completed", "trans": [t1, t2]}]
    logs = FakeLogs([["a", "b", "c"], ["d"], ["e", "f", "g"]])
    out = io.StringIO()
    st = sm.follow("ber-x", FakeSM(script), logs, poll=0, out=out)
    lines = out.getvalue().splitlines()
    assert st == "Completed"
    assert [ln for ln in lines if not ln.startswith("[")] == ["a", "b", "c", "d", "e", "f", "g"]   # each log line exactly once
    assert lines[0].startswith("[status] Starting") and sum(ln.startswith("[status]") for ln in lines) == 2
    assert lines[-1].startswith("[done] ber-x: Completed") and "$1.01" in lines[-1]               # 30 min on ml.g5.4xlarge


def test_dry_run_prints_the_request_without_aws(capsys):
    sm.main(["--profile", "none", "submit", "--stages", "smoke", "--run", "t", "--dry-run"])
    out = capsys.readouterr().out
    req = json.loads(out[: out.rindex("}") + 1])
    assert req["Environment"]["BER_STAGES"] == "smoke" and "<account>" in req["RoleArn"] and "code tarball" in out


def _entry(tmp_path, make_cmd, stages):
    env = {**os.environ, "BER_ML_ROOT": str(tmp_path / "ml"), "BER_CODE": str(tmp_path), "BER_MAKE_CMD": make_cmd,
           "BER_SKIP_INSTALL": "1", "BER_STAGES": stages, "BER_WORK_S3": "", "BER_HEARTBEAT": "3600"}
    (tmp_path / "ml").mkdir(exist_ok=True)
    return subprocess.run([sys.executable, str(Path(sm.__file__).parent / "entry.py")], env=env, capture_output=True, text=True)


def test_entry_runs_stages_in_order_and_reports_failures(tmp_path):
    ok = _entry(tmp_path, "echo made", "stage_a stage_b")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "STAGE stage_a: OK" in ok.stdout and "STAGE stage_b: OK" in ok.stdout and "ALL STAGES DONE" in ok.stdout
    bad = _entry(tmp_path, "false", "stage_a stage_b")
    assert bad.returncode == 1 and "STAGE stage_a: FAILED" in bad.stdout and "STAGE stage_b: started" not in bad.stdout
    assert "stage stage_a exited with 1" in (tmp_path / "ml" / "output" / "failure").read_text()


def test_stage_checkpoint_keeps_only_what_the_stage_wrote(tmp_path):
    import entry

    w = tmp_path / "work"
    old = [w / "models" / "v5a" / "xgb.json", w / "features" / "train" / "part_0000.parquet"]
    for p in old:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("old")
        os.utime(p, (1_000, 1_000))                                   # written by an earlier stage
    new = {"models/v5b/xgb.json": "m", "models/v5b/holdout.json": '{"macro_f05": 0.9561}', "models/v5b/p1_rest/part_0000.parquet": "x",
           "output/v5b/matching_results.tsv": "t", "output/v5b/candidate_pairs.tsv": "c", "features/test/part_0001.parquet": "f",
           "measure/holdout_v5b.json": '{"holdout_f05": 0.9561}', "dense_all/model_ft/model.safetensors": "w"}
    for rel, txt in new.items():
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        (w / rel).write_text(txt)
    got = sorted(p.relative_to(w).as_posix() for p in entry.stage_checkpoint_files(w, since=2_000))
    assert got == ["dense_all/model_ft/model.safetensors", "measure/holdout_v5b.json", "models/v5b/holdout.json", "models/v5b/xgb.json",
                   "output/v5b/matching_results.tsv"]                  # no p1_rest, no bulk features / candidates, nothing old
    man = entry.stage_manifest(w, [w / g for g in got], "v5_pass2", 2_000.0, 2_600.0)
    assert man["stage"] == "v5_pass2" and man["minutes"] == 10.0 and len(man["files"]) == 5
    assert man["metrics"]["models/v5b/holdout.json"]["macro_f05"] == 0.9561 and "measure/holdout_v5b.json" in man["metrics"]
    assert sm.headline(man) == "measure/holdout_v5b.json holdout_f05=0.9561; v5b macro_f05=0.9561"   # file-path order


def test_job_request_carries_the_checkpoint_location_and_numbering():
    r = sm.job_request("ber-v5-x-1", "arn:aws:iam::1:role/r", "bkt", "v5", ["v5_pass1"], "ml.g5.4xlarge", 2, ckpt_base=3, owner="barani")
    e = r["Environment"]
    assert e["BER_CKPT_S3"] == "s3://ml-challenge-nooglers/ml-challenge-2026/checkpoints/barani/v5/"   # the team's common bucket
    assert e["BER_CKPT_FALLBACK"] == "s3://bkt/ber/checkpoints/v5/" and e["BER_CKPT_BASE"] == "3" and e["BER_JOB_NAME"] == "ber-v5-x-1"
