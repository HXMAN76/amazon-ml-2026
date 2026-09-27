> **ARCHIVED (26 Sep 2026): SSH and local-container setup, not used any more; we run jobs through the S3 queue (handoff.md section 4). Kept for history only; do not follow its instructions.**

# SageMaker local-container remote development setup

This runbook reproduces the SSH-based development environment used for the Amazon ML Challenge. It also describes how another developer or coding agent can inspect notebooks, modify code, execute experiments, and interpret results inside the SageMaker local-mode training container.


## Note

you need to run remote-ssh.ipynb as a seperate notebook inside your notebook instance for this to work

<br>

## Current environment

| Component | Value |
|---|---|
| AWS profile | `sai-nivedh-26` |
| AWS Region | `us-east-1` |
| SageMaker notebook instance | `remote-tester-sai` |
| Connect target | `remote-tester-sai.notebook.sagemaker` |
| Notebook execution/SSM activation role | `sagemaker-competition-notebook-execution-role` |
| Local Python environment | `smssh-venv` (Python 3.12) |
| SageMaker Python SDK | `2.257.6` |
| SageMaker SSH Helper | `2.3.0` |
| AWS CLI | `2.35.21` |
| Session Manager plugin | `1.2.835.0` |
| Container Python | `/opt/conda/bin/python` (Python 3.8.10) |
| Notebook kernel | `conda_pytorch` |
| Mounted notebook directory | `/opt/ml/input/data/notebook` |

The local-mode training container sees the notebook instance metadata mounted under `/opt/ml/metadata`. SageMaker SSH Helper therefore registers this container under the notebook FQDN, even though the SSH endpoint is the local training container rather than the notebook host itself.

## 1. Install the compatible Python stack

From the project directory:

```bash
python3.12 -m venv smssh-venv
source smssh-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Verify the critical legacy API:

```bash
python - <<'PY'
from importlib.metadata import version
import sagemaker

print("sagemaker:", version("sagemaker"))
print("sagemaker-ssh-helper:", version("sagemaker-ssh-helper"))
assert hasattr(sagemaker, "Session")
print("sagemaker.Session: OK")
PY
```

Do not upgrade this environment to SageMaker SDK v3. SSH Helper 2.3.0 imports `sagemaker.Session`, which v3 no longer exports.

## 2. Configure AWS locally

```bash
export AWS_PROFILE=sai-nivedh-26
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_SUPPRESS_V2_WARNING=1
```

Confirm the active identity and Region:

```bash
aws sts get-caller-identity
aws configure list
```

## 3. Install the Session Manager plugin

The AWS CLI requires the separate Session Manager plugin for `aws ssm start-session`:

```bash
curl -fL \
  https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb \
  -o /tmp/session-manager-plugin.deb
sudo dpkg -i /tmp/session-manager-plugin.deb
session-manager-plugin --version
```

The verified version in this environment is `1.2.835.0`.

## 4. Required IAM configuration

The SSM hybrid activation uses `sagemaker-competition-notebook-execution-role`. Registration can succeed while the managed instance remains `ConnectionLost` if this role lacks the SSM agent channel permissions.

The role must:

1. Trust both `sagemaker.amazonaws.com` and `ssm.amazonaws.com`.
2. Allow the notebook workload to create and inspect SSM activations.
3. Allow `iam:PassRole` for this role, constrained to `ssm.amazonaws.com`.
4. Have AWS managed policy `AmazonSSMManagedInstanceCore` attached.

The repository contains the tested role documents:

```text
iam/sagemaker-ssh-helper-trust-policy.json
iam/sagemaker-ssh-helper-control-policy.json
```

Apply them to the existing execution role:

```bash
aws iam update-assume-role-policy \
  --role-name sagemaker-competition-notebook-execution-role \
  --policy-document file://iam/sagemaker-ssh-helper-trust-policy.json

aws iam put-role-policy \
  --role-name sagemaker-competition-notebook-execution-role \
  --policy-name SageMakerSSHHelperControl \
  --policy-document file://iam/sagemaker-ssh-helper-control-policy.json
```

The control policy scopes `iam:PassRole` to the exact execution role and to
`ssm.amazonaws.com`. Its S3 permissions are limited to the SSH public-key
prefix. If the account, Region, role name, bucket, or prefix changes, update
the ARNs before applying it.

Attach the agent policy if it is absent:

```bash
aws iam attach-role-policy \
  --role-name sagemaker-competition-notebook-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```

The policy supplies `ssm:UpdateInstanceInformation`, the four `ssmmessages` channel actions, and the legacy `ec2messages` actions used by the SSM agent.

The local developer identity also needs permission to discover the managed
instance, upload the public key, invoke the SSM command, and start a session.
At minimum, that means the applicable `ssm:DescribeInstanceInformation`,
`ssm:ListTagsForResource`, `ssm:SendCommand`, `ssm:GetCommandInvocation`,
`ssm:StartSession`, and S3 key-prefix permissions. Keep those caller
permissions separate from the notebook execution-role policy.

## 5. AWS CLI 2.35 compatibility patch

SSH Helper 2.3.0 parses this command using `awk '{print $2}'`:

```bash
aws configure list
```

AWS CLI 2.35 displays colon-separated columns, so the helper incorrectly reads `:` as the Region. In both `sm-connect-ssh-proxy` and `sm-local-start-ssh`, replace the old `aws configure list | grep region ...` assignment with:

```bash
CURRENT_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "${CURRENT_REGION}" ]]; then
  CURRENT_REGION=$(aws configure get region)
fi
if [[ -z "${CURRENT_REGION}" ]]; then
  echo "ERROR: AWS Region is not configured. Set AWS_REGION or AWS_DEFAULT_REGION."
  exit 1
fi
```

Patch these runtime files after installing or upgrading the helper:

```text
smssh-venv/bin/sm-connect-ssh-proxy
smssh-venv/bin/sm-local-start-ssh
```

The installed package also contains source copies under:

```text
smssh-venv/lib/python3.12/site-packages/sagemaker_ssh_helper/
```

Reinstalling `sagemaker-ssh-helper` can overwrite this patch.

## 6. Start the SSH-enabled local container from Jupyter

Make `requirements.txt` available in the notebook directory, install the
pinned requirements in the notebook kernel, then restart that kernel if
package versions changed:

```python
%pip install -r requirements.txt
```

The training entry point must start SSH and remain alive:

```python
import os
import time
import sagemaker_ssh_helper

sagemaker_ssh_helper.setup_and_start_ssh()

while os.environ.get("START_SSH", "false") == "true":
    time.sleep(10)
```

Create `SSHEstimatorWrapper` before calling `estimator.fit(...)`. For the current local-mode workflow, the input channel mounts the notebook directory at `/opt/ml/input/data/notebook` inside the container.

Keep the `estimator.fit(...)` cell running. Interrupting it stops the container and the SSH endpoint.

## 7. Connect from the developer machine

Activate the client environment and export AWS configuration:

```bash
source smssh-venv/bin/activate
export AWS_PROFILE=sai-nivedh-26
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_SUPPRESS_V2_WARNING=1

sm-ssh connect remote-tester-sai.notebook.sagemaker
```

The helper selects the newest matching SSM managed-instance registration. A successful shell looks similar to:

```text
root@algo-1-...:~#
```

The helper also forwards local port `17022` to container SSH port 22. While the main SSH session remains open, a second local process can run commands through the tunnel:

```bash
ssh -i ~/.ssh/sagemaker-ssh-gw \
  -p 17022 root@localhost \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null
```

## 8. Notebook and iterative-development workflow

The notebook directory is mounted read/write:

```text
/opt/ml/input/data/notebook
```

Currently available notebooks include:

```text
remote-ssh.ipynb
sai.ipynb
```

The training container does not host the live Jupyter kernel. It can read and modify notebook files, but it cannot press Jupyter's **Run Cell** button through SSH alone.

Recommended loop:

1. Keep the SSH bootstrap/training cell running.
2. Put reusable code in `/opt/ml/input/data/notebook/src/` rather than embedding all logic in notebook cells.
3. Modify modules or notebook JSON through SSH.
4. Execute experiments with `/opt/conda/bin/python` inside the container.
5. Capture stdout, stderr, metrics, tables, plots, and generated files.
6. Interpret results and iterate on the implementation.
7. Copy stable code and conclusions into notebook cells.
8. Use Jupyter for interactive display or execution when desired, then save the notebook so external tools can read its stored outputs.

Avoid editing the same `.ipynb` simultaneously from Jupyter and over SSH. Jupyter may overwrite external changes when it saves. Prefer Python modules as the shared source of truth.

### Coding-agent capabilities in this setup

With the tunnel open, a coding agent can:

- enumerate and read notebook cells and saved outputs;
- create or modify `.ipynb`, `.py`, configuration, and documentation files;
- inspect mounted datasets without copying them locally;
- execute Python scripts in the actual training container;
- capture and diagnose tracebacks, dependency failures, and resource issues;
- evaluate metrics and compare experiments;
- generate plots or result artifacts and place them in the notebook directory;
- update notebook cells with finalized code and explanations.

The agent cannot directly control the separate live Jupyter kernel through this container-only SSH tunnel. For a live cell, either execute equivalent code in the container or have a user run the cell in Jupyter and save the notebook before the agent reads its output.

## 9. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `module 'sagemaker' has no attribute 'Session'` | SageMaker SDK v3 installed | Reinstall the versions in `requirements.txt`; keep `sagemaker<3` |
| Managed instance is `ConnectionLost` | Activation role lacks agent permissions, or old container agent exited | Attach `AmazonSSMManagedInstanceCore`, then restart the local training cell/container |
| `Provided region_name ':'` | SSH Helper 2.3.0 misparses AWS CLI 2.35 output | Apply the Region patch in section 5 |
| `SessionManagerPlugin is not found` | System plugin absent | Install the `.deb` in section 3 |
| Both old and new `mi-...` IDs appear | Previous local containers left offline registrations | The helper sorts registrations by `SSHTimestamp` and selects the newest first |
| SageMaker v2 deprecation warning | Expected with the compatible SDK | Set `SAGEMAKER_SUPPRESS_V2_WARNING=1` |
| Notebook changes disappear | Jupyter and SSH edited the same notebook concurrently | Keep code in modules or close/save/refresh the notebook before external edits |

## 10. Security and lifecycle notes

- The SSH public key is stored under `s3://sagemaker-us-east-1-767397931665/ssh-authorized-keys/`.
- Never commit private SSH keys, AWS credentials, notebook tokens, or activation codes.
- Stop the local training cell when remote access is no longer needed.
- Old offline SSM managed-instance registrations may be deregistered later after confirming they are no longer used.
- Keep `iam:PassRole` scoped to the specific activation role and `ssm.amazonaws.com`.
