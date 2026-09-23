# Paste into the FIRST cell of a Kaggle notebook (Settings: GPU T4 x2 or P100, Internet ON).
# Kaggle Secrets needed (Add-ons > Secrets): GITHUB_TOKEN (fine-grained, read-only on the repo),
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (from aws/15_external_user.sh), AMLC_BUCKET.
# Free quota: ~30 GPU h/week per Kaggle account -> 4 teammates = ~120 GPU h/week.
import os
import subprocess

from kaggle_secrets import UserSecretsClient

s = UserSecretsClient()
for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AMLC_BUCKET"]:
    try:
        os.environ[k] = s.get_secret(k)
    except Exception:
        print("missing secret", k)
os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
os.environ["AMLC_MEMBER"] = "kaggle-" + os.environ.get("KAGGLE_USERNAME", "x")

REPO = "HXMAN76/amazon-ml-2026"
tok = s.get_secret("GITHUB_TOKEN")
subprocess.run(f"rm -rf /kaggle/working/amlc && git clone -q --depth 1 https://{tok}@github.com/{REPO}.git /kaggle/working/amlc",
               shell=True, check=True)
# keep Kaggle's CUDA torch; install our package + HF extras on top
subprocess.run("pip install -q -e '/kaggle/working/amlc[hf]' awscli", shell=True, check=True)
os.chdir("/kaggle/working/amlc")
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)

# Typical worker cell afterwards (each teammate uses a different --shard):
# !aws s3 cp s3://$AMLC_BUCKET/02-processed/test_img.parquet data/
# !python -m amlc.data.downloader --input data/test_img.parquet --url-col image_link --out data/images --max-side 512
# !python -m amlc.features.embed image --input data/test_img.parquet --col image_path \
#      --model google/siglip2-base-patch16-224 --out artifacts/emb/test/siglip2 --shard 0 --num-shards 4
# !aws s3 sync artifacts/ s3://$AMLC_BUCKET/03-features/artifacts/
