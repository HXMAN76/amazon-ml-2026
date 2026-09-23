#!/usr/bin/env bash
# Run from B/C/D (CloudShell or laptop) after 10_hub_bucket.sh. Proves cross-account access works
# and that 00-raw/ is write-protected for non-A accounts.
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh
ACCT=$(me)
key="07-experiments/_ping/$ACCT.txt"
echo "hello from $ACCT $(date -u +%FT%TZ)" > /tmp/ping.txt

aws s3 ls "s3://$HUB_BUCKET/" && echo "PASS list" || echo "FAIL list"
aws s3 cp /tmp/ping.txt "s3://$HUB_BUCKET/$key" --only-show-errors && echo "PASS write $key" || echo "FAIL write"
aws s3 cp "s3://$HUB_BUCKET/$key" - >/dev/null && echo "PASS read" || echo "FAIL read"
if [[ "$ACCT" != "$ACCOUNT_A" ]]; then
  if aws s3 cp /tmp/ping.txt "s3://$HUB_BUCKET/00-raw/_should_fail.txt" --only-show-errors 2>/dev/null; then
    echo "FAIL 00-raw is writable from $ACCT"
  else
    echo "PASS 00-raw write denied"
  fi
fi
