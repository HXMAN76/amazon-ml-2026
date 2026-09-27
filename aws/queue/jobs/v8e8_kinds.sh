# v8 (CPU lane jobs2): slot-fit decoy share of s28's thrpn-spared pairs above 0.995 by name relation, source and band (after typeswap), France
# against the US and India (reference): which kinds are decoy-rich only in France?
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
for c in france us india; do
  echo "== $c"
  python src/scripts/france_variants.py s28 v8_kinds --dry --country $c --rules typeswap:1.01,kind:one_swap:2:0.9999:0.995,kind:one_typo:2:0.9999:0.995,kind:no_common_word:2:0.9999:0.995,kind:words_dropped:2:0.9999:0.995,kind:words_added:2:0.9999:0.995,kind:glued_fuzzy:2:0.9999:0.995,kind:reordered:2:0.9999:0.995,kind:other:2:0.9999:0.995,kind:one_swap:3:0.9999:0.995,kind:one_typo:3:0.9999:0.995,kind:no_common_word:3:0.9999:0.995,kind:words_dropped:3:0.9999:0.995,kind:words_added:3:0.9999:0.995,kind:glued_fuzzy:3:0.9999:0.995,kind:reordered:3:0.9999:0.995,kind:other:3:0.9999:0.995,kind:one_swap:2:1.01:0.9999,kind:one_typo:2:1.01:0.9999,kind:no_common_word:2:1.01:0.9999,kind:words_dropped:2:1.01:0.9999,kind:words_added:2:1.01:0.9999,kind:glued_fuzzy:2:1.01:0.9999,kind:reordered:2:1.01:0.9999,kind:other:2:1.01:0.9999,kind:one_swap:3:1.01:0.9999,kind:one_typo:3:1.01:0.9999,kind:no_common_word:3:1.01:0.9999,kind:words_dropped:3:1.01:0.9999,kind:words_added:3:1.01:0.9999,kind:glued_fuzzy:3:1.01:0.9999,kind:reordered:3:1.01:0.9999,kind:other:3:1.01:0.9999 2>&1 | grep -E "^rule|Traceback|Error"
done
