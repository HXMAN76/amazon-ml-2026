# v8 (CPU lane jobs2): the one-word swaps France still predicts after the portal-best recipe (v8u_s27_AR), by swapped-in word; then the recipe plus
# dropping every non-noise one-word swap (swapn) below 0.9999 and at any p, with slot-fit decoy shares.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
python src/scripts/swap_left.py v8u_s27_AR s27
R=restore:noise_swap+noise_extra+initials+spelled_legal+glued
python src/scripts/france_variants.py s27 v8w_s27_AR_sw9999 --rules typeswap:1.01,swapn:0.9999,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
python src/scripts/france_variants.py s27 v8w_s27_AR_swall --rules typeswap:1.01,swapn:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore adds|with matches|PASS|FAIL|^Traceback"
