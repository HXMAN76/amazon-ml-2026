# v8 (GPU lane jobs, right after the Qwen scoring): stack s24 = s22's recipe + the Qwen3-0.6B score as xs3, trained on the same 1.5M S1, paired
# holdout tests against s22 and s27, then the France recipes on s24.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t

free -g | head -2; df -h $SM | tail -1
A="--base v7 --tag _q --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2F_v7 --xenc-dir2 xenc_v7 --xenc-dir3 xenc3Q_v7"
python -m ber.stages.stack build --split train $A
python -m ber.stages.stack tfidf --split train --tag _q
python -m ber.stages.stack build --split test $A
python -m ber.stages.stack tfidf --split test --tag _q
python -m ber.stages.stack train --name s24 --base v7 --tag _q --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
python -m ber.stages.stack predict --name s24
python src/scripts/paired_models.py s22 s24
python src/scripts/paired_models.py s27 s24
R=restore:noise_swap+initials+spelled_legal+glued
python src/scripts/france_variants.py s24 v8r_s24_A --rules typeswap:1.01,thrpn:0.995,protect:0.9 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
python src/scripts/france_variants.py s24 v8r_s24_AR --rules typeswap:1.01,thrpn:0.995,protect:0.9,$R:0.05 --cap 2>&1 | grep -E "fires on|^protect|^restore|with matches|PASS|FAIL|Traceback"
