# v9 (CPU lane jobs2): label-free France audit of every output file with a known portal score (backtest) and of tonight's candidates.
source <(aws s3 cp s3://sagemaker-us-east-1-645311222213/ber/queue/jobs/_header.sh -)
export BER_WORK=$SM/work_t
ls $BER_WORK/output | tr '\n' ' '; echo
python src/scripts/france_audit.py s22t2c s22sx v8u_s22F12n_AR v8u_s27_AR s28 v8u_s28_AR v8u_s28_FIN v8u_s28M_FIN v8u_s28L_FIN v8u_s28P2_FIN 2>&1 | grep -vE "Deprecation|empty_as_null"
