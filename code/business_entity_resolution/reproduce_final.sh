#!/bin/bash
# Reproduces the final submission (stack s28 with the version-8 France decoding, file v8u_s28_AR; the earlier s22 route is kept in steps 1-8) from the raw TSV files.
# Run from code/business_entity_resolution.
#   BER_DATA=<folder with train/ and test/>  BER_WORK=<scratch, about 120 GB>  bash reproduce_final.sh
# Two Python environments are used: `ber` (Python 3.12, requirements.txt) for everything except the steps marked [torch], which need the
# `pytorch` environment (requirements-gpu.txt: torch with CUDA, transformers, sentencepiece, plus polars, duckdb, pyyaml, scikit-learn).
# Expected time on a 64 vCPU + A10G machine: about 6 hours. Every stage logs to $BER_WORK/runs/runs.jsonl.
set -euxo pipefail
export PYTHONPATH=src
py()  { python -m "$@"; }                 # ber environment
pyt() { "${TORCH_PYTHON:-python}" -m "$@"; }   # [torch] steps: set TORCH_PYTHON to the interpreter of the pytorch environment

# 1. prepare and sample (text normalisation, 250k-S1 sample with folds)
py ber.stages.prepare
py ber.stages.sample

# 2. token blocking: 100 raw candidates per S1 for test and all train S1, learned pruner keeps the best 30
py ber.stages.block --split test --k 100 --out-name test_raw
py ber.stages.block --split train --all-train --k 100 --out-name train_raw
py ber.stages.prune --train
py ber.stages.prune --apply train test

# 3. dense channel for non-Latin names (top 5 per pool record), merged into the candidate shards
pyt ber.stages.dense finetune
pyt ber.stages.dense embed --split train
pyt ber.stages.dense embed --split test
pyt ber.stages.dense retrieve --split train
pyt ber.stages.dense retrieve --split test
py ber.stages.dense merge --split test
py ber.stages.dense merge --split train

# 4. dense channel over name plus address for every record (top 1 per pool record), merged into the shards
pyt ber.stages.dense_all finetune
pyt ber.stages.dense_all embed --split train
pyt ber.stages.dense_all embed --split test
pyt ber.stages.dense_all retrieve --split train
pyt ber.stages.dense_all retrieve --split test
py ber.stages.dense_all merge --split test
py ber.stages.dense_all merge --split train

# 5. first stage v7: features, XGBoost on 850k S1 (sample + 600k), scores for the rest of train (holdout report) and test
py ber.stages.pairs --split train
py ber.stages.pairs --split train --rest
py ber.stages.pairs --split test
py ber.stages.train_gpu --name v7 --extra 600000 --set max_depth=9,eta=0.05,rounds=3000,early_stop=50
py ber.stages.score_rest --name v7
py ber.stages.predict --name v7

# 6. two cross-encoders on the uncertain band of v7 (data in `ber`, fitting and scoring in `pytorch`)
#    e5-small: band 0.02 to 0.98, 400k fit S1 (defaults); e5-base: band 0.01 to 0.99, 700k fit S1 (fit_more=300000), 3 epochs
py ber.stages.xenc data --base v7 --dir xenc_v7
pyt ber.stages.xenc train --dir xenc_v7 --model-dir xenc/model
pyt ber.stages.xenc score --split train --dir xenc_v7 --model-dir xenc/model
pyt ber.stages.xenc score --split test --dir xenc_v7 --model-dir xenc/model
py ber.stages.xenc data --base v7 --dir xenc2_v7 --set fit_more=300000,band_lo=0.01,band_hi=0.99
pyt ber.stages.xenc train --dir xenc2_v7 --base-model intfloat/multilingual-e5-base --model-dir xenc2/model --set fit_more=300000,band_lo=0.01,band_hi=0.99,epochs=3,lr=0.00002,batch=64
pyt ber.stages.xenc score --split train --dir xenc2_v7 --model-dir xenc2/model --set score_batch=256
pyt ber.stages.xenc score --split test --dir xenc2_v7 --model-dir xenc2/model --set score_batch=256

# 7. cross-encoder scores for EVERY shortlisted pair (the band-only scores above feed the older stack): e5-base scoring of all pairs of the shortlist
py ber.stages.xenc data --base v7 --dir xenc2F_v7 --set fit_more=300000,band_lo=0.005,band_hi=1.0
pyt ber.stages.xenc score --split train --dir xenc2F_v7 --model-dir xenc2/model --set score_batch=512
pyt ber.stages.xenc score --split test --dir xenc2F_v7 --model-dir xenc2/model --set score_batch=512

# 8. stack s22 (shortlist, consensus, digit, TF-IDF, decoy edit and carried features, e5-base score on all pairs as xs, e5-small band score as xs2,
#    competition on the refined probability), decision and outputs
py ber.stages.stack build --split train --base v7 --tag _p --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2F_v7 --xenc-dir2 xenc_v7
py ber.stages.stack tfidf --split train --tag _p
py ber.stages.stack build --split test --base v7 --tag _p --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2F_v7 --xenc-dir2 xenc_v7
py ber.stages.stack tfidf --split test --tag _p
py ber.stages.stack train --name s22 --base v7 --tag _p --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
py ber.stages.stack predict --name s22

# 8b. final model s28 (portal submission v8u_s28_AR): two seeds of a symmetric e5-base cross-encoder (random A/B order in training, both orders
#     averaged when scoring) on every short-listed pair, averaged (xs, xs_asym, xs_seed_gap), the band scores of a Qwen3-0.6B cross-encoder
#     (Apache-2.0; fitted by the team's job rn1 on the same pairs, one epoch, bf16: see EXPERIMENTS.md 6.2) as xs4, and the stack s28.
pyt ber.stages.xenc train --dir xenc2_v7 --base-model intfloat/multilingual-e5-base --model-dir xenc2sym/model --set fit_more=300000,band_lo=0.01,band_hi=0.99,epochs=3,lr=0.00002,batch=64,symmetric=1
pyt ber.stages.xenc train --dir xenc2_v7 --base-model intfloat/multilingual-e5-base --model-dir xenc2sym2/model --set fit_more=300000,band_lo=0.01,band_hi=0.99,epochs=3,lr=0.00002,batch=64,symmetric=1,train_seed=1
py ber.stages.xenc data --base v7 --dir xenc2Fsym_v7 --set fit_more=300000,band_lo=0.005,band_hi=1.0
py ber.stages.xenc data --base v7 --dir xenc2Fsym2_v7 --set fit_more=300000,band_lo=0.005,band_hi=1.0
for d in "xenc2Fsym_v7 xenc2sym" "xenc2Fsym2_v7 xenc2sym2"; do set -- $d
  pyt ber.stages.xenc score --split train --dir $1 --model-dir $2/model --set score_batch=512,symmetric=1
  pyt ber.stages.xenc score --split test --dir $1 --model-dir $2/model --set score_batch=512,symmetric=1
done
python src/scripts/avg_xenc.py xenc2SymE_v7 xenc2Fsym_v7 xenc2Fsym2_v7
py ber.stages.xenc data --base v7 --dir xenc3Q_v7 --set fit_more=300000,band_lo=0.01,band_hi=0.99
pyt ber.stages.xenc train --dir xenc2_v7 --base-model Qwen/Qwen3-0.6B --model-dir xenc3Q/model --set fit_more=300000,band_lo=0.01,band_hi=0.99,epochs=1
pyt ber.stages.xenc score --split train --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192
pyt ber.stages.xenc score --split test --dir xenc3Q_v7 --model-dir xenc3Q/model --set score_batch=256,max_len=192
A="--base v7 --tag _eq --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2SymE_v7 --xenc-dir2 xenc_v7 --xenc-dir3 xenc2F_v7 --xenc-dir4 xenc3Q_v7"
py ber.stages.stack build --split train $A
py ber.stages.stack tfidf --split train --tag _eq
py ber.stages.stack build --split test $A
py ber.stages.stack tfidf --split test --tag _eq
py ber.stages.stack train --name s28 --base v7 --tag _eq --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
py ber.stages.stack predict --name s28

# 8c. final stack s29 (the submitted model): s27's features with the Qwen3-0.6B band score in the second cross-encoder slot
A="--base v7 --tag _w --xenc --xcons --xenc-fit-more 300000 --decoy --extra --sub-q 1500000 --xenc-dir xenc2SymE_v7 --xenc-dir2 xenc3Q_v7 --xenc-dir3 xenc2F_v7"
py ber.stages.stack build --split train $A
py ber.stages.stack tfidf --split train --tag _w
py ber.stages.stack build --split test $A
py ber.stages.stack tfidf --split test --tag _w
py ber.stages.stack train --name s29 --base v7 --tag _w --set max_depth=9,eta=0.05,rounds=1500,early_stop=50
py ber.stages.stack predict --name s29

# 9. structural decoding of the test predictions (src/scripts/france_variants.py; France only, see README "Decoding"): decoys that swap the type word of the name
#    (learned vocabulary, from slot occupancy), a stricter cut-off for France's over-confident probabilities, and the training maximum of 5 S2 and 6 S3 matches per S1.
#    The candidate file is unchanged; only matches are removed.
python src/scripts/france_variants.py s22 s22final --rules "typeswap:1.01,thr:0.985" --cap
#    Version 8 of the decoding (README "Decoding (France)") on s28. Every rule was chosen by comparing France's pairs per 1,000 S1 with the US / India
#    rates of the same kind (the labelled holdout keeps those kinds 99%+ true): a French excess is decoys. First the rule-based run, then the pair lists
#    built on it (namesakes on another street, coined copies the 0.9999 cut over-drops), then the final file (v8u_s28_FIN).
R="typeswap:1.01,typeswap:1.01:0.6:30:300,thrpn:0.9999,legalx:1.01"
python src/scripts/france_variants.py s28 v8u_s28_ALL2 --rules "$R,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap
python src/scripts/france_lists.py v8u_s28_ALL2
python src/scripts/france_variants.py s28 v8u_s28_FIN --rules "$R,droplist:fb_ns_ref,droplist:fb_nsnear_ref,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:fb_coined_hi" --cap
#    The submitted file (v8w_s29_FIN, leaderboard 0.987745): the same recipe on s29, lists built from s29's own decoding.
python src/scripts/france_variants.py s29 v8w_s29_ALL2 --rules "$R,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05" --cap
python src/scripts/france_lists.py v8w_s29_ALL2 v8x/s29 s29
python src/scripts/france_variants.py s29 final --rules "$R,droplist:s29/fb_ns_ref,droplist:s29/fb_nsnear_ref,protect:0.9,restore:noise_swap+noise_extra+initials+spelled_legal+glued:0.05,addlist:s29/fb_coined_hi" --cap

# 10. checks (the official validator is also run by `emit` when work/official/validate_submission.py exists)
python src/scripts/check_submission.py "$BER_WORK/output/final" "$BER_DATA/test"
echo "outputs: $BER_WORK/output/final/matching_results.tsv and candidate_pairs.tsv"
