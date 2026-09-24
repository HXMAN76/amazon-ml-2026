# Business Entity Resolution: baseline v0

Pipeline: normalise -> TF-IDF char-ngram kNN blocking (per country, S2+S3 pooled) -> 49 pair features
(rapidfuzz, TF-IDF cosines, numeric/PIN overlap, per-S1 and per-candidate rank/gap) -> LightGBM
(grouped 5-fold OOF) -> threshold and one-to-one choice tuned on OOF macro F_0.5 -> TSV outputs.
No external data or lookups. No pretrained model in v0.

## Run
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m ber.run train   --data dataset --model models/v0        # prints blocking recall + OOF F0.5
python -m ber.run predict --data dataset --model models/v0 --out output
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test   # official checker
pytest tests
```
`--data` must contain `train/` and `test/` in the competition layout (TSV files).
