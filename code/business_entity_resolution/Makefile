# Stage DAG with hash-based caching. A stage reruns exactly when its params section or its source changes.
#   make prepare sample        (BER_DATA=dataset BER_WORK=work by default)
PY ?= python
export PYTHONPATH := src
WORK ?= $(or $(BER_WORK),work)
export BER_WORK := $(WORK)
STAMPS := $(WORK)/.stamps
DONE := $(WORK)/.done

.PHONY: all prepare sample block_eval reproduce test clean-stamps FORCE
all: sample
FORCE:

$(STAMPS)/prepare.hash: FORCE
	@$(PY) -m ber.stamp prepare src/ber/text.py src/ber/stages/prepare.py
$(STAMPS)/sample.hash: FORCE
	@$(PY) -m ber.stamp sample src/ber/stages/sample.py

$(STAMPS)/block_eval.hash: FORCE
	@$(PY) -m ber.stamp block_eval --sections=block,block_eval src/ber/stages/block.py src/ber/stages/block_eval.py

$(DONE)/prepare: $(STAMPS)/prepare.hash
	$(PY) -m ber.stages.prepare
	@mkdir -p $(DONE) && touch $@
$(DONE)/sample: $(DONE)/prepare $(STAMPS)/sample.hash
	$(PY) -m ber.stages.sample
	@mkdir -p $(DONE) && touch $@

$(DONE)/block_eval: $(DONE)/sample $(STAMPS)/block_eval.hash
	$(PY) -m ber.stages.block_eval
	@mkdir -p $(DONE) && touch $@

prepare: $(DONE)/prepare
sample: $(DONE)/sample
block_eval: $(DONE)/block_eval
test:
	$(PY) -m pytest -q src/tests
clean-stamps:
	rm -rf $(STAMPS) $(DONE)

# End-to-end reproduction of output/matching_results.tsv and output/candidate_pairs.tsv from the raw TSVs.
# Needs BER_DATA (folder with train/ and test/), BER_WORK (scratch, about 60 GB) and, for speed, a CUDA GPU
# (XGBoost falls back to CPU automatically). Model name defaults to v0; outputs land in $(WORK)/output/$(NAME).
NAME ?= v1
reproduce: sample
	$(PY) -m ber.stages.block --split test
	$(PY) -m ber.stages.block --split train --all-train
	$(PY) -m ber.stages.pairs --split train
	$(PY) -m ber.stages.train_gpu --name $(NAME)
	$(PY) -m ber.stages.pairs --split test
	$(PY) -m ber.stages.predict --name $(NAME)
	@echo "outputs: $(WORK)/output/$(NAME)/matching_results.tsv and candidate_pairs.tsv"

# bs line (v6). One target per checkpointed stage (aws/sm/entry.py runs `make <stage>` and saves a checkpoint after each).
# bs_s6 = the s6 recipe (v5.md section 7) rebuilt as the paired baseline; BS_V / BS_S name the first-stage and stacked models.
BS_V ?= bs_v5
BS_S ?= bs_s6
.PHONY: bs_prepare bs_block bs_dense bs_dense_all bs_first bs_stack
bs_prepare:
	$(PY) -m ber.stages.prepare
	$(PY) -m ber.stages.sample
bs_block:
	$(PY) -m ber.stages.block --split test --k 100 --out-name test_raw
	$(PY) -m ber.stages.block --split train --all-train --k 100 --out-name train_raw
	$(PY) -m ber.stages.prune --train
	$(PY) -m ber.stages.prune --apply train test
bs_dense:
	$(PY) -m ber.stages.dense finetune
	for s in train test; do $(PY) -m ber.stages.dense embed --split $$s && $(PY) -m ber.stages.dense retrieve --split $$s && $(PY) -m ber.stages.dense merge --split $$s || exit 1; done
bs_dense_all:
	$(PY) -m ber.stages.dense_all finetune
	for s in train test; do $(PY) -m ber.stages.dense_all embed --split $$s && $(PY) -m ber.stages.dense_all retrieve --split $$s || exit 1; done
	$(PY) -m ber.stages.dense_all report
	for s in test train; do $(PY) -m ber.stages.dense_all merge --split $$s || exit 1; done
bs_first:
	$(PY) -m ber.stages.pairs --split train
	$(PY) -m ber.stages.train_gpu --name $(BS_V)
	$(PY) -m ber.stages.pairs --split train --rest
	$(PY) -m ber.stages.score_rest --name $(BS_V)
	$(PY) -m ber.stages.pairs --split test
	$(PY) -m ber.stages.predict --name $(BS_V)
bs_stack:
	for s in train test; do $(PY) -m ber.stages.stack build --split $$s --base $(BS_V) && $(PY) -m ber.stages.stack tfidf --split $$s || exit 1; done
	$(PY) -m ber.stages.stack train --name $(BS_S) --base $(BS_V)
	$(PY) -m ber.stages.stack predict --name $(BS_S)

# v6 steps after the baseline.
.PHONY: bs_density bs_tw bs_errors bs_encoder bs_xenc bs_final_stack bs_set
bs_density:
	$(PY) src/scripts/density_check.py $(BS_S)
bs_tw:  # holdout score and threshold re-weighted to the test's country and look-alike mix
	$(PY) src/scripts/test_weighted.py $(BS_S)
bs_errors:  # where the final stack still loses on the locked holdout (blocking vs matcher, error examples)
	$(PY) src/scripts/error_analysis.py bs_w2 holdout
bs_encoder:  # second name+address channel (dense_all2) and the first stage on it
	$(PY) -m ber.stages.dense_all finetune --tag 2
	for s in train test; do $(PY) -m ber.stages.dense_all embed --split $$s --tag 2 && $(PY) -m ber.stages.dense_all retrieve --split $$s --tag 2 || exit 1; done
	$(PY) -m ber.stages.dense_all report --tag 2
	for s in test train; do $(PY) -m ber.stages.dense_all merge --split $$s --tag 2 || exit 1; done
	$(PY) -m ber.stages.pairs --split train
	$(PY) -m ber.stages.train_gpu --name bs_w1
	$(PY) -m ber.stages.pairs --split train --rest
	$(PY) -m ber.stages.score_rest --name bs_w1
	$(PY) -m ber.stages.pairs --split test
	$(PY) -m ber.stages.predict --name bs_w1
bs_xenc:
	$(PY) -m ber.stages.xenc finetune --base bs_w1
	for s in train test; do $(PY) -m ber.stages.xenc score --split $$s --base bs_w1 || exit 1; done
	$(PY) -m ber.stages.xenc report --base bs_w1
bs_final_stack:
	for s in train test; do $(PY) -m ber.stages.stack build --split $$s --base bs_w1 --xenc --decoy --extra --tag w && $(PY) -m ber.stages.stack tfidf --split $$s --tag w || exit 1; done
	$(PY) -m ber.stages.stack train --name bs_w2 --base bs_w1 --tag w
	$(PY) -m ber.stages.stack predict --name bs_w2 --tag w
	$(PY) src/scripts/paired_models.py $(BS_S) bs_w2 || true
bs_set:
	$(PY) -m ber.stages.setmodel train --name bs_set --stack bs_w2
	$(PY) -m ber.stages.setmodel blend --name bs_final --stack bs_w2 --set bs_set
