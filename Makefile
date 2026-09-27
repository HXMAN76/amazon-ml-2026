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
