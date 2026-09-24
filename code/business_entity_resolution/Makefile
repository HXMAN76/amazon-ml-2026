# Stage DAG with hash-based caching. A stage reruns exactly when its params section or its source changes.
#   make prepare sample        (BER_DATA=dataset BER_WORK=work by default)
PY ?= python
export PYTHONPATH := src
WORK ?= $(or $(BER_WORK),work)
export BER_WORK := $(WORK)
STAMPS := $(WORK)/.stamps
DONE := $(WORK)/.done

.PHONY: all prepare sample block_eval test clean-stamps FORCE
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
	$(PY) -m pytest -q tests
clean-stamps:
	rm -rf $(STAMPS) $(DONE)
