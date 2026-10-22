# make install | test | smoke | reproduce | tables | figures | bench
PYTHON ?= python
JOBS ?= 4
DEVICE ?= cpu
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.PHONY: install test smoke reproduce tables figures bench clean-smoke

install:
	$(PYTHON) -m pip install -r requirements.txt || $(PYTHON) -m pip install -e .[dev,parquet]
	bash scripts/install_ev2gym.sh
	$(PYTHON) -m pip install --no-deps -e .

test:
	$(PYTHON) -m pytest -q tests

smoke:
	PYTHON=$(PYTHON) ./reproduce.sh --smoke --jobs $(JOBS)

reproduce:
	PYTHON=$(PYTHON) ./reproduce.sh --jobs $(JOBS) --device $(DEVICE)

tables figures:
	@if [ -f ../analysis/scripts/make_all.py ]; then \
	  $(PYTHON) ../analysis/scripts/make_all.py --raw results/raw --out ../analysis; \
	else echo "../analysis/scripts/make_all.py not found (analysis agent A4); nothing to do"; fi

bench:
	$(PYTHON) scripts/benchmark.py

clean-smoke:
	rm -rf results/smoke
