.PHONY: test smoke config
PYTHON ?= python3
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) scripts/smoke.py
config:
	bash -n scripts/compose.sh scripts/run.sh
	@for mode in cpu gpu wsl; do bash scripts/compose.sh --mode $$mode --profile all config --quiet || exit; done
