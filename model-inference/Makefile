.PHONY: test up down logs smoke
PYTHON ?= python3
test:
	$(PYTHON) -m pytest -q
up:
	bash scripts/compose.sh up -d --build
down:
	bash scripts/compose.sh down
logs:
	bash scripts/compose.sh logs --tail=80
smoke:
	$(PYTHON) scripts/smoke.py
