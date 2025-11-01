PYTHON ?= python3
PIP ?= pip3

.PHONY: dev run-api test

dev:
	$(PIP) install -r requirements.txt

run-api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

