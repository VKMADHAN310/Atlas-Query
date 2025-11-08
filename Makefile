PYTHON ?= python3
PIP ?= pip3
NPM ?= npm

.PHONY: dev run-api test run-frontend frontend-install run-all

dev:
	$(PIP) install -r requirements.txt

run-api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

# Install frontend dependencies
frontend-install:
	cd interactive-us-county-map && $(NPM) install

# Run only the frontend (Vite dev server)
run-frontend: frontend-install
	cd interactive-us-county-map && $(NPM) run dev

# Run backend (FastAPI) and frontend (Vite) together
# - Ensures Python and Node deps are installed first
# - Starts both processes and cleans them up on exit (CTRL+C)
run-all: dev frontend-install
	bash -lc 'set -e; \
	  echo "Starting backend on http://0.0.0.0:8000"; \
	  (uvicorn api:app --reload --host 0.0.0.0 --port 8000) & \
	  BACK_PID=$$!; \
	  echo "Starting frontend (Vite)"; \
	  (cd interactive-us-county-map && $(NPM) run dev) & \
	  FRONT_PID=$$!; \
	  cleanup() { \
	    echo "\nStopping servers..."; \
	    kill $$BACK_PID $$FRONT_PID 2>/dev/null || true; \
	    wait $$BACK_PID $$FRONT_PID 2>/dev/null || true; \
	  }; \
	  trap cleanup INT TERM EXIT; \
	  echo "Both servers started. Press CTRL+C to stop."; \
	  wait'

