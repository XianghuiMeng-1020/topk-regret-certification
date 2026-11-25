# Two pinned environments, both CPython 3.11:
#   .venv            requirements.lock            numpy 1.24.4  (real-data experiments)
#   .venv-synthetic  requirements-synthetic.lock  numpy 2.1.3   (synthetic experiments)

PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python
VENV_S := .venv-synthetic
PY_S   := $(VENV_S)/bin/python
export PYTHONPATH := src
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONHASHSEED := 0

.PHONY: help setup verify test data reproduce check-claims figures clean

help:
	@echo "make setup         create .venv and .venv-synthetic from the lock files (needs $(PYTHON))"
	@echo "make verify        environment, clean-room imports, committed outputs, source objects if present"
	@echo "make test          full pytest suite in both environments"
	@echo "make data          download the public source objects (about 5.9 GB) and verify SHA-256"
	@echo "make reproduce     rerun every experiment into reproductions/<run id>/ and compare with the committed outputs"
	@echo "make check-claims  recompute every registered number from the ledger rows"
	@echo "make figures       redraw figures/*.svg from the committed outputs"

$(PY):
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11.x is required for numpy 1.24.4'"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --disable-pip-version-check --quiet -r requirements.lock

$(PY_S):
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11.x is required'"
	$(PYTHON) -m venv $(VENV_S)
	$(PY_S) -m pip install --disable-pip-version-check --quiet -r requirements-synthetic.lock

setup: $(PY) $(PY_S)
	$(PY) scripts/verify_environment.py
	$(PY_S) scripts/verify_environment.py --lock requirements-synthetic.lock

verify: $(PY)
	$(PY) scripts/verify_environment.py
	$(PY) scripts/verify_imports.py
	$(PY) scripts/verify_results.py
	$(PY) scripts/bootstrap_data.py --check-only --allow-missing

test: $(PY) $(PY_S)
	$(PY_S) -m pytest
	$(PY) -m pytest

data: $(PY)
	$(PY) scripts/bootstrap_data.py

reproduce: $(PY) $(PY_S)
	$(PY) scripts/bootstrap_data.py --check-only
	$(PY) scripts/reproduce.py --synthetic-python $(PY_S)

check-claims: $(PY)
	$(PY) scripts/check_claims.py

figures: $(PY)
	$(PY) scripts/make_figures.py

clean:
	rm -rf reproductions .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
