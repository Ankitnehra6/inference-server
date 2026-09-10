.DEFAULT_GOAL := help
VENV ?= .venv
PY   := $(VENV)/bin/python

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3.12 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip

.PHONY: install
install: $(VENV) ## Create the venv and install in editable mode
	$(PY) -m pip install -q -e ".[dev]"

.PHONY: test
test: install ## Run the test suite
	$(PY) -m pytest -q

.PHONY: bench
bench: install ## Run the full benchmark (virtual clock, a few seconds)
	$(PY) -m inference_server.bench.run

.PHONY: bench-quick
bench-quick: install ## Just the scheduler comparison
	$(PY) -m inference_server.bench.run --quick

.PHONY: serve
serve: install ## Start the server and live view on :8000
	@echo "http://localhost:$${PORT:-8000}"
	$(PY) -m inference_server.server --port $${PORT:-8000}

.PHONY: lint
lint: install ## Check formatting and lints
	$(PY) -m pip install -q ruff && $(PY) -m ruff check src tests

.PHONY: clean
clean: ## Remove the venv and build artefacts
	rm -rf $(VENV) build dist src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
