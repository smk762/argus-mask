UV ?= uv

.PHONY: help install lint format test build clean head schema demo

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install with test extras
	$(UV) venv
	$(UV) pip install -e ".[dev,server,cli]"

lint:  ## Ruff check + format check
	$(UV)x ruff@0.15.16 check src/ tests/ demo/
	$(UV)x ruff@0.15.16 format --check src/ tests/ demo/

format:  ## Ruff autoformat + autofix
	$(UV)x ruff@0.15.16 format src/ tests/ demo/
	$(UV)x ruff@0.15.16 check --fix src/ tests/ demo/

test:  ## Run the test suite
	$(UV) run --no-sync pytest --tb=short -q

head:  ## Download the stock head into assets/ (checked against its lock file)
	python3 scripts/fetch_stock_head.py

demo:  ## Build the hosted demo into demo/out (stock heads from assets/ if checked there, else fetched)
	python3 demo/build.py --faces-from assets

schema:  ## Regenerate the committed wire schema
	$(UV) run --no-sync argus-mask schema

build:  ## Build sdist + wheel
	$(UV) build

clean:  ## Remove build + cache artifacts
	rm -rf dist build .venv *.egg-info .pytest_cache .ruff_cache src/argus_mask/_version.py
