# Common tasks. Run from the repo root; `make help` lists them.
.PHONY: help hooks check test snapshots snapshots-check snapshot

PARTS := $(shell uv run python scripts/snapshots.py list)

help:  ## this list
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

hooks:  ## install the git pre-commit hook (snapshot freshness)
	uv run pre-commit install

check:  ## everything that must pass before handing work back
	uv run marimo check lectures homework solutions
	uv run ruff check . && uv run ruff format --check .
	uv run scripts/check_layouts.py
	uv run pytest tests

test:  ## the physics tests only
	uv run pytest tests

snapshots:  ## regenerate the molab snapshots of ALL published parts, and stamp them
	uv run python scripts/snapshots.py build

snapshot:  ## regenerate one part: make snapshot PART=1_a_locating_with_gradients
	uv run python scripts/snapshots.py build $(PART)

snapshots-check:  ## fail if a published snapshot is stale (what the hook runs)
	uv run python scripts/snapshots.py check
