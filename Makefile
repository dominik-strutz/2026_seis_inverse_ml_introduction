# Common tasks. Run from the repo root; `make help` lists them.
.PHONY: help hooks check test layouts snapshots snapshots-check snapshot

PARTS := $(shell uv run python scripts/snapshots.py list)

help:  ## this list
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

hooks:  ## install the pre-commit checks for snapshots and embedded layouts
	uv run pre-commit install

check:  ## everything that must pass before handing work back
	uv run marimo check lectures $(wildcard homework solutions)
	uv run ruff check . && uv run ruff format --check .
	uv run scripts/check_layouts.py
	uv run python scripts/embed_layouts.py check
	uv run python scripts/snapshots.py check
	uv run pytest tests

test:  ## the physics tests only
	uv run pytest tests

layouts:  ## re-embed the slide layouts into the notebooks (molab sees one file)
	uv run python scripts/embed_layouts.py build

snapshots: layouts  ## embed layouts, then regenerate all molab snapshots and stamps
	uv run python scripts/snapshots.py build

snapshot: layouts  ## embed layouts, then regenerate one part: make snapshot PART=...
	uv run python scripts/snapshots.py build $(PART)

snapshots-check:  ## fail if a published snapshot is stale (what the hook runs)
	uv run python scripts/snapshots.py check
