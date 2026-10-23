"""check_layouts.py: code and playground tasks must stay out of the slides."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layouts.py"

NOTEBOOK = """import marimo

app = marimo.App()


@app.cell(hide_code=True)
def _(show_source, travel_times):
    show_source(travel_times)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("text")
    return
"""


def run_check(tmp_path, types):
    spec = importlib.util.spec_from_file_location("check_layouts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "lectures" / "layouts").mkdir(parents=True)
    notebook = tmp_path / "lectures" / "1_x_test.py"
    notebook.write_text(NOTEBOOK)
    layout = {"type": "slides", "data": {"cells": [{"type": t} for t in types]}}
    (tmp_path / "lectures" / "layouts" / "1_x_test.slides.json").write_text(
        json.dumps(layout)
    )
    return module.check(notebook)


def test_source_cell_must_be_skip(tmp_path):
    assert run_check(tmp_path, ["slide", "slide"]) == 1


def test_source_cell_skipped_passes(tmp_path):
    assert run_check(tmp_path, ["skip", "slide"]) == 0
