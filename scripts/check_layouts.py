"""Check that the slide layouts of the lectures match their notebooks.

A layout lists one entry per cell, in notebook order, so any added, removed or moved
cell shifts every later slide. This reports

* a different number of cells and layout entries,
* cells that call ``show_source(`` or ``try_it(`` (code and playground tasks) but are
  not ``skip``: those belong in the notebook only,
* layout entries without a ``type`` (marimo shows them as blank slides),
* an Explore cell ("... things to look for") that is not part of a ``slide`` cell: it
  is shown on the playground's own slide, never as a click-to-reveal ``fragment``,
* ``slide`` / ``fragment`` cells that would be blank: code cells (``hide_code`` not
  set) without ``showCode`` that display nothing.

usage: uv run scripts/check_layouts.py [lectures/1_a_*.py ...]   (default: all parts)
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPLORE = re.compile(r"things to look for")
NOTEBOOK_ONLY = re.compile(r"\b(show_source|try_it)\(")


def cells_of(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                if "app.cell" in ast.unparse(deco):
                    yield node, "hide_code=True" in ast.unparse(deco)


def shows_output(node: ast.FunctionDef) -> bool:
    """Does the cell end in an expression (which marimo displays)?"""
    last = [n for n in node.body if not isinstance(n, ast.Return)]
    return bool(last) and isinstance(last[-1], ast.Expr)


def check(path: Path) -> int:
    layout = json.loads(
        (ROOT / "lectures" / "layouts" / f"{path.stem}.slides.json").read_text()
    )
    entries = layout["data"]["cells"]
    cells = list(cells_of(path))
    problems = []
    if len(cells) != len(entries):
        problems.append(f"{len(cells)} cells but {len(entries)} layout entries")
    for i, ((node, hidden), entry) in enumerate(zip(cells, entries, strict=False)):
        if "type" not in entry:
            problems.append(f"cell {i} has no type in the layout")
            continue
        if EXPLORE.search(ast.unparse(node)) and entry["type"] != "slide":
            problems.append(f"cell {i} has the Explore text but is a {entry['type']}")
        if entry["type"] == "skip":
            continue
        if NOTEBOOK_ONLY.search(ast.unparse(node)):
            problems.append(f"cell {i} shows code or tasks but is a {entry['type']}")
        if not hidden and not entry.get("showCode") and not shows_output(node):
            problems.append(f"cell {i} is a {entry['type']} but shows nothing")
    for problem in problems:
        print(f"{path.name}: {problem}")
    return len(problems)


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        (ROOT / "lectures").glob("[0-9]_[a-z]_*.py")
    )
    sys.exit(1 if sum(check(p) for p in paths) else 0)
