"""Embed the slide layout of each lecture into its notebook file.

molab, and anyone who downloads a single notebook, gets one file: the
``layout_file`` reference to ``lectures/layouts/`` dangles, marimo falls back
to its default view and the slide deck (arrangement, skipped cells) is lost.
marimo also accepts the layout itself in ``layout_file``, as a base64 data
URI -- the same form ``marimo export`` writes into its output files -- so this
script copies each ``lectures/layouts/<part>.slides.json`` into its notebook
header, wrapped to stay under the line limit.

The JSON files stay the source of truth: the layout editor writes them and
``scripts/check_layouts.py`` checks them. Editing a layout in marimo rewrites
``layout_file`` back to the file path, so run ``build`` after; ``check`` (the
pre-commit hook) fails while a notebook carries an older, path-only or missing
layout.

usage (from the repo root; ``make layouts`` wraps ``build``)

    uv run python scripts/embed_layouts.py build [part ...]   re-embed
    uv run python scripts/embed_layouts.py check [part ...]   fail if stale
"""

import ast
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"
PREFIX = "data:application/json;base64,"
# layout_file="..." (the editor's form) or layout_file=( ... ), (this script's)
KWARG = re.compile(r'layout_file=(?:"[^"]*"|\([^)]*\)),')
CHUNK = 74  # 8 indent + 2 quotes + 74 = 84 characters, under the line limit


def parts() -> list[str]:
    """Every lecture whose slide layout exists as a JSON file."""
    return sorted(
        path.stem
        for path in LECTURES.glob("*.py")
        if (LECTURES / "layouts" / f"{path.stem}.slides.json").is_file()
    )


def _uri(part: str) -> str:
    """The layout of ``part`` as a data URI (minified JSON, base64)."""
    layout = json.loads(
        (LECTURES / "layouts" / f"{part}.slides.json").read_text(encoding="utf-8")
    )
    mini = json.dumps(layout, separators=(",", ":"))
    return PREFIX + base64.b64encode(mini.encode()).decode("ascii")


def _literal(uri: str) -> str:
    """``layout_file=(...),`` with the URI wrapped in string chunks."""
    chunks = [uri[i : i + CHUNK] for i in range(0, len(uri), CHUNK)]
    body = "\n".join(f'        "{chunk}"' for chunk in chunks)
    return f"layout_file=(\n{body}\n    ),"


def _embedded(path: Path) -> str | None:
    """The layout_file value a notebook declares (None if it declares none)."""
    for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            for kw in stmt.value.keywords:
                if kw.arg == "layout_file" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def _decoded(value: str | None) -> object | None:
    """The layout a layout_file value carries, or None if it is not embedded."""
    if isinstance(value, str) and value.startswith(PREFIX):
        return json.loads(base64.b64decode(value[len(PREFIX) :]))
    return None


def build(names: list[str]) -> int:
    for part in names:
        notebook = LECTURES / f"{part}.py"
        text = notebook.read_text(encoding="utf-8")
        found = KWARG.findall(text)
        if len(found) != 1:
            print(
                f"FAILED: {part}; expected one layout_file argument, found {len(found)}"
            )
            print("Save a layout from marimo first, then re-run.")
            return 1
        literal = _literal(_uri(part))
        if found[0] != literal:
            notebook.write_text(KWARG.sub(literal, text, count=1), encoding="utf-8")
            print(f"embedded the layout of {part}")
    return 0


def check(names: list[str]) -> int:
    stale = []
    for part in names:
        layout = json.loads(
            (LECTURES / "layouts" / f"{part}.slides.json").read_text(encoding="utf-8")
        )
        if _decoded(_embedded(LECTURES / f"{part}.py")) != layout:
            stale.append(part)
    if stale:
        print("The slide layout embedded in these notebooks is out of date:")
        for part in stale:
            print(f"  {part}")
        print("Run `make layouts`, then commit.")
        print("(Skip once with `SKIP=layouts git commit ...` if you must.)")
        return 1
    return 0


def main(argv: list[str]) -> int:
    command, names = (argv[0] if argv else "check"), argv[1:] or parts()
    if command == "list":
        print("\n".join(parts()))
        return 0
    if command not in ("build", "check"):
        print(__doc__)
        return 2
    return (build if command == "build" else check)(names)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
