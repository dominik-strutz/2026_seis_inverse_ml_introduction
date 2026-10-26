"""Keep the molab session snapshots of the published parts in step with their code.

A snapshot (``lectures/__marimo__/session/<part>.json``) embeds the outputs of a part,
so it goes stale whenever the notebook, or any ``ml_seis`` code, figure or widget file
it uses, changes. ``snapshots.json`` next to the snapshots stores a hash of those
inputs for each published part, written only by ``build``. ``check`` (the pre-commit
hook) fails when a hash no longer matches, so a stale snapshot cannot be committed.

The inputs of a part: the notebook; every ``ml_seis`` module it imports, directly or
through other ``ml_seis`` modules; the files of those modules with the same name (a
widget's ``.js``); the HTML artefacts it embeds with ``artifact("name")``; and the
shared ``controls.js``, styles and data files.

usage (from the repo root; ``make snapshots`` and ``make snapshots-check`` wrap it)

    uv run python scripts/snapshots.py build [part ...]   regenerate and stamp
    uv run python scripts/snapshots.py check [part ...]   fail if a stamp is stale
    uv run python scripts/snapshots.py list               the published parts
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "ml_seis"
SESSIONS = ROOT / "lectures" / "__marimo__" / "session"
STAMPS = SESSIONS / "snapshots.json"

# the published parts: keep in step with the un-ignored snapshots in .gitignore
PARTS = [
    "1_a_locating_with_gradients",
    "1_b_learning_the_inverse_map",
    "1_c_learning_a_posterior",
]
ALWAYS = ["controls.js", "ridgecrest_prior.npz", "styles"]  # used by many modules


def _imports(path: Path) -> set[str]:
    """Names of the ml_seis modules (file stems) that a Python file imports."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "ml_seis" and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")
            if node.level == 1 and path.parent == PKG:  # from .x import y
                if module[0]:
                    found.add(module[0])
                else:
                    found.update(a.name for a in node.names)
            elif node.level == 0 and module[0] == "ml_seis":
                if len(module) > 1:
                    found.add(module[1])
                else:  # from ml_seis import physics, sample_prior
                    found.update(a.name for a in node.names)
    return {name for name in found if (PKG / f"{name}.py").exists()}


def inputs(part: str) -> list[Path]:
    """Every file whose content decides what the snapshot of ``part`` shows."""
    notebook = ROOT / "lectures" / f"{part}.py"
    files = {notebook}
    todo, seen = sorted(_imports(notebook)), set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo += sorted(_imports(PKG / f"{name}.py") - seen)
    for name in seen | {"__init__"}:
        files |= {p for p in PKG.glob(f"{name}.*") if p.is_file()}  # x.py, x.js, ...
    for name in re.findall(
        r"""artifact\(\s*["']([\w.-]+)["']""", notebook.read_text(encoding="utf-8")
    ):
        files.add(PKG / "artifacts" / f"{name}.html")
    for extra in ALWAYS:
        path = PKG / extra
        files |= set(path.rglob("*")) if path.is_dir() else {path}
    return sorted(p for p in files if p.is_file())


def digest(part: str) -> str:
    h = hashlib.sha256()
    for path in inputs(part):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _stamps() -> dict:
    return json.loads(STAMPS.read_text()) if STAMPS.exists() else {}


def build(parts: list[str]) -> int:
    for part in parts:
        print(f"regenerating the snapshot of {part} ...", flush=True)
        run = subprocess.run(
            ["uv", "run", "marimo", "export", "session", "--no-sandbox",
             "--force-overwrite", "--no-continue-on-error",
             str(ROOT / "lectures" / f"{part}.py")],
            cwd=ROOT,
        )  # fmt: skip
        snapshot = SESSIONS / f"{part}.py.json"
        if run.returncode or not snapshot.exists():
            print(f"FAILED: {part}; its stamp is left as it was")
            return 1
        stamps = _stamps()
        stamps[part] = digest(part)  # the inputs as they were when it ran
        STAMPS.write_text(json.dumps(stamps, indent=2, sort_keys=True) + "\n")
    return 0


def check(parts: list[str]) -> int:
    stamps, stale = _stamps(), []
    for part in parts:
        if not (SESSIONS / f"{part}.py.json").exists() or stamps.get(part) != digest(
            part
        ):
            stale.append(part)
    if stale:
        print("The molab snapshot of these parts is out of date:")
        for part in stale:
            print(f"  {part}")
        print("Run `make snapshots`, then add lectures/__marimo__/session/ and commit.")
        print("(Skip once with `SKIP=snapshots git commit ...` if you must.)")
        return 1
    return 0


def main(argv: list[str]) -> int:
    command, parts = (argv[0] if argv else "check"), argv[1:] or PARTS
    unknown = [p for p in parts if p not in PARTS]
    if command == "list":
        print("\n".join(PARTS))
        return 0
    if unknown or command not in ("build", "check"):
        print(__doc__)
        return 2
    return (build if command == "build" else check)(parts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
