"""Lecture plumbing that is not part of the lesson: slide figures and graph text.

Slide figures (``artifact``) follow one contract, so a figure never sizes itself:

* The artifact file has a ``#stage`` element of a fixed logical size and says so in
  ``<meta name="artifact-size" content="1100x440">``.
* The helper puts the iframe in a *slot*. In slides the slot takes all the room left
  under the text; elsewhere it has the aspect ratio of the stage.
* A small script injected into the iframe scales ``#stage`` to the iframe so that it
  fits (contain) and centres it horizontally and vertically. Nothing is measured across
  the iframe boundary, so slide zoom, reveal scaling and resizing cannot disturb it.
"""

import json
import re
from html import escape
from pathlib import Path

_ARTIFACTS = Path(__file__).parent / "artifacts"

_SLOT_CSS = """<style>
.ml-artifact { position: relative; margin-left: auto; margin-right: auto;
  aspect-ratio: var(--w) / var(--h); max-width: calc(var(--w) * var(--max) * 1px); }
.ml-artifact iframe { position: absolute; inset: 0; width: 100%; height: 100%;
  border: 0; display: block; background: transparent; }
.ml-artifact.fixed { aspect-ratio: auto; max-width: none; }
.reveal .ml-artifact.fill { aspect-ratio: auto; max-width: none; flex: 1 1 0;
  min-height: 12rem; }
.reveal .mo-slide-content:has(.ml-artifact.fill) { align-self: stretch;
  margin-bottom: 0 !important; }
.reveal .mo-slide-content div:not(.contents):has(.ml-artifact.fill) {
  display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
</style>"""

_STAGE_CSS = """<style>
html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
  background: transparent; }
#stage { position: absolute !important; left: 0 !important; top: 0 !important;
  transform-origin: 0 0 !important; }
</style>"""

# Scales #stage (its layout size, which transforms do not change) to the iframe viewport
# and centres it. ``window.STAGE_SCALE`` and the ``stagefit`` event tell the artifact
# the scale, for canvases and measurements that need it.
_FIT_JS = """<script>
(function () {
  const max = (window.ARTIFACT_FIT || {}).max || 2;
  let queued = false;
  function fit() {
    queued = false;
    const stage = document.getElementById("stage");
    if (!stage) return;
    const w = stage.offsetWidth, h = stage.offsetHeight;
    const iw = innerWidth, ih = innerHeight;
    if (!(w && h && iw && ih)) return;
    const s = Math.min(iw / w, ih / h, max);
    const x = (iw - w * s) / 2, y = (ih - h * s) / 2;
    stage.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
    if (window.STAGE_SCALE !== s) {
      window.STAGE_SCALE = s;
      window.dispatchEvent(new Event("stagefit"));
    }
  }
  function queue() { if (!queued) { queued = true; requestAnimationFrame(fit); } }
  addEventListener("resize", queue);
  addEventListener("load", queue);
  addEventListener("DOMContentLoaded", () => {
    const stage = document.getElementById("stage");
    if (stage) new ResizeObserver(queue).observe(stage);
    if (document.fonts) document.fonts.ready.then(queue);
    queue();
  });
})();
</script>"""

_SIZE = re.compile(r'<meta\s+name="artifact-size"\s+content="(\d+)x(\d+)"')


def artifact(
    name: str,
    width: str = "100%",
    height=None,
    gap: str = "2.5rem",
    max_scale: float = 2.0,
    **params,
):
    """Embed ``src/ml_seis/artifacts/<name>.html``; ``params`` reach it as
    ``window.ARTIFACT``. The files ship with the package, so a notebook needs no
    ``artifacts/`` folder next to it (a single notebook file on molab works).

    The figure is scaled to fit its slot and centred in it, never beyond ``max_scale``
    times its logical size. In slides the slot takes the room left under the text; in
    the notebook it has the aspect ratio of the figure. ``width`` is the slot width as a
    share of the slide (``"70%"``). ``height`` (pixels) gives the slot a fixed height
    instead of filling. ``gap`` is the space between the text above and the slot."""
    import marimo as mo

    html = (_ARTIFACTS / f"{name}.html").read_text(encoding="utf-8")
    size = _SIZE.search(html)
    if size is None:
        raise ValueError(f'artifacts/{name}.html has no <meta name="artifact-size">')
    w, h = size.groups()

    head = _STAGE_CSS
    head += f"<script>window.ARTIFACT_FIT = {json.dumps({'max': max_scale})};</script>"
    if params:
        head += f"<script>window.ARTIFACT = {json.dumps(params)};</script>"
    head += _FIT_JS
    html = html.replace("<head>", "<head>" + head, 1)

    fixed = height is not None
    style = f"--w:{w};--h:{h};--max:{max_scale};width:{width};margin-top:{gap};"
    if fixed:
        style += f"height:{height}px;"
    kind = "fixed" if fixed else "fill"
    return mo.Html(
        f'{_SLOT_CSS}<div class="ml-artifact {kind}" style="{style}">'
        f'<iframe srcdoc="{escape(html, quote=True)}" title="{escape(name)}"></iframe>'
        "</div>"
    )


def show_source(*functions):
    """Print the source of functions from ``ml_seis.physics`` as a code block.

    The shown code is the code that runs. Anything that is not a public function of
    ``physics`` is refused, so a slide can never show something else than the physics.
    In the slide layout, set the cell to ``skip`` to keep it in the notebook only."""
    import inspect

    import marimo as mo

    from . import physics

    for function in functions:
        name = getattr(function, "__name__", None)
        if name not in physics.__all__ or getattr(physics, name) is not function:
            raise ValueError(f"{function!r} is not a function of ml_seis.physics")
    sources = (inspect.getsource(function).rstrip() for function in functions)
    return mo.md("```python\n" + "\n\n\n".join(sources) + "\n```")


def try_it(tasks):
    """Notebook-only task list of a playground, one collapsed item per task.

    ``tasks`` is a list of ``(concept, question, setup, look_for)`` markdown strings.
    The label is the concept and the question, so the reader predicts before opening
    the setup. The cell that shows it is ``skip`` in the slide layout: tasks never go
    on a slide."""
    import marimo as mo

    return mo.accordion(
        {
            f"**{concept}.** {question}": mo.md(
                f"**Setup.** {setup}\n\n**Look for.** {look_for}"
            )
            for concept, question, setup, look_for in tasks
        }
    )


def graph_text(grad_fn) -> str:
    """Indented text tree of the autograd graph behind ``tensor.grad_fn``."""
    seen = set()

    def walk(fn, depth):
        name = type(fn).__name__
        if id(fn) in seen:
            return ["  " * depth + name + "  (shown above)"]
        seen.add(id(fn))
        lines = ["  " * depth + name]
        for child, _ in fn.next_functions:
            if child is not None:
                lines += walk(child, depth + 1)
        return lines

    return "\n".join(walk(grad_fn, 0))
