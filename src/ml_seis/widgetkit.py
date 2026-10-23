"""Front-end plumbing shared by the course widgets (not part of the lessons)."""

import base64
import io
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def esm(name: str) -> str:
    """The widget module ``name`` with the shared ``controls.js`` in front of it, so
    both playgrounds use the same colour tokens, header bar and controls."""
    kit = (_HERE / "controls.js").read_text(encoding="utf-8")
    return kit + "\n" + (_HERE / name).read_text(encoding="utf-8")


class Progress:
    """Reports how far a training loop is to a widget's ``progress`` trait (0 to 1),
    at most every ``interval`` seconds; marimo sends each change to the page while
    the cell is still running, so the widget can draw a bar."""

    def __init__(self, widget, interval: float = 0.1):
        self.widget, self.interval, self._last = widget, interval, 0.0

    def __call__(self, fraction: float):
        now = time.perf_counter()
        if fraction < 1 and now - self._last < self.interval:
            return
        self._last = now
        self.widget.progress = round(float(fraction), 3)


def png_data_url(rgba: np.ndarray) -> str:
    """An (ny, nx, 4) colour array as a PNG data URL, row 0 at the bottom."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png", origin="lower")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colour_layer(values, lo, hi, cmap, label, ticks=None) -> dict:
    """A map background for the widgets: ``values`` (ny, nx) coloured between ``lo``
    and ``hi`` with a matplotlib colormap, plus a colour bar image and its labels."""
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    scaled = np.clip((np.asarray(values) - lo) / (hi - lo), 0, 1)
    layer = {
        "png": png_data_url(cm(scaled)),
        "bar": png_data_url(cm(np.linspace(0, 1, 64))[None, :, :]),
        "lo": float(lo),
        "hi": float(hi),
        "label": label,
    }
    if ticks is not None:
        layer["ticks"] = ticks
    return layer
