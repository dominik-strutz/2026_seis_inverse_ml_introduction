"""A stepper for going through numbered samples (anywidget).

``Stepper(40, label="record")`` shows ``RECORD (−) [ 1 ] / 40 (+)``: round minus and
plus buttons around a number that can also be typed. Use it wherever a student picks one
record, event or trace by its number. A slider is too coarse for that, and its ends give
no feel for the count.
In a marimo notebook::

    record = mo.ui.anywidget(Stepper(40, label="record"))
    record.value["value"]  # the chosen number, start..stop (1-based by default)
"""

from __future__ import annotations

import anywidget
import traitlets

from .widgetkit import esm


class Stepper(anywidget.AnyWidget):
    """Minus, a number and plus; ``value`` runs from ``start`` to ``stop``."""

    _esm = esm("stepperwidget.js")

    label = traitlets.Unicode("").tag(sync=True)
    start = traitlets.Int(1).tag(sync=True)
    stop = traitlets.Int(1).tag(sync=True)
    value = traitlets.Int(1).tag(sync=True)

    def __init__(
        self, stop: int, label: str = "", start: int = 1, value: int | None = None
    ):
        if stop < start:
            raise ValueError(f"stop ({stop}) is below start ({start})")
        value = start if value is None else value
        super().__init__(
            label=label, start=start, stop=stop, value=min(max(value, start), stop)
        )
