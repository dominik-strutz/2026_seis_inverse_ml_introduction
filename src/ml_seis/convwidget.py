"""The Part 2 convolution playground (anywidget).

One convolutional filter slides along a real Llaima record, in the notation of the
notebook: the record ``u``, the filter ``w`` of ``K`` weights, the bias ``b`` and the
feature map ``h``, h_t = b + sum_tau w_tau u_{t+tau}. Pick a preset (identity,
smoothing, high-pass, band-pass, edge detectors, slope, bump) or drag the weights
yourself, move the filter along the record, and see the arithmetic of the marked
position: one bar per weight and the products ``w_tau u_{t+tau}``, which add up to the
feature map's sample ``h_t``. A few records per class are sent once at full
resolution (about 0.25 MB); the convolution itself runs in the browser, so every
change updates instantly and the first view survives a static export.

After training, ``set_kernels`` adds the learned first-layer filters to the presets.
"""

from __future__ import annotations

import base64

import anywidget
import numpy as np
import traitlets

from .plotting import style
from .waveform_plotting import _colour
from .widgetkit import esm


def _class_colours(classes) -> list[str]:
    import matplotlib.pyplot as plt

    with plt.style.context(style("open")):
        return [_colour(name) for name in classes]


class ConvFilterPlayground(anywidget.AnyWidget):
    """Slide one filter along real records; ``kernels`` holds learned presets."""

    _esm = esm("convwidget.js")

    classes = traitlets.List().tag(sync=True)
    colours = traitlets.List().tag(sync=True)
    label = traitlets.List().tag(sync=True)  # class index per sent record
    records = traitlets.Unicode().tag(sync=True)  # int16 (n, samples), base64
    samples = traitlets.Int().tag(sync=True)
    rate = traitlets.Float().tag(sync=True)  # Hz
    kernels = traitlets.List().tag(sync=True)  # learned filters, weights each

    def __init__(self, data, per_class: int = 10, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        x = data.u[:, 0].numpy()
        label = data.label.numpy()
        train = data.split.numpy() == 0
        rng = np.random.default_rng(seed)
        keep = []
        for k in range(len(data.classes)):
            members = np.flatnonzero((label == k) & train)
            keep += sorted(rng.choice(members, per_class, replace=False))
        picked = np.clip(x[keep], -1.0, 1.0)
        self.classes = list(data.classes)
        self.colours = _class_colours(data.classes)
        self.label = label[keep].tolist()
        self.samples = int(picked.shape[1])
        self.rate = float(data.sampling_rate)
        quantised = np.round(picked * 32767).astype("<i2")
        self.records = base64.b64encode(quantised.tobytes()).decode()

    def set_kernels(self, weight) -> None:
        """Offer the filters of a trained first layer, a ``(C_out, 1, K)`` weight
        tensor of ``nn.Conv1d(1, C_out, K)``, as presets."""
        w = np.asarray(weight.detach().cpu() if hasattr(weight, "detach") else weight)
        self.kernels = np.round(w[:, 0].astype(np.float64), 4).tolist()


__all__ = ["ConvFilterPlayground"]
