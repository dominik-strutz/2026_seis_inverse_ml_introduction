"""The Part 2 Llaima explorer (anywidget).

Browse the whole labelled data set at once: one row per class, three records each,
paged together (a class that has run out shows empty frames), with the balance bars
lining up with the rows and the train/validation/test chips filtering. Clicking a record
opens it: its amplitude spectrum against the class spectra, its spectrogram, or its
waveform at full resolution.

The labels, splits and per-record features are sent once, so paging and filtering run
in the browser. The grid asks Python (``want``) for the actual samples of the records
on the page and the next one, and the record you open gets its spectrum and
spectrogram computed in Python (``selected``).
"""

from __future__ import annotations

import base64

import anywidget
import matplotlib
import numpy as np
import traitlets

from .plotting import style
from .waveform_metrics import (
    SPEC_WINDOW,
    amplitude_spectrum,
    record_features,
    spectrogram,
    spectrogram_axes,
)
from .waveform_plotting import _colour
from .widgetkit import esm, png_data_url

BAND = (10, 90)  # percentiles of the per-record spectra drawn as a band
SPEC_RANGE = 60.0  # dB below the peak where the colour scale starts


def _class_colours(classes) -> list[str]:
    import matplotlib.pyplot as plt

    with plt.style.context(style("open")):
        return [_colour(name) for name in classes]


def _rounded(a, digits: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


class LlaimaExplorer(anywidget.AnyWidget):
    """Browse all Llaima records; ``selected`` is the open record (-1: none)."""

    _esm = esm("llaimawidget.js")

    # the data set, sent once (Python -> JS)
    classes = traitlets.List().tag(sync=True)
    names = traitlets.List().tag(sync=True)  # long names of the classes
    colours = traitlets.List().tag(sync=True)
    label = traitlets.List().tag(sync=True)  # class index per record
    split = traitlets.List().tag(sync=True)  # 0 train, 1 validation, 2 test, per record
    features = traitlets.Dict().tag(sync=True)  # name -> value per record
    duration = traitlets.Float().tag(sync=True)  # seconds per record
    class_spectra = traitlets.Dict().tag(sync=True)  # freq, mean, low, high per class
    # the open record (JS -> Python), and what Python sends back
    selected = traitlets.Int(-1).tag(sync=True)
    detail = traitlets.Dict().tag(sync=True)
    # the records the grid wants at full resolution (JS -> Python), and the answer
    want = traitlets.List().tag(sync=True)
    chunk = traitlets.Dict().tag(sync=True)

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self._x = data.u[:, 0].numpy()  # (N, samples)
        self._rate = float(data.sampling_rate)
        # samples before the flat padding: the spectrogram covers only these
        self._live = (
            np.full(len(self._x), self._x.shape[1])
            if data.n_live is None
            else data.n_live.numpy()
        )
        n, length = self._x.shape

        self.classes = list(data.classes)
        self.names = [self._long_name(c) for c in data.classes]
        self.colours = _class_colours(data.classes)
        self.label = data.label.tolist()
        self.split = data.split.tolist()
        features = record_features(self._x, self._rate)
        self.features = {k: _rounded(v, 4) for k, v in features.items()}
        self.duration = length / self._rate
        self.class_spectra = self._class_spectra(data.label.numpy())

    @staticmethod
    def _long_name(code: str) -> str:
        from .waveform_data import LLAIMA_NAMES

        return LLAIMA_NAMES.get(code, code)

    def _class_spectra(self, label) -> dict:
        """Mean and percentile band of the per-record normalised spectra."""
        amplitude = np.abs(np.fft.rfft(self._x.astype(np.float64), axis=1))
        amplitude /= amplitude.max(axis=1, keepdims=True)
        freq = np.fft.rfftfreq(self._x.shape[1], 1.0 / self._rate)
        keep = freq > 0  # the DC bin only carries the offset
        out = {"freq": _rounded(freq[keep], 3), "mean": [], "low": [], "high": []}
        for k in range(len(self.classes)):
            a = amplitude[label == k][:, keep]
            out["mean"].append(_rounded(a.mean(axis=0), 3))
            out["low"].append(_rounded(np.percentile(a, BAND[0], axis=0), 3))
            out["high"].append(_rounded(np.percentile(a, BAND[1], axis=0), 3))
        return out

    @traitlets.observe("want")
    def _send_records(self, change):
        """Full-resolution records for the grid, as base64 int16 (records are within
        [-1, 1]): 6 kB each, so a page and the next one arrive almost at once."""
        ids = [int(i) for i in change["new"] if 0 <= int(i) < len(self._x)]
        if not ids:
            return
        scaled = np.round(np.clip(self._x[ids], -1, 1) * 32767).astype(np.int16)
        self.chunk = {
            "indices": ids,
            "data": base64.b64encode(scaled.tobytes()).decode(),
        }

    @traitlets.observe("selected")
    def _open_record(self, change):
        k = change["new"]
        if not 0 <= k < len(self._x):
            self.detail = {}
            return
        x = self._x[k]
        freq, spectrum = amplitude_spectrum(x, self._rate)
        # the spectrogram of the training target of 2c, of the live part of the record
        # (the flat padding after it would only fill the picture with the floor)
        live = x[: int(self._live[k])]
        spec = spectrogram(live[None])[0].numpy()
        row_freq, row_time = spectrogram_axes(len(live), self._rate)
        dt, df = row_time[1] - row_time[0], row_freq[1] - row_freq[0]
        # the colour scale covers the SPEC_RANGE dB below the peak
        scaled = np.clip((spec - spec.max()) / SPEC_RANGE + 1.0, 0.0, 1.0)
        viridis = matplotlib.colormaps["viridis"]
        image = png_data_url(viridis(scaled))
        self.detail = {
            "index": int(k),
            "x": _rounded(x, 4),
            "freq": _rounded(freq[1:], 3),
            "spectrum": _rounded(spectrum[1:], 3),
            "spectrogram": {
                "png": image,
                "bar": png_data_url(viridis(np.linspace(0, 1, 64))[None, :, :]),
                "range": SPEC_RANGE,
                "window": SPEC_WINDOW / self._rate,
                # the edges of the first and last cell, not their centres
                "t0": float(row_time[0] - dt / 2),
                "t1": float(row_time[-1] + dt / 2),
                "f0": float(row_freq[0] - df / 2),
                "f1": float(row_freq[-1] + df / 2),
            },
        }
