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
from .waveform_metrics import amplitude_spectrum, log_spectrogram, record_features
from .waveform_plotting import _colour
from .widgetkit import esm, png_data_url

SPECTROGRAM_WINDOW = 128  # samples (2.6 s at 50 Hz)
SPECTROGRAM_HOP = 16
BAND = (10, 90)  # percentiles of the per-record spectra drawn as a band


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
        spec, spec_freq, spec_time = log_spectrogram(
            x, self._rate, SPECTROGRAM_WINDOW, SPECTROGRAM_HOP
        )
        dt, df = spec_time[1] - spec_time[0], spec_freq[1] - spec_freq[0]
        # row 0 = lowest frequency; the mean-free frames leave nothing at 0 Hz, drop it
        scaled = np.clip((spec[1:] + 60.0) / 60.0, 0, 1)
        image = png_data_url(matplotlib.colormaps["viridis"](scaled))
        self.detail = {
            "index": int(k),
            "x": _rounded(x, 4),
            "freq": _rounded(freq[1:], 3),
            "spectrum": _rounded(spectrum[1:], 3),
            "spectrogram": {
                "png": image,
                # the edges of the first and last cell, not their centres
                "t0": float(spec_time[0] - dt / 2),
                "t1": float(spec_time[-1] + dt / 2),
                "f0": float(spec_freq[1] - df / 2),
                "f1": float(spec_freq[-1] + df / 2),
            },
        }
