"""The Part 2d explorer (anywidget): browse the picking data, then the picker's results.

One widget, two modes, with the same look.

``mode="data"`` shows the whole CEED subset before anything is trained: a map of
the earthquakes and stations, the record section of the selected earthquake (the
vertical component of each station that recorded it, ordered by distance and aligned
on P), and one trace with its three components and the analyst P and S picks.

``mode="results"`` shows the test traces after training: the same trace view with the
picker's P and S probability curves under it, the model's picks against the analyst's
and the error of each, a switch between networks, and a threshold that turns the curves
into picks. The list on the left can be sorted and filtered, for example by the largest
errors.

The tables (events, stations, one row per trace) are sent once. In data mode the
traces of every earthquake with at least three stations (478 of the 2390 traces) are
sent with the widget too, thinned to 25 Hz and stored in 8 bits, about 3 MB, so that it
works without asking Python: Python is often busy training when the slide opens. In
results mode the samples of the traces on screen are asked for (``want``) and answered
in ``chunk``, as in the Llaima explorer.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import anywidget
import numpy as np
import traitlets
from scipy import signal

from .waveform_plotting import (
    COMPONENT_COLOURS,
    PHASE_COLOURS,
    SPLIT_COLOURS,
    _c,
)
from .widgetkit import esm

SPLIT_NAMES = ("train", "validation", "test")
MIN_STATIONS = 3  # earthquakes with fewer stations have no record section
THIN = 4  # data mode: 100 Hz -> 25 Hz
MU = 255.0  # mu-law companding of the 8-bit samples: constant relative precision
BASEMAP = Path(__file__).parent / "artifacts" / "regional_map.json"


def _compress(x):
    """(n, 3, T) samples as int8 with one scale per trace: mu-law, so that the noise
    before the first arrival keeps its shape next to the largest wave."""
    scale = np.maximum(np.abs(x).max(axis=(1, 2)), 1e-9)
    y = np.log1p(MU * np.abs(x) / scale[:, None, None]) / np.log1p(MU)
    q = np.round(127 * np.sign(x) * y).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), _rounded(scale, 4)


def _rounded(a, digits: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


def _colours() -> dict:
    import matplotlib.pyplot as plt

    from .plotting import style

    with plt.style.context(style("open")):
        return {
            "split": [_c(c) for c in SPLIT_COLOURS],
            "component": [_c(c) for c in COMPONENT_COLOURS],
            "P": _c(PHASE_COLOURS["P"]),
            "S": _c(PHASE_COLOURS["S"]),
        }


class PickerExplorer(anywidget.AnyWidget):
    """Browse picking records (``mode="data"``) or a picker's results
    (``mode="results"``). ``selected`` is the open trace (-1: none)."""

    _esm = esm("pickerwidget.js")

    mode = traitlets.Unicode("data").tag(sync=True)
    colours = traitlets.Dict().tag(sync=True)
    rate = traitlets.Float(100.0).tag(sync=True)
    length = traitlets.Int(0).tag(sync=True)  # samples per trace
    meta = traitlets.Dict().tag(sync=True)  # one entry per trace
    events = traitlets.Dict().tag(sync=True)  # one entry per earthquake
    stations = traitlets.List().tag(sync=True)  # [name, lon, lat]
    models = traitlets.List().tag(sync=True)  # results: name and peaks per network
    store = traitlets.Dict().tag(sync=True)  # data mode: the samples of every trace
    basemap = traitlets.Dict().tag(sync=True)  # coastline and faults for the map
    threshold = traitlets.Float(0.2).tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)
    want = traitlets.List().tag(sync=True)
    chunk = traitlets.Dict().tag(sync=True)

    def __init__(self, data, results=None, **kwargs):
        """``data`` is a ``PickerData``. ``results`` (optional) has ``index`` (the
        test traces in ``data``), ``u`` (their crops, (n, 3, T)), ``p`` and ``s``
        (analyst samples in the crop) and ``prob`` (a dict from a network's name to
        its probabilities, (n, 3, T))."""
        super().__init__(**kwargs)
        self.colours = _colours()
        if results is None:
            self.mode = "data"
            self.rate = 100.0 / THIN
            index = self._keep(data)
            u = signal.decimate(
                data.u.numpy()[index].astype(np.float64),
                THIN,
                ftype="fir",
                axis=-1,
                zero_phase=True,
            ).astype(np.float32)
            p = np.round(data.p.numpy()[index] / THIN).astype(int)
            s = np.round(data.s.numpy()[index] / THIN).astype(int)
            self._u, self._prob = u, {}
            samples, scale = _compress(u)
            self.store = {"samples": samples, "scale": scale}
            if BASEMAP.exists():
                self.basemap = json.loads(BASEMAP.read_text(encoding="utf-8"))
        else:
            self.mode = "results"
            self.rate = 100.0
            self._u = results.u.numpy()
            index = np.asarray(results.index)
            p, s = np.asarray(results.p), np.asarray(results.s)
            self._prob = {k: np.asarray(v) for k, v in results.prob.items()}
        self.length = int(self._u.shape[-1])
        self._tables(data, index, p, s)
        self.models = [self._peaks(name, prob) for name, prob in self._prob.items()]

    @staticmethod
    def _keep(data) -> np.ndarray:
        """The traces of the earthquakes that enough stations recorded."""
        _, event_of, count = np.unique(
            data.event_id, return_inverse=True, return_counts=True
        )
        return np.flatnonzero(count[event_of] >= MIN_STATIONS)

    def _tables(self, data, index, p, s):
        """Stations and earthquakes of the whole data set (for the map); one row per
        trace of ``index`` (for the traces that are shown)."""
        names, first, station_of = np.unique(
            data.station, return_index=True, return_inverse=True
        )
        self.stations = [
            [str(n), float(x), float(y)]
            for n, (x, y) in zip(names, data.station_lonlat[first], strict=True)
        ]
        ids, first, event_of, count = np.unique(
            data.event_id, return_index=True, return_inverse=True, return_counts=True
        )
        split = data.split.numpy()
        self.events = {
            "lon": _rounded(data.event_lonlat[first, 0], 3),
            "lat": _rounded(data.event_lonlat[first, 1], 3),
            "mag": _rounded(data.magnitude[first], 2),
            "split": split[first].tolist(),
            "id": [str(e) for e in ids],
            "n": count.tolist(),
        }
        self.meta = {
            "event": event_of[index].tolist(),
            "station": station_of[index].tolist(),
            "distance": _rounded(data.distance_km[index], 1),
            "snr": _rounded(data.snr[index], 2),
            "mag": _rounded(data.magnitude[index], 2),
            "split": split[index].tolist(),
            "p": [int(v) for v in p],
            "s": [int(v) for v in s],
        }

    @staticmethod
    def _peaks(name, prob) -> dict:
        """Highest sample and height of each curve, so the threshold is applied in the
        browser without asking Python."""
        curves = prob[:, :2]
        return {
            "name": name,
            "at": curves.argmax(axis=2).T.tolist(),  # [P list, S list]
            "peak": _rounded(curves.max(axis=2).T, 3),
        }

    @traitlets.observe("want")
    def _send_records(self, change):
        """int8 samples (mu-law, scaled per trace) and, in results mode, uint8
        probability curves of P and S for each network."""
        ids = [int(i) for i in change["new"] if 0 <= int(i) < len(self._u)]
        if not ids:
            return
        samples, scale = _compress(self._u[ids].astype(np.float32))
        chunk = {"indices": ids, "scale": scale, "samples": samples}
        if self._prob:
            chunk["prob"] = {
                name: base64.b64encode(
                    np.round(prob[ids][:, :2] * 255).astype(np.uint8).tobytes()
                ).decode()
                for name, prob in self._prob.items()
            }
        self.chunk = chunk
