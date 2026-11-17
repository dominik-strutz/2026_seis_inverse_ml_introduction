"""The Part 2c autoencoder flow (anywidget): input, encoder, latent vector, decoder,
rebuilt spectrogram and target, for a record you choose.

One picture of the whole network with real numbers from the trained autoencoder: the
waveform of a test record, its latent vector (four numbers, drawn among the latent
vectors of all test records, one dot each, coloured by class), the spectrogram the
decoder rebuilds from those four numbers and the target it is compared with. The latent
vector is drawn as what we keep, the decoder, the rebuild and the target as scaffolding
that is only needed to train the encoder. A class and a record number choose the input;
nothing travels back to Python.
"""

from __future__ import annotations

import anywidget
import numpy as np
import traitlets

from .clusterwidget import _class_colours
from .waveform_plotting import _numpy
from .widgetkit import esm

SHOW = 250  # points of the waveform that are drawn (the record is thinned to this)


def _rounded(a, digits: int) -> list:
    return np.round(np.asarray(_numpy(a), dtype=np.float64), digits).tolist()


def choose_records(label, per_class: int) -> list[list[int]]:
    """The first ``per_class`` records (positions) of every class, a list per class."""
    label = np.asarray(_numpy(label))
    return [
        np.flatnonzero(label == c)[:per_class].tolist() for c in range(label.max() + 1)
    ]


class AutoencoderFlow(anywidget.AnyWidget):
    """The autoencoder from waveform to target, for one record at a time."""

    _esm = esm("autoencoderwidget.js")

    classes = traitlets.List().tag(sync=True)
    colours = traitlets.List().tag(sync=True)  # one per class
    cmap = traitlets.List().tag(sync=True)  # viridis, 64 hex colours
    records = traitlets.List().tag(
        sync=True
    )  # per class, the records that can be shown
    latent = traitlets.Dict().tag(sync=True)  # every record: latent vectors and labels
    axes = traitlets.Dict().tag(sync=True)  # duration, frequencies of the spectrogram

    def __init__(
        self,
        classes,
        waves,
        latent,
        label,
        target,
        rebuilt,
        index=None,
        duration: float = 15.0,
        fmin: float = 0.4,
        fmax: float = 12.5,
        per_class: int = 12,
        **kwargs,
    ):
        """``waves`` (records, samples), ``latent`` (records, d), ``label`` (records,)
        class indices, ``target`` and ``rebuilt`` (records, frequencies, time bins) in
        decibels, for the same records (the test records, never trained on).
        ``index`` (optional) is the number of every record in the data set, which the
        widget shows; ``duration`` is the length of a record (s), ``fmin`` and ``fmax``
        the frequencies of the first and the last row of the spectrograms (Hz)."""
        waves, latent = _numpy(waves), _numpy(latent)
        label = np.asarray(_numpy(label)).astype(int)
        target, rebuilt = _numpy(target), _numpy(rebuilt)
        if not len(waves) == len(latent) == len(label) == len(target) == len(rebuilt):
            raise ValueError("every array needs one row per record")
        if index is None:
            index = np.arange(len(label))
        index = np.asarray(_numpy(index)).astype(int)
        super().__init__(**kwargs)
        import matplotlib

        self.classes = list(classes)
        self.colours = _class_colours(classes)
        viridis = matplotlib.colormaps["viridis"]
        self.cmap = [
            matplotlib.colors.to_hex(viridis(x)) for x in np.linspace(0, 1, 64)
        ]
        step = max(waves.shape[-1] // SHOW, 1)
        shown = []
        for positions in choose_records(label, per_class):
            rows = []
            for p in positions:
                wave = waves[p, ::step]
                rows.append(
                    {
                        "index": int(index[p]),
                        "wave": _rounded(wave / max(np.abs(wave).max(), 1e-12), 3),
                        "latent": _rounded(latent[p], 3),
                        "target": _rounded(target[p], 1),
                        "rebuilt": _rounded(rebuilt[p], 1),
                    }
                )
            shown.append(rows)
        self.records = shown
        self.latent = {"values": _rounded(latent, 2), "label": label.tolist()}
        self.axes = {
            "duration": float(duration),
            "fmin": float(fmin),
            "fmax": float(fmax),
        }
