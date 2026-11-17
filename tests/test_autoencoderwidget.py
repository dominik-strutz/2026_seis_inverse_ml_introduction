"""The Part 2c autoencoder flow: what it sends, and what it refuses."""

import numpy as np
import pytest

from ml_seis.autoencoderwidget import AutoencoderFlow, choose_records

CLASSES = ("LP", "TR", "VT", "TC")
N, T, F, B = 40, 750, 16, 30


def make(**kwargs):
    rng = np.random.default_rng(0)
    label = np.arange(N) % 4
    return AutoencoderFlow(
        CLASSES,
        rng.normal(size=(N, T)),
        rng.normal(size=(N, 4)),
        label,
        rng.normal(size=(N, F, B)),
        rng.normal(size=(N, F, B)),
        **kwargs,
    )


def test_the_first_records_of_every_class_are_chosen():
    assert choose_records([0, 1, 0, 1, 0], 2) == [[0, 2], [1, 3]]


def test_records_and_all_latent_vectors_are_sent_once():
    widget = make(index=np.arange(N) + 500, per_class=3)
    assert widget.classes == list(CLASSES) and len(widget.colours) == 4
    assert len(widget.cmap) == 64 and widget.cmap[0].startswith("#")
    assert [len(rows) for rows in widget.records] == [3, 3, 3, 3]
    first = widget.records[2][0]  # the first VT record is number 2 of the data
    assert first["index"] == 502 and len(first["latent"]) == 4
    assert len(first["target"]) == F and len(first["target"][0]) == B
    assert len(first["rebuilt"]) == F and len(first["wave"]) == 250
    assert max(abs(v) for v in first["wave"]) == pytest.approx(1.0, abs=1e-3)
    assert len(widget.latent["values"]) == N and len(widget.latent["label"]) == N
    assert widget.axes == {"duration": 15.0, "fmin": 0.4, "fmax": 12.5}


def test_every_array_needs_one_row_per_record():
    with pytest.raises(ValueError, match="one row per record"):
        AutoencoderFlow(
            CLASSES,
            np.zeros((N, T)),
            np.zeros((N - 1, 4)),
            np.zeros(N, dtype=int),
            np.zeros((N, F, B)),
            np.zeros((N, F, B)),
        )
