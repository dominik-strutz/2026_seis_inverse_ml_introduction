"""The spectrogram of an open record in the Llaima explorer."""

import base64
import io

import numpy as np
import pytest
import torch

from ml_seis.llaimawidget import SPEC_RANGE, LlaimaExplorer
from ml_seis.waveform_data import LlaimaData
from ml_seis.waveform_metrics import SPEC_BINS, SPEC_POOL, SPEC_WINDOW

FS = 50.0


def _explorer(n_live):
    rng = np.random.default_rng(0)
    n, length = 4, 3008
    x = rng.standard_normal((n, length)).astype(np.float32)
    x[:, n_live:] = x[:, n_live - 1 : n_live]  # the flat padding of the data set
    u = torch.from_numpy(x)[:, None, :] / 4
    data = LlaimaData(
        u=u,
        label=torch.tensor([0, 1, 2, 3]),
        split=torch.tensor([0, 0, 1, 2]),
        classes=("LP", "TR", "VT", "TC"),
        sampling_rate=FS,
        provenance={},
        n_live=torch.full((n,), n_live),
    )
    return LlaimaExplorer(data)


def _image(url):
    from PIL import Image

    png = base64.b64decode(url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_spectrogram_covers_the_live_part_only():
    widget = _explorer(n_live=1000)
    widget.selected = 0
    sp = widget.detail["spectrogram"]
    # the first window starts at 0 and the last ends at the end of the live signal
    assert sp["t0"] < SPEC_WINDOW / 2 / FS
    assert 1000 / FS - SPEC_WINDOW / FS < sp["t1"] <= 1000 / FS
    assert sp["window"] == pytest.approx(SPEC_WINDOW / FS)
    assert sp["range"] == SPEC_RANGE


def test_colours_run_over_the_range_below_the_peak():
    widget = _explorer(n_live=1000)
    widget.selected = 1
    pixels = _image(widget.detail["spectrogram"]["png"])
    bar = _image(widget.detail["spectrogram"]["bar"])
    viridis_top = bar[0, -1]  # the bar runs from low to high
    assert (pixels == viridis_top).all(axis=-1).any()  # the peak cell is the top colour
    assert pixels.shape[0] == SPEC_BINS // SPEC_POOL and pixels.shape[1] > 20


def test_a_short_record_still_gets_a_spectrogram():
    widget = _explorer(n_live=200)
    widget.selected = 2
    assert widget.detail["spectrogram"]["t1"] <= 200 / FS
