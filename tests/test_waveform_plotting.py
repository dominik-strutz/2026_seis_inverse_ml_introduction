"""The Part 2 figures draw from arrays only; none may fail, and the padding helper
must keep the signal and replace only the flat samples."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure

from ml_seis import waveform_metrics as metrics
from ml_seis import waveform_plotting as plots
from ml_seis.locatorwidget import station_list
from ml_seis.scene import make_scene

SPECS = torch.randn(6, 65, 64)


def test_diagrams_and_filter_bank():
    t = np.arange(300) / 50.0
    assert isinstance(plots.plot_arc(), Figure)
    assert isinstance(plots.plot_autoencoder(4160, 32), Figure)
    assert isinstance(
        plots.plot_filter_bank(t, np.sin(t), np.random.randn(3, 290), list("abc")),
        Figure,
    )


def test_spectrogram_figures():
    classes = ("LP", "TR", "VT", "TC")
    assert isinstance(
        plots.plot_spectrogram_row(SPECS, classes, range(4), 47, 50.0), Figure
    )
    grid = plots.plot_reconstruction_grid(SPECS, SPECS, classes, range(4), 47, 50.0)
    assert isinstance(grid, Figure)
    titles = [str(k) for k in range(8)]
    blend = plots.plot_interpolation(SPECS[[0] * 8], titles, 47, 50.0)
    assert isinstance(blend, Figure)


def test_bar_figures():
    assert isinstance(plots.plot_bars(["a", "b"], [0.4, 0.9], "share"), Figure)
    recalls = {"original": np.full(4, 0.9), "noise": np.full(4, 0.8)}
    assert isinstance(plots.plot_recall_comparison(list("abcd"), recalls), Figure)


def test_handoff_figure():
    scene = make_scene(station_list(5), "homogeneous", 0, 0.3)
    assert isinstance(plots.plot_handoff(scene, [55.0, 45.0], [0.1, 0.5]), Figure)


def test_fill_padding_replaces_only_flat_samples():
    u = torch.randn(3, 1, 200)
    u[1, 0, 120:] = 0.7  # padded record
    filled = metrics.fill_padding(u, scale=0.0)
    assert torch.equal(filled[0], u[0])  # no flat samples, nothing changes
    assert torch.equal(filled[1, 0, :119], u[1, 0, :119])
    assert torch.all(filled[1, 0, 121:] == 0.0)
    noisy = metrics.fill_padding(u, scale=1.0)
    assert noisy[1, 0, 121:].std() > 0.3
