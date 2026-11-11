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
    assert isinstance(plots.plot_sigmoid_softmax(), Figure)
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
    curves = {"train": [1.0, 0.5], "validation": [1.1, 0.7]}
    assert isinstance(plots.plot_history_comparison({"a": curves, "b": curves}), Figure)
    recalls = {"original": np.full(4, 0.9), "noise": np.full(4, 0.8)}
    assert isinstance(plots.plot_recall_comparison(list("abcd"), recalls), Figure)


def test_classification_figures():
    cm = np.array([[8, 1], [2, 9]])
    report = metrics.class_report(cm)
    assert isinstance(plots.plot_confusion(cm, ("a", "b"), report), Figure)
    marked = plots.plot_confusion(cm, ("a", "b"), report, highlight=1)
    assert isinstance(marked, Figure)
    assert isinstance(
        plots.plot_augmentations(np.random.randn(300), 50.0, 10, 4.0), Figure
    )


def test_activation_figure_scales_with_channels():
    maps = [torch.randn(4, 100), torch.randn(8, 100)]
    fig = plots.plot_activations(np.random.randn(100), maps, ["a", "b"], 50.0)
    heights = [ax.get_position().height for ax in fig.axes[1:]]
    assert heights[1] > 1.5 * heights[0]  # twice the channels, twice the height


def test_live_length_figure_and_baseline():
    n_live = np.array([100, 120, 300, 320, 310, 110])
    label = np.array([0, 0, 1, 1, 1, 0])
    fig = plots.plot_live_length(n_live, label, ("a", "b"), 50.0, 150)
    assert isinstance(fig, Figure)
    predicted = metrics.length_baseline(n_live, label, [105, 305], [200], 2)
    assert predicted.tolist() == [0, 1]


def test_handoff_figure():
    scene = make_scene(station_list(5), "homogeneous", 0, 0.3)
    assert isinstance(plots.plot_handoff(scene, [55.0, 45.0], [0.1, 0.5]), Figure)


def test_fill_tail_replaces_only_the_padding():
    u = torch.randn(3, 1, 400)
    u[1, 0, 120:] = 0.7  # padded record, 120 live samples
    n_live = torch.tensor([400, 120, 400])
    for kind in ("white", "band"):
        filled = metrics.fill_tail(u, n_live, kind)
        assert torch.equal(filled[0], u[0])  # nothing to replace
        assert torch.equal(filled[1, 0, :120], u[1, 0, :120])
        assert filled[1, 0, 120:].std() > 0.3
        assert not torch.any(filled[1, 0, 120:] == 0.7)


def test_activation_records_shape_and_size():
    import base64

    import torch
    from torch import nn

    from ml_seis import waveform_metrics as metrics

    block = nn.Sequential(nn.Conv1d(1, 4, 3, padding=1), nn.ReLU())
    features = nn.Sequential(
        block, nn.MaxPool1d(4), nn.Sequential(nn.Conv1d(4, 6, 3, padding=1), nn.ReLU())
    )
    u = torch.randn(3, 1, 40)
    out = metrics.activation_records(
        features, u, torch.tensor([0, 1, 1]), torch.full((3, 2), 0.5), ("A", "B")
    )
    assert out["rows"] == [4, 6] and out["cols"] == [40, 10] and out["samples"] == 40
    assert len(out["records"]) == 3
    assert len(base64.b64decode(out["records"][0]["maps"][1])) == 6 * 10
    assert len(base64.b64decode(out["records"][0]["x"])) == 40
    assert [f["shape"] for f in out["filters"]] == [[4, 1, 3], [6, 4, 3]]
    assert len(base64.b64decode(out["filters"][1]["w"])) == 6 * 4 * 3
