"""The Part 2 figures draw from arrays only; none may fail, and the padding helper
must keep the signal and replace only the flat samples."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure

from ml_seis import waveform_metrics as metrics
from ml_seis import waveform_plotting as plots

SPECS = torch.randn(6, 65, 64)


def test_diagrams_and_filter_bank():
    t = np.arange(300) / 50.0
    assert isinstance(plots.plot_arc(), Figure)
    assert isinstance(plots.plot_sigmoid_softmax(), Figure)
    assert isinstance(
        plots.plot_filter_bank(t, np.sin(t), np.random.randn(3, 290), list("abc")),
        Figure,
    )


def test_spectrogram_figures():
    classes = ("LP", "TR", "VT", "TC")
    assert isinstance(
        plots.plot_spectrogram_row(SPECS, classes, range(4), 47, 50.0), Figure
    )
    assert isinstance(
        plots.plot_spectrogram_row(SPECS, classes, range(4), 25, 50.0, 128, 12.5),
        Figure,
    )
    waves = np.random.randn(4, 750)
    assert isinstance(
        plots.plot_spectrogram_row(
            SPECS, classes, range(4), 25, 50.0, 128, 12.5, waves=waves
        ),
        Figure,
    )
    rebuilt = plots.plot_reconstruction(
        np.random.randn(4, 750), SPECS[:4], SPECS[2:6], classes, torch.randn(4, 4),
        25, 50.0, 128, 12.5,
    )  # fmt: skip
    assert isinstance(rebuilt, Figure)
    members = plots.plot_cluster_members(
        np.random.randn(5, 750), SPECS[:5], [0, 1, 2, 3, 0], classes, 25, 50.0, 128
    )
    assert isinstance(members, Figure)


def test_spectrogram_panel_puts_frames_at_their_centre():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plots._spectrogram_panel(ax, np.zeros((32, 30)), 25, 50.0, window=128, fmax=12.5)
    left, right, bottom, top = ax.images[0].get_extent()
    assert np.isclose(left, (128 - 25) / 2 / 50)  # a time bin is drawn at its centre
    assert np.isclose(right - left, 30 * 25 / 50)
    assert np.isclose(bottom, 12.5 / 32 / 2) and np.isclose(top, 12.5 + 12.5 / 32 / 2)
    assert ax.get_ylim() == (0, 12.5)
    plt.close(fig)


def test_corner_figure():
    latent = torch.randn(40, 4)
    group = torch.arange(40) % 4
    classes = ("LP", "TR", "VT", "TC")
    assert isinstance(plots.plot_corner(latent), Figure)
    assert isinstance(
        plots.plot_corner(latent, group, classes, centres=latent[:3], title="k = 3"),
        Figure,
    )


def test_bar_figures():
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


def test_split_by_event_figure_and_leak():
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    event = np.repeat([f"e{i}" for i in range(60)], 8)
    split = np.repeat(rng.choice(3, size=60, p=[0.7, 0.15, 0.15]), 8)
    data = SimpleNamespace(event_id=event, split=torch.as_tensor(split))
    assert isinstance(plots.plot_split_by_event(data, n_events=10), Figure)
    # dealing traces at random leaks; dealing whole events does not
    draw = metrics.random_trace_split(len(event))
    assert metrics.trace_split_leak(event, draw) > 0.5
    assert metrics.trace_split_leak(event, split) == 0.0


def test_pick_figures():
    p, s = torch.tensor([300, 310, 320]), torch.tensor([900, 950, 1000])
    picks = np.array([[301, 905], [-1, 951], [500, -1]])
    assert isinstance(plots.plot_pick_scatter(picks, p, s), Figure)
    residual = np.array([0.01, -0.02, np.nan, 3.0])
    both = {"skips": (residual, residual), "no skips": (residual, residual)}
    assert isinstance(plots.plot_pick_residuals(both), Figure)


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
