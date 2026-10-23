"""The Part 1b figures draw from a scene and from run curves; none may fail."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure

from ml_seis import learning_plotting as lp
from ml_seis.locatorwidget import DEFAULTS, station_list
from ml_seis.scene import make_scene

SCENE = make_scene(station_list(5), "homogeneous", 0, DEFAULTS["sigma"])


def test_dataset_and_split_figures():
    assert isinstance(lp.dataset_figure(SCENE), Figure)
    assert isinstance(lp.split_figure(SCENE, (30, 10, 10)), Figure)


def test_scaling_figure():
    d = torch.randn(40, 5, dtype=torch.float64) * 3
    m = torch.rand(40, 2, dtype=torch.float64) * 100
    fig = lp.scaling_figure(d, d / d.std(), m, (m - m.mean(0)) / m.std(0))
    assert isinstance(fig, Figure)


def test_posterior_figure():
    assert isinstance(lp.posterior_figure(SCENE, "ridgecrest"), Figure)


def test_learning_curves_figure():
    curve = np.linspace(20, 2, 30)
    runs = [("run", curve, curve + 1, 29)] * 3
    assert isinstance(lp.learning_curves_figure(runs, 1.5), Figure)
