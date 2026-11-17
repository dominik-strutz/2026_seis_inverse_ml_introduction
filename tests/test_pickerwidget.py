"""The Part 2d explorer: its tables, and the samples it sends when asked."""

import base64
from types import SimpleNamespace

import numpy as np
import torch

from ml_seis.pickerwidget import PickerExplorer

N, LENGTH = 6, 200


def fake_data():
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        u=torch.randn(N, 3, LENGTH),
        p=torch.full((N,), 50),
        s=torch.full((N,), 120),
        split=torch.tensor([0, 0, 1, 1, 2, 2]),
        event_id=np.array(["a", "a", "a", "b", "b", "c"]),
        station=np.array(["X", "Y", "X", "Z", "Y", "Z"]),
        distance_km=rng.uniform(5, 100, N),
        snr=rng.uniform(1, 30, N),
        magnitude=rng.uniform(1, 4, N),
        event_lonlat=rng.uniform(-120, -115, (N, 2)),
        station_lonlat=rng.uniform(33, 36, (N, 2)),
    )


def decode(chunk, n, channels, length=LENGTH):
    return np.frombuffer(base64.b64decode(chunk), dtype=np.int8).reshape(
        n, channels, length
    )


def test_data_mode_sends_everything_it_shows():
    widget = PickerExplorer(fake_data())
    assert widget.mode == "data" and widget.rate == 25.0
    # only the earthquake with three stations has a record section: 3 of 6 traces
    assert len(widget.meta["p"]) == 3 and len(widget.events["lon"]) == 3
    assert widget.events["n"] == [3, 2, 1]
    assert widget.length == LENGTH // 4
    samples = decode(widget.store["samples"], 3, 3, widget.length)
    assert abs(samples).max() == 127 and len(widget.store["scale"]) == 3
    assert widget.meta["p"] == [round(50 / 4)] * 3  # picks follow the thinned samples


def test_results_mode_sends_curves_and_peaks():
    data = fake_data()
    prob = np.random.default_rng(1).dirichlet([1, 1, 1], size=(4, LENGTH))
    prob = prob.transpose(0, 2, 1)  # (traces, classes, samples)
    results = SimpleNamespace(
        index=np.array([0, 2, 4, 5]),
        u=data.u[[0, 2, 4, 5]],
        p=np.array([50, 60, 70, 80]),
        s=np.array([120, 130, 140, 150]),
        prob={"skips": prob, "no skips": prob},
    )
    widget = PickerExplorer(data, results)
    assert widget.mode == "results" and len(widget.meta["p"]) == 4
    assert [m["name"] for m in widget.models] == ["skips", "no skips"]
    assert len(widget.models[0]["at"]) == 2 and len(widget.models[0]["at"][0]) == 4
    widget.want = [1]
    assert decode(widget.chunk["samples"], 1, 3).shape == (1, 3, LENGTH)
    curves = np.frombuffer(
        base64.b64decode(widget.chunk["prob"]["skips"]), dtype=np.uint8
    )
    assert curves.shape == (2 * LENGTH,)
