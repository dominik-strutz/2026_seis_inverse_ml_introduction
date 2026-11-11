"""The Llaima window: records start at the onset and the padding is a flat tail."""

from pathlib import Path

import pytest
import torch

from ml_seis import waveform_data as wd

CACHE = Path(__file__).resolve().parents[1] / "data" / "waveforms"


def test_live_length_finds_the_flat_tail():
    u = torch.randn(3, 1, 100)
    u[1, 0, 40:] = u[1, 0, 39]  # repeats the last live sample
    u[2, 0, 89:] = 0.5
    assert wd.live_length(u).tolist() == [100, 40, 90]


def test_window_and_selection_hold_no_padding():
    u = torch.randn(4, 1, 100)
    u[1, 0, 30:] = u[1, 0, 29]
    data = wd.LlaimaData(
        u,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        wd.LLAIMA_CLASSES,
        50.0,
        {},
        wd.live_length(u),
    )
    assert data.window(50).shape == (4, 1, 50)
    assert data.indices("train", min_live=50).tolist() == [0, 2, 3]


@pytest.mark.skipif(not wd.llaima_cache_path(CACHE).exists(), reason="no local cache")
def test_real_records_have_no_flat_tail_in_the_window():
    data = wd.load_llaima(CACHE)
    window = data.window()
    for split in wd.SPLITS:
        idx = data.indices(split, min_live=wd.LLAIMA_WINDOW)
        assert len(idx) > 0.97 * len(data.indices(split))
        tail = window[idx, 0, -50:]
        assert (tail[:, 1:] - tail[:, :-1]).abs().max(dim=1).values.min() > 1e-6
