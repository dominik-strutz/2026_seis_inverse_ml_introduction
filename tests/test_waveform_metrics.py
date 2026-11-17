"""The spectrogram shared by the Llaima widget and the 2c training target."""

import numpy as np
import torch

from ml_seis.waveform_metrics import (
    SPEC_BINS,
    SPEC_HOP,
    SPEC_POOL,
    SPEC_WINDOW,
    spectrogram,
    spectrogram_axes,
)

FS = 50.0


def test_shape_and_axes_agree():
    x = torch.randn(3, 981)
    spec = spectrogram(x)
    freq, time = spectrogram_axes(981, FS)
    assert spec.shape == (3, SPEC_BINS // SPEC_POOL, len(time))
    assert len(freq) == spec.shape[1]
    assert len(time) == (981 - SPEC_WINDOW) // SPEC_HOP + 1
    assert time[0] == SPEC_WINDOW / 2 / FS  # the centre of the first window


def test_tone_sits_at_its_frequency():
    t = np.arange(1500) / FS
    spec = spectrogram(torch.tensor(np.sin(2 * np.pi * 5.0 * t))[None])[0]
    freq, _ = spectrogram_axes(1500, FS)
    pooled_width = SPEC_POOL * FS / SPEC_WINDOW
    assert abs(freq[spec.mean(dim=1).argmax()] - 5.0) < pooled_width


def test_the_mean_of_the_record_is_removed():
    t = np.arange(1500) / FS
    tone = torch.tensor(np.sin(2 * np.pi * 5.0 * t))[None]
    assert torch.allclose(spectrogram(tone + 50.0), spectrogram(tone), atol=1e-3)


def test_pooling_averages_power_before_the_logarithm():
    x = torch.randn(2, 500)
    full = spectrogram(x, pool=1)
    pooled = spectrogram(x)
    power = 10 ** (full / 10)
    expected = 10 * torch.log10((power[:, 0::2] + power[:, 1::2]) / 2)
    assert torch.allclose(pooled, expected, atol=1e-3)
