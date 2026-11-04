"""Metrics and small analysis helpers for Part 2 (no models, no plotting)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts with rows = analyst label, columns = prediction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def class_report(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class recall, precision and the overall accuracy from a confusion matrix."""
    true_count, pred_count = cm.sum(axis=1), cm.sum(axis=0)
    diagonal = np.diag(cm).astype(float)
    return {
        "recall": diagonal / np.maximum(true_count, 1),
        "precision": diagonal / np.maximum(pred_count, 1),
        "accuracy": diagonal.sum() / cm.sum(),
        "majority_accuracy": true_count.max() / cm.sum(),
    }


def confident_mistakes(probability, label, k: int = 4) -> np.ndarray:
    """Indices of the ``k`` wrong predictions made with the highest confidence.

    ``probability`` is ``(N, n_classes)``; the binary task is passed as two columns.
    """
    probability, label = np.asarray(probability), np.asarray(label)
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    wrong = np.flatnonzero(predicted != label)
    return wrong[np.argsort(-confidence[wrong])][:k]


def pca_2d(z) -> np.ndarray:
    """Project rows of ``z`` onto their first two principal axes."""
    z = np.asarray(z, dtype=np.float64).reshape(len(z), -1)
    z = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return z @ vt[:2].T


def receptive_field(model) -> tuple[int, int]:
    """Receptive field (samples) and total stride of the conv/pool layers in order."""
    import torch.nn as nn

    field, jump = 1, 1
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d | nn.MaxPool1d | nn.AvgPool1d):
            kernel = (
                layer.kernel_size[0]
                if isinstance(layer.kernel_size, tuple)
                else layer.kernel_size
            )
            stride = (
                layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            )
            dilation = getattr(layer, "dilation", 1)
            dilation = dilation[0] if isinstance(dilation, tuple) else dilation
            field += (kernel - 1) * dilation * jump
            jump *= stride
    return field, jump


def layer_spec(features, n_samples: int) -> list[dict]:
    """Conv, pool and average layers of ``features`` for a record of ``n_samples``.

    One dict per layer: ``kind`` (conv, pool, avg), ``kernel``, ``stride``, ``filters``
    (feature maps after the layer), ``length`` (samples after the layer), ``field``
    (receptive field) and ``spacing`` (input samples between neighbouring units).
    The figures of Part 2 draw the network from this list."""
    import torch.nn as nn

    def first(value):
        return value[0] if isinstance(value, tuple) else value

    spec, length, field, spacing, filters = [], n_samples, 1, 1, 1
    for layer in features.modules():
        if isinstance(layer, nn.AdaptiveAvgPool1d):
            kernel, length = length, first(layer.output_size)
            field, spacing, stride = n_samples, n_samples, 1
            kind = "avg"
        elif isinstance(layer, nn.Conv1d | nn.MaxPool1d | nn.AvgPool1d):
            kernel, stride = first(layer.kernel_size), first(layer.stride)
            dilation = first(getattr(layer, "dilation", 1))
            padding = first(layer.padding)
            if isinstance(layer, nn.Conv1d):
                kind, filters = "conv", layer.out_channels
            else:
                kind = "pool"
            span = dilation * (kernel - 1) + 1
            length = (length + 2 * padding - span) // stride + 1
            field += (kernel - 1) * dilation * spacing
            spacing *= stride
        else:
            continue
        spec.append(
            dict(
                kind=kind,
                kernel=kernel,
                stride=stride,
                filters=filters,
                length=length,
                field=field,
                spacing=spacing,
            )
        )
    return spec


def shape_table(model, x) -> list[tuple[str, tuple[int, ...]]]:
    """Output shape after each top-level layer of ``model.features`` for input ``x``."""
    import torch

    rows = [("input", tuple(x.shape))]
    with torch.no_grad():
        for name, layer in model.features.named_children():
            x = layer(x)
            rows.append((f"{name}: {layer.__class__.__name__}", tuple(x.shape)))
    return rows


def layer_activations(features, x):
    """Outputs of every convolution block of ``features`` for one record ``x``."""
    import torch
    import torch.nn as nn

    activations, names = [], []
    with torch.no_grad():
        for layer in features:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                activations.append(x[0].cpu())
                names.append(f"block {len(names) + 1}")
    return activations, names


def extract_picks(probability, threshold: float = 0.3):
    """One P and one S pick per trace from the picker's probability curves.

    ``probability`` is ``(B, 3, L)`` with channels P, S, neither. A phase is picked at
    the highest sample of its curve if the peak reaches ``threshold``; otherwise the
    pick is -1 (missed). Returns ``(picks, peaks)``, both ``(B, 2)``.
    """
    curves = np.asarray(probability)[:, :2]
    peaks = curves.max(axis=2)
    picks = curves.argmax(axis=2)
    return np.where(peaks >= threshold, picks, -1), peaks


def pick_residuals(picks, p, s, sampling_rate: float):
    """Predicted minus analyst pick time (s); ``nan`` where nothing was picked."""
    truth = np.stack([np.asarray(p), np.asarray(s)], axis=1)
    residual = (np.asarray(picks) - truth) / sampling_rate
    residual[np.asarray(picks) < 0] = np.nan
    return residual[:, 0], residual[:, 1]


def pick_summary(residual, tolerances=(0.1, 0.25, 0.5)):
    """Summary of pick residuals in seconds (predicted minus analyst).

    Misses are ``nan``: they are counted in ``miss_rate`` and as failures in the
    ``within_*`` fractions, but they do not enter the bias and error statistics.
    """
    residual = np.asarray(residual, dtype=float)
    found = residual[np.isfinite(residual)]
    summary = {
        "n": len(residual),
        "miss_rate": 1.0 - len(found) / max(len(residual), 1),
        "bias": float(np.median(found)) if len(found) else np.nan,
        "median_abs": float(np.median(np.abs(found))) if len(found) else np.nan,
        "rms": float(np.sqrt(np.mean(found**2))) if len(found) else np.nan,
    }
    for tol in tolerances:
        summary[f"within_{tol}"] = float(
            (np.abs(found) <= tol).sum() / max(len(residual), 1)
        )
    return summary


def worst_picks(residual_p, residual_s, k: int = 3) -> np.ndarray:
    """Indices of the ``k`` traces with the largest absolute P or S error.

    Missed picks (``nan`` residuals) rank first.
    """
    worst = np.fmax(np.abs(residual_p), np.abs(residual_s))
    worst = np.where(np.isnan(worst), np.inf, worst)
    return np.argsort(-worst)[:k]


def record_features(x, sampling_rate: float) -> dict[str, np.ndarray]:
    """Per-record summaries of ``x`` with shape ``(N, samples)``, for sorting.

    ``flat`` is the share of consecutive samples that are identical (the padding
    that fills a record shorter than the window), ``rms`` the root-mean-square
    amplitude, ``f_dom`` the frequency of the largest spectral peak (Hz, DC
    excluded) and ``f_mean`` the amplitude-weighted mean frequency (Hz).
    """
    x = np.asarray(x, dtype=np.float64)
    amplitude = np.abs(np.fft.rfft(x, axis=1))[:, 1:]
    freq = np.fft.rfftfreq(x.shape[1], 1.0 / sampling_rate)[1:]
    return {
        "flat": (np.abs(np.diff(x, axis=1)) < 1e-6).mean(axis=1),
        "rms": np.sqrt((x**2).mean(axis=1)),
        "f_dom": freq[amplitude.argmax(axis=1)],
        "f_mean": (amplitude * freq).sum(axis=1) / amplitude.sum(axis=1),
    }


def amplitude_spectrum(x, sampling_rate: float):
    """Amplitude spectrum of one record, normalised to its maximum: ``(freq, a)``."""
    a = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    return np.fft.rfftfreq(len(x), 1.0 / sampling_rate), a / a.max()


def log_spectrogram(
    x, sampling_rate: float, window: int = 128, hop: int = 16, pad: int = 1
):
    """Log-power spectrogram of one record: ``(spec, freq, time)`` with ``spec`` of
    shape ``(freq, time)`` in dB relative to its maximum, floored at -60 dB.

    ``scipy.signal.spectrogram`` with ObsPy's choices (``obspy.imaging.spectrogram``):
    a Hann window of ``window`` samples (128), zero-padded to ``pad * window`` points
    (``pad=1``: none, so one value per cell; ObsPy pads 8x), and 10 log10 of the power.
    Each segment also has its mean removed, so an offset does not leak into low bins."""
    freq, time, power = signal.spectrogram(
        np.asarray(x, dtype=np.float64),
        fs=sampling_rate,
        window="hann",
        nperseg=window,
        noverlap=window - hop,
        nfft=pad * window,
        detrend="constant",
        mode="psd",
    )
    db = 10.0 * np.log10(power / (power.max() + 1e-24) + 1e-12)
    return np.maximum(db, -60.0), freq, time


def fill_padding(u, scale: float, seed: int = 0):
    """Replace the flat padding of records ``u`` (N, 1, T) by Gaussian noise.

    A sample is flat when it repeats the previous one. The noise has ``scale`` times
    the standard deviation of the rest of the record, so ``scale = 0`` keeps a flat
    line of zeros and ``scale = 1`` makes the padding as loud as the signal."""
    import torch

    x = u[:, 0]
    step = (x[:, 1:] - x[:, :-1]).abs() < 1e-6
    flat = torch.cat([step[:, :1], step], dim=1)
    live = (~flat).float()
    count = live.sum(dim=1).clamp(min=1)
    mean = (x * live).sum(dim=1) / count
    spread = (((x - mean[:, None]) ** 2 * live).sum(dim=1) / count).sqrt()
    torch.manual_seed(seed)
    noise = torch.randn(x.shape)
    return torch.where(flat, noise * (scale * spread)[:, None], x)[:, None]
