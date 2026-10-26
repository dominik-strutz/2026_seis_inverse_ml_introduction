"""Where we expect earthquakes: prior densities on the map, and samples from them.

All priors live on the same square map (km). ``uniform`` covers the whole map; the
catalogue-based ones are restricted to the box ``[MARGIN, BOX - MARGIN]``.

``uniform``    every position on the map is equally likely.
``ridgecrest`` the 2019 Ridgecrest aftershock catalogue (USGS ComCat, M >= 1.5,
               4 Jul - 4 Aug 2019): events counted on a 2 km grid, one Gaussian kernel
               per cell weighted by its share of the events, truncated to the box and
               mixed with a weak uniform background. The cell weights and the two
               constants are stored in ``ridgecrest_prior.npz``; rebuild it with
               ``scripts/fit_ridgecrest_prior.py``.
``ridgecrest_faults`` the same kernel mixture without the uniform background: only
               the fault seismicity, so events far from the faults are (all but)
               impossible.

The ``ridgecrest`` density is smooth, so ``log_prior`` can be differentiated with
respect to the position. Sampling is exact: pick a cell from the categorical
distribution of weights, scatter the position with the kernel's Gaussian, and keep it
if it is inside the box.

In Lecture 1 this is "the distribution we draw training events from". The second half
of Lecture 1 treats it as a prior density in Bayes' theorem.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import torch

from .geometry import BOX

MARGIN = 5.0  # events are kept this far from the edge of the map (km)
PRIORS = ("uniform", "ridgecrest", "ridgecrest_faults")


def _check(name: str) -> None:
    if name not in PRIORS:
        raise ValueError(f"unknown prior {name!r}; choose from {PRIORS}")


def _background(name: str, fitted: float) -> float:
    """Weight of the uniform background: the fitted one, or none for faults only."""
    return 0.0 if name == "ridgecrest_faults" else fitted


def _margin(name: str) -> float:
    """The uniform prior covers the whole map; the catalogue-based ones keep events
    ``MARGIN`` away from the edge, where the kernel weights were fitted."""
    return 0.0 if name == "uniform" else MARGIN


def _inside(m_x, m_y, margin: float = MARGIN):
    lo, hi = margin, BOX - margin
    return (m_x >= lo) & (m_x <= hi) & (m_y >= lo) & (m_y <= hi)


@cache
def _kernels() -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Kernel centres (C, 2), log weights (C,), bandwidth, background, log Z.

    ``Z`` is the mass of the untruncated kernel mixture inside the box, which is
    analytic: each isotropic Gaussian factorises over x and y.
    """
    file = np.load(Path(__file__).with_name("ridgecrest_prior.npz"))
    weights = torch.from_numpy(file["weights"].astype(np.float64))
    h, background = float(file["bandwidth"]), float(file["background"])
    n = weights.shape[0]
    cell = (BOX - 2 * MARGIN) / n
    nodes = MARGIN + cell * (torch.arange(n, dtype=torch.float64) + 0.5)
    cy, cx = torch.meshgrid(nodes, nodes, indexing="ij")  # [iy, ix]
    keep = weights.flatten() > 1e-6  # drop empty cells: they carry no mass
    centres = torch.stack([cx.flatten(), cy.flatten()], dim=1)[keep]
    w = weights.flatten()[keep]
    w = w / w.sum()

    normal = torch.distributions.Normal(0.0, 1.0)
    inside_mass = (
        normal.cdf((BOX - MARGIN - centres) / h) - normal.cdf((MARGIN - centres) / h)
    ).prod(dim=1)  # (C,) mass of each kernel inside the box
    return centres, torch.log(w), h, background, math.log((w * inside_mass).sum())


def log_prior(name: str, m_x, m_y) -> torch.Tensor:
    """Log prior density (1/km²) at map positions; ``-inf`` outside the box.

    For ``ridgecrest`` the result is differentiable with respect to the positions.
    """
    _check(name)
    m_x, m_y = torch.broadcast_tensors(
        *(torch.as_tensor(a, dtype=torch.float64) for a in (m_x, m_y))
    )
    margin = _margin(name)
    log_uniform = -2 * math.log(BOX - 2 * margin)
    if name == "uniform":
        logp = torch.full_like(m_x, log_uniform)
    else:
        centres, log_w, h, background, log_z = _kernels()
        background = _background(name, background)
        pts = torch.stack([m_x, m_y], dim=-1).reshape(-1, 2)
        chunks = []
        for chunk in pts.split(4096):  # (points, kernels) stays small for big grids
            d2 = ((chunk[:, None, :] - centres) ** 2).sum(dim=-1)
            log_mix = torch.logsumexp(log_w - d2 / (2 * h**2), dim=1)
            chunks.append(log_mix - math.log(2 * math.pi * h**2) - log_z)
        log_mix = torch.cat(chunks).reshape(m_x.shape)
        logp = math.log(1 - background) + log_mix
        if background > 0:
            logp = torch.logaddexp(
                logp,
                torch.tensor(math.log(background) + log_uniform, dtype=torch.float64),
            )
    return torch.where(
        _inside(m_x, m_y, margin), logp, torch.full_like(logp, -math.inf)
    )


def sample_prior(name: str, n: int):
    """Draw ``n`` source positions from a prior: a tensor of shape (n, 2) in km."""
    _check(name)
    lo, hi = _margin(name), BOX - _margin(name)
    kwargs = {"dtype": torch.float64}
    if name == "uniform":
        return lo + (hi - lo) * torch.rand(n, 2, **kwargs)

    centres, log_w, h, background, _ = _kernels()
    is_background = torch.rand(n, **kwargs) < _background(name, background)
    n_mix = int((~is_background).sum())
    mix = torch.empty(0, 2, dtype=torch.float64)
    while len(mix) < n_mix:  # rejection truncates the kernel mixture to the box
        k = torch.multinomial(log_w.exp(), 2 * n_mix, replacement=True)
        pts = centres[k] + h * torch.randn(2 * n_mix, 2, **kwargs)
        mix = torch.cat([mix, pts[_inside(pts[:, 0], pts[:, 1])]])
    out = lo + (hi - lo) * torch.rand(n, 2, **kwargs)  # background positions
    out[~is_background] = mix[:n_mix]
    return out
