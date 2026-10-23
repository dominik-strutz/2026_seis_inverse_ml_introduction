"""The exact answer on a grid: the posterior of a source location (classical, not ML).

For centred picks d~ with Gaussian noise of standard deviation sigma on every pick,

    p(m | d~)  ∝  exp(-|C F(m) - d~|² / (2 sigma²)) · rho(m),

with C the station-mean projection and rho the prior. C is a projection, so the
centred residual's sum of squares is the exact Gaussian misfit of the centred data.
A muted (dead) station is left out: the remaining picks are centred among themselves.

Everything is evaluated on a regular grid over the map (``GRID`` nodes per side), so
it is exact up to the grid spacing. ``posterior_means`` does the same for many events
at once: the posterior mean is the best possible answer under a squared-error loss,
so its error is the floor that no locator trained with MSE can beat on average.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from .geometry import BOX
from .priors import log_prior

GRID = 101  # nodes per side: 1 km spacing on the default map


def grid_sources(n: int = GRID, box: float = BOX) -> torch.Tensor:
    """Grid nodes as source positions, shape (n * n, 2); x runs fastest, so a field
    on the grid reshapes to (n, n) indexed [y, x]."""
    nodes = torch.linspace(0, box, n, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(nodes, nodes, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)


@cache
def grid_log_prior(prior: str, n: int = GRID) -> torch.Tensor:
    """log rho on the grid nodes, shape (n * n,); computed once per prior."""
    m = grid_sources(n)
    return log_prior(prior, m[:, 0], m[:, 1])


def _centre(a: torch.Tensor) -> torch.Tensor:
    return a - a.mean(-1, keepdim=True)


def log_posterior(F, d_obs_c, sigma, prior=None, live=None, n: int = GRID, F_grid=None):
    """Unnormalised log posterior on the grid, shape (E, n * n), for E events.

    ``F(m_x, m_y)`` is the forward operator the analyst assumes, ``d_obs_c`` the
    centred picks (E, N) or (N,), ``prior`` a name from ``PRIORS`` or ``None`` for
    a flat prior over the whole map, ``live`` a boolean mask of working stations.
    ``F_grid`` may pass F already evaluated on the grid nodes.
    """
    d = torch.as_tensor(d_obs_c, dtype=torch.float64)
    d = d[None] if d.ndim == 1 else d
    if F_grid is None:
        m = grid_sources(n)
        F_grid = F(m[:, 0], m[:, 1])  # (G, N)
    if live is not None:
        live = torch.as_tensor(live, dtype=torch.bool)
        F_grid, d = F_grid[:, live], d[:, live]
    F_c, d_c = _centre(F_grid), _centre(d)
    # 1/2 |F_c - d_c|² expanded, so (E, G, N) is never formed
    S = 0.5 * ((F_c**2).sum(-1)[None, :] - 2 * d_c @ F_c.T + (d_c**2).sum(-1)[:, None])
    log_like = -S / sigma**2
    if prior is None:
        return log_like
    return log_like + grid_log_prior(prior, n)[None, :]


@dataclass(frozen=True)
class Posterior:
    """Posterior of one event on the grid."""

    density: torch.Tensor  # (n, n) indexed [y, x], 1/km²
    mean: torch.Tensor  # (2,) km
    level_68: float  # density above which 68 % of the mass lies
    level_95: float  # same for 95 %


def grid_posterior(F, d_obs_c, sigma, prior=None, live=None, n: int = GRID):
    """Normalised posterior of one event, its mean and the 68 / 95 % levels."""
    return posterior_from_log(log_posterior(F, d_obs_c, sigma, prior, live, n)[0], n)


def posterior_from_log(logp: torch.Tensor, n: int = GRID) -> Posterior:
    """A ``Posterior`` from an unnormalised log density on the grid nodes, shape
    (n * n,): the exact one, or a density learned by a network."""
    p = torch.softmax(logp.double(), dim=0)  # probability per grid cell
    m = grid_sources(n)
    mean = p @ m
    ranked = p.sort(descending=True).values
    mass = ranked.cumsum(0)

    def level(q):
        k = int(torch.searchsorted(mass, torch.tensor(q, dtype=mass.dtype)))
        return float(ranked[min(k, len(ranked) - 1)])

    cell = (BOX / (n - 1)) ** 2
    return Posterior(
        density=(p / cell).reshape(n, n),
        mean=mean,
        level_68=level(0.68) / cell,
        level_95=level(0.95) / cell,
    )


def posterior_means(F, d_obs_c, sigma, prior, live=None, n: int = GRID, chunk=64):
    """Posterior mean for each of E events, shape (E, 2)."""
    m = grid_sources(n)
    F_grid = F(m[:, 0], m[:, 1])
    out = []
    for d in torch.as_tensor(d_obs_c, dtype=torch.float64).split(chunk):
        p = torch.softmax(log_posterior(F, d, sigma, prior, live, n, F_grid), dim=1)
        out.append(p @ m)
    return torch.cat(out)


def posterior_rms(F, d_obs_c, m_true, sigma, prior, live=None, n: int = GRID):
    """RMS location error (km) of the posterior mean over events with centred picks
    ``d_obs_c`` (E, N) and true sources ``m_true`` (E, 2): the lowest average error
    that any locator trained with a squared-error loss can reach."""
    m_post = posterior_means(F, d_obs_c, sigma, prior, live, n)
    return ((m_post - torch.as_tensor(m_true)) ** 2).sum(-1).mean().sqrt().item()


def posterior_samples(post: Posterior, n_samples: int, generator=None):
    """Samples of a grid posterior, shape (n_samples, 2) km: a grid cell drawn with
    its probability, then a uniform position inside the cell."""
    n = post.density.shape[0]
    p = post.density.flatten() / post.density.sum()
    k = torch.multinomial(p, n_samples, replacement=True, generator=generator)
    jitter = torch.rand(n_samples, 2, generator=generator, dtype=torch.float64) - 0.5
    return (grid_sources(n)[k] + jitter * BOX / (n - 1)).clamp(0, BOX)


def _bilinear(field: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Fields on the grid (E, n, n) indexed [y, x], read at one point each (E, 2)."""
    n = field.shape[-1]
    f = (m / BOX * (n - 1)).clamp(0, n - 1 - 1e-9)
    i0, j0 = f[:, 0].floor().long(), f[:, 1].floor().long()
    tx, ty = f[:, 0] - i0, f[:, 1] - j0
    e = torch.arange(len(m))

    def at(j, i):
        return field[e, j, i]

    bottom = (1 - tx) * at(j0, i0) + tx * at(j0, i0 + 1)
    top = (1 - tx) * at(j0 + 1, i0) + tx * at(j0 + 1, i0 + 1)
    return (1 - ty) * bottom + ty * top


def posterior_scores(F, d_obs_c, m_true, sigma, prior, live=None, n: int = GRID):
    """How the exact posterior scores the true sources of E events.

    Returns ``nll`` (E,), the negative log posterior density at the true source
    (density in 1/km²), and ``level`` (E,), the smallest credible level whose
    highest-density region contains the true source. For a calibrated posterior the
    levels are uniform on [0, 1]; the mean ``nll`` is the floor for any learned
    density trained under the same prior.
    """
    m_true = torch.as_tensor(m_true, dtype=torch.float64)
    cell = (BOX / (n - 1)) ** 2
    m = grid_sources(n)
    F_grid = F(m[:, 0], m[:, 1])
    nll, level = [], []
    for d, mt in zip(
        torch.as_tensor(d_obs_c, dtype=torch.float64).split(64),
        m_true.split(64),
        strict=True,
    ):
        p = torch.softmax(log_posterior(F, d, sigma, prior, live, n, F_grid), dim=1)
        density = p / cell
        at_truth = _bilinear(density.reshape(-1, n, n), mt)
        nll.append(-at_truth.clamp_min(1e-300).log())
        level.append((p * (density > at_truth[:, None])).sum(1))
    return torch.cat(nll), torch.cat(level)
