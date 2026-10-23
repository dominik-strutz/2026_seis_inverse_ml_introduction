"""The physics of the location problem, as small pure functions.

The lecture notebooks print these functions on a slide with ``show_source``, so the
source below is what students read. Rules that keep it readable:

* Every function takes everything it needs as an argument: no globals, no closures.
* Names are the maths symbols spelled out (see the notation table in AGENTS.md).
* Shapes and units are in the docstring. ``...`` stands for any batch of sources.
* Only ``torch`` and the other functions of this file are used.

``tests/test_physics.py`` checks these rules. Speed models with bent rays and the
exact posterior are machinery and live elsewhere (``velocity.py``, ``posterior.py``).
"""

import torch

__all__ = ["center", "observe", "simulate_picks", "travel_times"]


def travel_times(m_x, m_y, xi_x, xi_y, v_ref):
    """Forward operator F(m; xi): straight rays at constant P speed.

    m_x, m_y: source coordinates (km), tensors of shape (...)
    xi_x, xi_y: station coordinates (km), shape (N,)
    v_ref: P speed (km/s)
    returns: travel time (s) from every source to every station, shape (..., N)
    """
    delta_x = m_x[..., None] - xi_x
    delta_y = m_y[..., None] - xi_y
    Delta = torch.sqrt(delta_x**2 + delta_y**2)
    return Delta / v_ref


def center(d):
    """Station-mean projection C d = d - <d>: removes the origin time t0.

    d: (..., N) times (s), either picks or travel times
    returns: centred times, shape (..., N), with zero mean over the stations
    """
    return d - d.mean(-1, keepdim=True)


def simulate_picks(m_x, m_y, xi_x, xi_y, v_ref, sigma, t0, generator=None):
    """Noisy P arrival times d_obs = t0 + F(m) + eps with eps ~ N(0, sigma^2).

    m_x, m_y, xi_x, xi_y, v_ref: as in ``travel_times``
    sigma: standard deviation of a pick (s)
    t0: origin time (s), unknown to the inversion
    generator: optional torch.Generator, for reproducible noise
    returns: arrival times (s), shape (..., N)
    """
    F = travel_times(m_x, m_y, xi_x, xi_y, v_ref)
    eps = sigma * torch.randn(F.shape, dtype=F.dtype, generator=generator)
    return t0 + F + eps


def observe(F, m, sigma, live, generator=None):
    """Centred noisy picks d~ = C (F(m) + eps): what the locator network sees.

    F: forward operator, F(m_x, m_y) -> travel times (s), shape (n, N)
    m: sources (km), shape (n, 2)
    sigma: standard deviation of a pick (s)
    live: which stations work, bool, shape (N,); a muted station is left out of the
        mean and reads 0
    generator: optional torch.Generator, for reproducible noise
    returns: centred picks (s), shape (n, N)
    """
    F_m = F(m[:, 0], m[:, 1])
    d = F_m + sigma * torch.randn(F_m.shape, dtype=F_m.dtype, generator=generator)
    d_c = d - d[:, live].mean(-1, keepdim=True)
    return d_c * live
