"""What a density lets us do that a point cannot: combine it with a new prior
(anywidget).

Colours follow the course convention: prior blue, likelihood green, posterior red.
Two maps of one event, zoomed on its answer. Left: the likelihood (the network's
answer, with its training prior divided out) and a new prior, a band across the map.
Right: their product, the posterior, with the prior as its line only. One line per
distribution, its 1σ level: the prior's ±w, the 68 % region of the likelihood and of
the posterior. The prior's line is drawn in both maps and dragged to change its
width; its two anchors (left map) move it. One number reads off what changed.
Everything is computed in the browser from one grid, so dragging needs no Python.
"""

from __future__ import annotations

import base64

import anywidget
import numpy as np
import torch
import traitlets

from .density_plotting import _joint_limits
from .geometry import BOX
from .posterior import grid_log_prior, grid_sources, posterior_from_log
from .widgetkit import esm

N_NODES = 201  # grid nodes per side (0.5 km): thin lobes stay resolved


def _probability(density: np.ndarray) -> str:
    """Probability per grid cell, row-major [y, x], as little-endian float32."""
    p = density.astype("<f4")
    return base64.b64encode((p / p.sum()).tobytes()).decode()


class CombineFigure(anywidget.AnyWidget):
    """The network's likelihood times a new prior, a band across the map."""

    _esm = esm("combinewidget.js")

    box = traitlets.Float(BOX).tag(sync=True)
    n = traitlets.Int(N_NODES).tag(sync=True)  # grid nodes per side
    p_likelihood = traitlets.Unicode("").tag(sync=True)  # probability per cell, base64
    lo = traitlets.List().tag(sync=True)  # lower left corner of the (square) view, km
    hi = traitlets.List().tag(sync=True)
    m_true = traitlets.List().tag(sync=True)
    stations = traitlets.List().tag(sync=True)
    line_y = traitlets.Float(allow_none=True).tag(sync=True)  # a line of stations
    band = traitlets.List().tag(sync=True)  # two points of its centre line: x0 y0 x1 y1
    width = traitlets.Float(4.0).tag(
        sync=True
    )  # standard deviation across it (km), dragged

    @classmethod
    def from_answers(
        cls,
        posterior_of,
        d,
        m_true,
        xi_x,
        xi_y,
        train_prior="uniform",
        strike=20.0,
        width=4.0,
    ):
        """Widget for centred picks ``d`` (1, N) of an event at ``m_true`` (km).

        ``posterior_of(d)`` is the network's posterior in km (a torch distribution
        with batch shape (n,)), learned under ``train_prior``; dividing that prior out
        leaves the likelihood. The new prior opens as a band through the true event at
        angle ``strike`` (degrees, counter-clockwise from east); ``width`` (km) is its
        standard deviation across at the start. The figure drags its place at the
        anchors and its width at its ±w line, in either map."""
        grid = grid_sources(N_NODES)
        with torch.no_grad():
            log_q = posterior_of(d.expand(len(grid), -1)).log_prob(grid.float())
        log_rho = grid_log_prior(train_prior, N_NODES)  # -inf where it has no events
        log_like = torch.where(
            torch.isfinite(log_rho),
            log_q.double() - log_rho,
            torch.full_like(log_rho, -torch.inf),
        )
        likelihood = posterior_from_log(log_like, N_NODES)
        m_true = np.asarray(m_true, dtype=float)
        xi = np.stack([np.asarray(xi_x, dtype=float), np.asarray(xi_y, dtype=float)], 1)
        lo, hi = _joint_limits(likelihood, [m_true], pad=6.0)
        half = max(max(hi - lo) / 2, 12.0)  # a square view inside the map
        centre = np.clip((lo + hi) / 2, half, BOX - half)
        direction = np.array([np.cos(np.radians(strike)), np.sin(np.radians(strike))])
        reach = 0.4 * half * direction  # the band's two ends, either side of the event
        ends = np.stack([m_true - reach, m_true + reach])
        return cls(
            p_likelihood=_probability(likelihood.density.numpy()),
            lo=(centre - half).round(2).tolist(),
            hi=(centre + half).round(2).tolist(),
            m_true=m_true.round(2).tolist(),
            stations=xi.round(2).tolist(),
            line_y=float(xi[0, 1]) if np.ptp(xi[:, 1]) < 1e-6 else None,
            band=ends.round(2).flatten().tolist(),
            width=width,
        )
