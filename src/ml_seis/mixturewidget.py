"""How many Gaussians a density needs (anywidget).

A target density on the map (green) with the samples a Gaussian mixture is fitted to
(blue dots). Two controls pick how many samples the mixture is fitted to and the
number of components K; the second map shows the mixture fitted with those and its
components, the plot beside it the NLL of the fitted and of held-out samples for
every K. Everything is computed once in Python, both are changed in the browser.
"""

from __future__ import annotations

import anywidget
import numpy as np
import torch
import traitlets

from .density_plotting import _joint_limits, mixture_components
from .geometry import BOX
from .posterior import Posterior, grid_sources, posterior_from_log
from .summarieswidget import _contour_lines, _density_image
from .widgetkit import esm

VERMILLION = (213, 94, 0)


def _ellipse(cov: np.ndarray) -> dict:
    """One standard deviation ellipse of a covariance: semi-axes (km) and the angle
    (degrees, counter-clockwise from the x axis) of the first."""
    values, vectors = np.linalg.eigh(cov)
    return {
        "a": round(float(np.sqrt(max(values[1], 0))), 2),
        "b": round(float(np.sqrt(max(values[0], 0))), 2),
        "angle": round(float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))), 1),
    }


class MixtureFitFigure(anywidget.AnyWidget):
    """Target density, fitted samples, and a mixture of K components."""

    _esm = esm("mixturewidget.js")

    box = traitlets.Float(BOX).tag(sync=True)
    image = traitlets.Unicode("").tag(sync=True)  # the target density, PNG
    fitted = traitlets.List().tag(sync=True)  # per n: the fitted samples
    ns = traitlets.List().tag(sync=True)  # the numbers of fitted samples
    ks = traitlets.List().tag(sync=True)  # the numbers of components
    nll_fit = traitlets.List().tag(sync=True)  # per n: NLL of the fitted samples, per K
    nll_held = traitlets.List().tag(sync=True)  # same for the held-out samples
    contours = traitlets.Dict().tag(sync=True)  # the target's 68 / 95 % regions
    models = traitlets.List().tag(sync=True)  # per n, per K: density image, components
    lo = traitlets.List().tag(sync=True)  # lower left corner of the (square) view, km
    hi = traitlets.List().tag(sync=True)
    n_held = traitlets.Int(0).tag(sync=True)  # held-out samples behind the NLL
    k = traitlets.Int(2).tag(sync=True)  # the K shown
    n = traitlets.Int(0).tag(sync=True)  # the number of fitted samples shown

    @classmethod
    def from_fits(cls, post: Posterior, m_fit, m_held, fits: dict, k: int = 2, n=None):
        """``fits`` maps a number of fitted samples to {K: the fitted density over
        km (a torch distribution)}; the figure opens on k and the middle count."""
        grid = grid_sources().float()
        ns, ks = list(fits), list(fits[next(iter(fits))])
        n = ns[len(ns) // 2] if n is None else n
        models, nll_fit, nll_held, fitted = [], [], [], []
        for n_i in ns:
            m_i = m_fit[:n_i]
            row, s_fit, s_held = [], [], []
            for K in ks:
                q = fits[n_i][K]
                with torch.no_grad():
                    shown = posterior_from_log(q.log_prob(grid))
                    s_fit.append(-q.log_prob(m_i.float()).mean().item())
                    s_held.append(-q.log_prob(m_held.float()).mean().item())
                row.append(
                    {
                        "image": _density_image(
                            shown.density.numpy(), upsample=2, colour=VERMILLION
                        ),
                        "components": [
                            {
                                "w": round(w, 3),
                                "x": round(float(mean[0]), 2),
                                "y": round(float(mean[1]), 2),
                                **_ellipse(cov),
                            }
                            for w, mean, cov in mixture_components(q)
                        ],
                    }
                )
            models.append(row)
            nll_fit.append(np.round(s_fit, 3).tolist())
            nll_held.append(np.round(s_held, 3).tolist())
            fitted.append(np.round(m_i.numpy(), 2).tolist())

        lo, hi = _joint_limits(post, m_fit.numpy(), pad=4.0)
        centre, half = (lo + hi) / 2, max(hi - lo) / 2
        centre = np.clip(centre, half, BOX - half)  # a square view inside the map
        return cls(
            lo=(centre - half).round(2).tolist(),
            hi=(centre + half).round(2).tolist(),
            image=_density_image(post.density.numpy(), gamma=0.4),
            contours={
                "68": _contour_lines(post.density.numpy(), post.level_68),
                "95": _contour_lines(post.density.numpy(), post.level_95),
            },
            fitted=fitted,
            n_held=len(m_held),
            ns=ns,
            ks=ks,
            nll_fit=nll_fit,
            nll_held=nll_held,
            models=models,
            k=k,
            n=n,
        )
