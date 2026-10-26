"""Where the point summaries of a posterior land (anywidget).

The map, zoomed in, with the posterior of m_x above it and of m_y beside it (the
other direction summed out). Each summary is a marker on the map whose coordinates
are carried by thin lines to dots on the two marginal curves. A list on the right
names every summary and gives its position and its distance to the true event:
hovering a row (or a marker) isolates that summary, clicking a row hides it.
"""

from __future__ import annotations

import anywidget
import numpy as np
import torch
import traitlets

from .density_plotting import _joint_limits, _nodes
from .geometry import BOX
from .posterior import Posterior, grid_sources
from .widgetkit import esm, png_data_url

# the exact posterior is green, anything a network returns vermillion; the three
# summaries of the exact posterior take blue, orange and purple
GREEN = (0, 158, 115)
SUMMARIES = [
    # key, name, what it minimises, marker, colour
    ("mean", "posterior mean", "minimises the squared error (MSE)", "diamond",
     "#0072B2"),
    ("median", "median", "minimises the absolute error (MAE)", "square", "#E69F00"),
    ("map", "most likely point", "the highest peak of the density (MAP)", "ring",
     "#CC79A7"),
]  # fmt: skip


def _density_image(
    density: np.ndarray, upsample: int = 4, gamma: float = 0.6, colour=GREEN
) -> str:
    """The density as a green colour ramp that fades to transparent at zero, bicubic
    upsampled so the zoomed map does not show the 1 km grid cells."""
    n = density.shape[0]
    d = torch.as_tensor(density, dtype=torch.float32)[None, None]
    d = torch.nn.functional.interpolate(
        d, size=(upsample * (n - 1) + 1,) * 2, mode="bicubic", align_corners=True
    )[0, 0].clamp_min(0)
    d = (d / d.max()).numpy() ** gamma
    rgba = np.empty((*d.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.array(colour) / 255
    rgba[..., 3] = 0.85 * d
    return png_data_url(rgba)


def _contour_lines(density: np.ndarray, level: float) -> list:
    """The ``level`` contour of a grid density as polylines in km."""
    from contourpy import contour_generator

    nodes = _nodes(density.shape[0])
    lines = contour_generator(nodes, nodes, density).lines(level)
    return [np.round(line, 2).tolist() for line in lines]


class SummariesFigure(anywidget.AnyWidget):
    """Map plus both marginals of one posterior, with its point summaries."""

    _esm = esm("summarieswidget.js")

    box = traitlets.Float(BOX).tag(sync=True)
    lo = traitlets.List().tag(sync=True)  # lower left corner of the map view (km)
    hi = traitlets.List().tag(sync=True)  # upper right corner (km)
    image = traitlets.Unicode("").tag(sync=True)  # density over the whole map, PNG
    contours = traitlets.Dict().tag(sync=True)  # "68", "95" -> polylines (km)
    marginal_x = traitlets.List().tag(sync=True)  # p(m_x | d), 1/km, on the nodes
    marginal_y = traitlets.List().tag(sync=True)
    marks = traitlets.List().tag(sync=True)  # the true event and each summary
    stations = traitlets.List().tag(sync=True)

    @classmethod
    def from_posterior(cls, post: Posterior, m_true, xi_x, xi_y, m_hat=None):
        """Widget for the grid posterior ``post`` of one event at ``m_true`` (km),
        seen by stations at (``xi_x``, ``xi_y``); ``m_hat`` adds a network's answer."""
        density = post.density.numpy()
        n = density.shape[0]
        nodes = _nodes(n)
        p = density / density.sum()
        m = grid_sources(n).numpy()
        points = {
            "mean": p.flatten() @ m,
            "median": [np.interp(0.5, np.cumsum(p.sum(a)), nodes) for a in (0, 1)],
            "map": m[np.argmax(p)],
        }
        marks = [("true", "true event", "where the event really is", "star", "#111111",
                  np.asarray(m_true, dtype=float))]  # fmt: skip
        marks += [(k, name, sub, shape, colour, np.asarray(points[k], dtype=float))
                  for k, name, sub, shape, colour in SUMMARIES]  # fmt: skip
        if m_hat is not None:
            marks.append(
                ("network", "MSE network", "what a trained network answers",
                 "hexagon", "#D55E00", np.asarray(m_hat, dtype=float))
            )  # fmt: skip
        lo, hi = _joint_limits(post, [mark[-1] for mark in marks])
        dx = nodes[1] - nodes[0]
        truth = marks[0][-1]
        return cls(
            lo=[float(v) for v in lo],
            hi=[float(v) for v in hi],
            image=_density_image(density),
            contours={
                "68": _contour_lines(density, post.level_68),
                "95": _contour_lines(density, post.level_95),
            },
            marginal_x=np.round(p.sum(0) / dx, 5).tolist(),
            marginal_y=np.round(p.sum(1) / dx, 5).tolist(),
            marks=[
                {
                    "key": key,
                    "name": name,
                    "sub": sub,
                    "shape": shape,
                    "colour": colour,
                    "x": round(float(pt[0]), 2),
                    "y": round(float(pt[1]), 2),
                    "error": None
                    if key == "true"
                    else round(float(np.hypot(*(pt - truth))), 1),
                }
                for key, name, sub, shape, colour, pt in marks
            ],
            stations=np.round(
                np.stack(
                    [np.asarray(xi_x, dtype=float), np.asarray(xi_y, dtype=float)], 1
                ),
                2,
            ).tolist(),
        )
