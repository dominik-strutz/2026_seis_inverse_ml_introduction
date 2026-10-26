"""Helpers to draw posteriors and Gaussian mixtures on the map, shared by the widgets
and figures of Parts 1b and 1c. Same style as ``plotting.py``: the exact posterior
is green, anything the network returns is vermillion."""

from __future__ import annotations

import numpy as np

from .geometry import BOX
from .posterior import GRID, Posterior


def mixture_components(q):
    """Weight, mean (km) and covariance (km²) of each component, if ``q`` is a
    Gaussian mixture for one event, possibly behind an affine map from network
    units to km (``TransformedDistribution``); ``None`` for any other density."""
    from torch.distributions import MixtureSameFamily, TransformedDistribution

    base, loc, scale = q, 0.0, 1.0
    if isinstance(q, TransformedDistribution):
        if len(q.transforms) != 1 or not hasattr(q.transforms[0], "scale"):
            return None
        base, loc, scale = q.base_dist, q.transforms[0].loc, q.transforms[0].scale
    if not isinstance(base, MixtureSameFamily):
        return None
    components = base.component_distribution
    weights = base.mixture_distribution.probs.reshape(-1, components.loc.shape[-2])
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (2,))
    loc = np.broadcast_to(np.asarray(loc, dtype=float), (2,))
    means = components.loc.reshape(-1, *components.loc.shape[-2:])[0].detach()
    covs = components.covariance_matrix.reshape(-1, *components.loc.shape[-2:], 2)
    return [
        (
            float(w),
            loc + scale * mu.double().numpy(),
            cov.double().numpy() * scale[:, None] * scale[None, :],
        )
        for w, mu, cov in zip(weights[0].detach(), means, covs[0].detach(), strict=True)
    ]


def _nodes(n: int = GRID):
    return np.linspace(0, BOX, n)


def _contours(ax, post, colour, label=None, fill=None, lw=1.4, lines=True):
    """68 % (solid) and 95 % (dashed) regions of a grid posterior; ``fill`` a
    colormap name to shade the density underneath."""
    density = post.density.numpy()
    nodes = _nodes(density.shape[0])
    if fill:
        ax.contourf(
            nodes, nodes, density / density.max(), levels=np.linspace(0.02, 1, 12),
            cmap=fill, alpha=0.9,
        )  # fmt: skip
    if lines:
        ax.contour(
            nodes, nodes, density, levels=sorted([post.level_95, post.level_68]),
            colors=[colour], linestyles=["--", "-"], linewidths=lw,
        )  # fmt: skip
    if label:  # a proxy line for the legend
        ax.plot([], [], color=colour, lw=lw, label=label)


def _zoom(ax, points, pad=8.0):
    """Square limits around ``points`` (k, 2) km, kept inside the map."""
    points = np.asarray(points)
    lo, hi = points.min(0) - pad, points.max(0) + pad
    half = max(hi - lo) / 2
    centre = np.clip((lo + hi) / 2, half, BOX - half)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)


def _joint_limits(post: Posterior, points, pad=8.0, aspect=(1.0, 1.0)):
    """Map limits around the 95 % region of ``post`` and the extra ``points`` (k, 2)
    km, kept inside the map. ``aspect`` bounds width / height of the box so the
    narrow lobes of a line array still get a usable panel."""
    density = post.density.numpy()
    nodes = _nodes(density.shape[0])
    iy, ix = np.nonzero(density >= post.level_95)
    corners = np.array([[nodes[ix].min(), nodes[iy].min()],
                        [nodes[ix].max(), nodes[iy].max()]])  # fmt: skip
    corners = np.vstack([corners, np.asarray(points, dtype=float).reshape(-1, 2)])
    lo, hi = corners.min(0) - pad, corners.max(0) + pad
    centre, size = (lo + hi) / 2, hi - lo
    size[0] = np.clip(size[0], aspect[0] * size[1], aspect[1] * size[1])
    size[1] = max(size[1], size[0] / aspect[1])
    size = np.minimum(size, BOX)
    centre = np.clip(centre, size / 2, BOX - size / 2)
    return centre - size / 2, centre + size / 2
