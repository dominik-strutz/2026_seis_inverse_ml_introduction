"""Figures for the second half of Lecture 1: posteriors, Gaussian mixtures and their
calibration. Same style as ``plotting.py``: the exact posterior
is green, anything the network returns is vermillion."""

from __future__ import annotations

import numpy as np

from .geometry import BOX
from .plotting import _c, map_axes, plot_stations, style
from .posterior import GRID, Posterior, grid_sources, posterior_from_log


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


def _contours(ax, post: Posterior, colour, label=None, fill=None, lw=1.4):
    """68 % (solid) and 95 % (dashed) regions of a grid posterior; ``fill`` a
    colormap name to shade the density underneath."""
    density = post.density.numpy()
    nodes = _nodes(density.shape[0])
    if fill:
        ax.contourf(
            nodes, nodes, density / density.max(), levels=np.linspace(0.02, 1, 12),
            cmap=fill, alpha=0.9,
        )  # fmt: skip
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


def _ellipse(ax, mean, cov, colour, weight=None, n_std=1.0):
    """A component of a Gaussian mixture: its 1-standard-deviation ellipse."""
    from matplotlib.patches import Ellipse

    values, vectors = np.linalg.eigh(np.asarray(cov))
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(values[::-1], 0))
    ax.add_patch(
        Ellipse(mean, w, h, angle=angle, fill=False, ec=colour, lw=1.1, ls=(0, (3, 2)))
    )
    if weight is not None:
        ax.annotate(
            f"{weight:.2f}", mean, textcoords="offset points", xytext=(5, 4),
            fontsize=8, color=colour,
        )  # fmt: skip


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


def summaries_figure(post: Posterior, m_true, xi_x, xi_y, m_hat=None):
    """Where the point summaries of a two-lobe posterior land: the map, zoomed in,
    with the posterior of m_x above and of m_y beside it (the other direction summed
    out), as in a seaborn joint plot. The summaries are dotted lines through all
    three panels."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    density = post.density.numpy()
    nodes = _nodes(density.shape[0])
    p = density / density.sum()
    m = grid_sources(density.shape[0]).numpy()
    mean = p.flatten() @ m
    median = [np.interp(0.5, np.cumsum(p.sum(a)), nodes) for a in (0, 1)]
    peak = m[np.argmax(p)]
    dx = nodes[1] - nodes[0]
    marginal_x, marginal_y = p.sum(0) / dx, p.sum(1) / dx  # densities, 1/km
    marks = [
        (mean, "D", "posterior mean (MSE)", "C2"),
        (median, "s", "median (absolute error)", "C0"),
        (peak, "P", "most likely point (MAP)", "C4"),
    ]
    if m_hat is not None:
        marks.append((np.asarray(m_hat, dtype=float), "o", "MSE network", "C1"))
    lo, hi = _joint_limits(post, [pt for pt, *_ in marks] + [m_true])

    # panel sizes in inches: the map is exactly to scale (equal aspect), the
    # marginals share its axes
    height, gap, margin = 2.6, 0.07, 0.62
    width = height * (hi[0] - lo[0]) / (hi[1] - lo[1])
    side = 0.7  # marginal panels
    with plt.style.context(style("boxed")):
        green = _c("C2")
        fig = plt.figure(
            figsize=(margin + width + gap + side + 2.9, height + side + 0.62),
            layout="none",
        )
        fw, fh = fig.get_size_inches()

        def panel(left, bottom, w, h, **share):
            return fig.add_axes([left / fw, bottom / fh, w / fw, h / fh], **share)

        ax = panel(margin, 0.5, width, height)
        ax_x = panel(margin, 0.5 + height + gap, width, side, sharex=ax)
        ax_y = panel(margin + width + gap, 0.5, side, height, sharey=ax)

        _contours(ax, post, green, fill="Greens")
        plot_stations(ax, xi_x, xi_y, s=30)
        for point, marker, _, colour in marks:
            ax.plot(*point, marker, color=_c(colour), ms=8, mec="w", zorder=5)
        ax.plot(*m_true, "*", mfc="none", mec="k", ms=15, mew=1.2, zorder=6)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("$m_x$ (km)")
        ax.set_ylabel("$m_y$ (km)")

        # marginals: filled density, summaries as dotted lines through every panel
        ax_x.fill_between(nodes, marginal_x, color=green, alpha=0.25, lw=0)
        ax_x.plot(nodes, marginal_x, color=green, lw=1.2)
        ax_y.fill_betweenx(nodes, marginal_y, color=green, alpha=0.25, lw=0)
        ax_y.plot(marginal_y, nodes, color=green, lw=1.2)
        ax_x.set_ylim(0, 1.1 * marginal_x.max())
        ax_y.set_xlim(0, 1.1 * marginal_y.max())
        # the summaries often share a coordinate: stagger the dashes so every
        # colour shows
        lines = [(m_true, "k")] + [(point, _c(colour)) for point, _, _, colour in marks]
        for i, (point, colour) in enumerate(lines):
            dotted = (i * 3 / len(lines), (1.0, 2.0))
            for a in (ax, ax_x):
                a.axvline(point[0], color=colour, lw=2.0, ls=dotted, zorder=2)
            for a in (ax, ax_y):
                a.axhline(point[1], color=colour, lw=2.0, ls=dotted, zorder=2)
        ax_x.text(0.02, 0.92, r"$p(m_x \mid \tilde{\mathbf{d}})$", va="top",
                  transform=ax_x.transAxes)  # fmt: skip
        ax_y.text(0.96, 0.99, r"$p(m_y \mid \tilde{\mathbf{d}})$", va="top",
                  ha="right", transform=ax_y.transAxes)  # fmt: skip
        for a in (ax_x, ax_y):
            a.set_axis_off()

        handles = [
            Line2D([], [], color=green, lw=1.4, label="posterior 68 / 95 %"),
            Line2D([], [], color="k", marker="*", mfc="none", mew=1.2, ms=15, ls=":",
                   lw=2.0, label="true event"),
        ]  # fmt: skip
        handles += [
            Line2D(
                [],
                [],
                color=_c(colour),
                marker=marker,
                ms=8,
                mec="w",
                ls=":",
                lw=2.0,
                label=label,
            )
            for _, marker, label, colour in marks
        ]
        handles.append(
            Line2D([], [], color=_c("C6"), marker="^", ls="", ms=6, label="stations")
        )
        legend_left = (margin + width + gap + side + 0.25) / fw
        fig.legend(handles=handles, loc="center left", fontsize=8,
                   bbox_to_anchor=(legend_left, 0.5))  # fmt: skip
    plt.close(fig)
    return fig


def density_fit_figure(post: Posterior, samples, log_q, components, curves, chosen):
    """One learned density against the exact posterior it was fitted to.

    ``samples`` (n, 2) are the fitting samples, ``log_q`` the mixture's log density
    on the grid nodes (n_grid²,), ``components`` a list of (weight, mean, cov),
    ``curves`` the NLL per number of components ({"fit": {K: nll}, "held-out":
    {K: nll}}) and ``chosen`` the K shown on the map.
    """
    import matplotlib.pyplot as plt

    learned = posterior_from_log(log_q)
    with plt.style.context(style("boxed")):
        fig, (ax, ax_nll) = plt.subplots(
            1, 2, figsize=(9, 4.0), layout="constrained", width_ratios=[1, 1.15]
        )
        _contours(
            ax, learned, _c("C1"), f"mixture, K = {chosen}: 68 / 95 %", fill="Oranges"
        )
        _contours(ax, post, _c("C2"), "exact posterior 68 / 95 %")
        ax.scatter(*np.asarray(samples).T, s=3, color=_c("C6"), alpha=0.5, lw=0,
                   label="samples it was fitted to")  # fmt: skip
        for weight, mean, cov in components or []:
            _ellipse(ax, mean, cov, _c("C1"), weight)
        map_axes(ax)
        _zoom(ax, samples, pad=16)
        ax.legend(loc="lower left", fontsize=7.5)
        ks = sorted(curves["fit"])
        ax_nll.plot(ks, [curves["fit"][k] for k in ks], "o-", color=_c("C0"), ms=4,
                    label="fitting samples")  # fmt: skip
        ax_nll.plot(ks, [curves["held-out"][k] for k in ks], "o-", color=_c("C1"),
                    ms=4, label="held-out samples")  # fmt: skip
        ax_nll.plot(chosen, curves["held-out"][chosen], "o", ms=11, mfc="none",
                    mec="k", mew=1.2)  # fmt: skip
        ax_nll.set_xticks(ks)
        ax_nll.set_xlabel("number of components $K$")
        ax_nll.set_ylabel("negative log-likelihood (per sample)")
        ax_nll.set_title("lower is better; held-out decides")
        ax_nll.legend(fontsize=7.5)
    plt.close(fig)
    return fig


def calibration_figure(levels: dict, nll: dict):
    """Coverage: the fraction of events whose true source lies inside the credible
    region of each level (``levels``: name -> credible level of every true source),
    and the mean negative log-likelihood of each method (``nll``: name -> value)."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("boxed")):
        # the exact posterior green, the networks vermillion, blue, purple, orange
        others = [k for k in levels if k != "exact posterior"]
        colours = {"exact posterior": _c("C2")}
        colours |= {k: _c(f"C{(1, 0, 4, 3)[i % 4]}") for i, k in enumerate(others)}
        fig, (ax, ax_nll) = plt.subplots(
            1, 2, figsize=(9, 3.6), layout="constrained", width_ratios=[1, 1.2]
        )
        alpha = np.linspace(0, 1, 101)
        ax.plot([0, 1], [0, 1], color=_c("C6"), lw=1, ls="--")
        for name, lev in levels.items():
            lev = np.sort(np.asarray(lev))
            covered = np.searchsorted(lev, alpha, side="right") / len(lev)
            ax.plot(alpha, covered, color=colours[name], lw=1.6, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("credible level")
        ax.set_ylabel("fraction of true sources inside")
        ax.set_title("coverage: on the diagonal is honest")
        ax.text(0.62, 0.12, "too confident", color=_c("C6"), fontsize=8)
        ax.text(0.05, 0.85, "too cautious", color=_c("C6"), fontsize=8)
        ax.legend(loc="lower right", fontsize=7.5, bbox_to_anchor=(1, 0.18))
        names = list(nll)
        values = [nll[k] for k in names]
        bar_colours = [colours.get(k, _c("C6")) for k in names]
        ax_nll.barh(names[::-1], values[::-1], color=bar_colours[::-1], height=0.55)
        for y, v in enumerate(values[::-1]):
            ax_nll.text(v, y, f" {v:.2f}", va="center", fontsize=8)
        lo = min(values)
        ax_nll.set_xlim(lo - 0.6, max(values) + 0.5)
        ax_nll.set_xlabel("mean negative log-likelihood of the true source")
        ax_nll.set_title("validation events; the exact posterior is the floor")
    plt.close(fig)
    return fig


def ensemble_figure(cases, xi_x, xi_y, train_events=None):
    """Several networks (seeds) on the same events. ``cases`` is a list of
    (title, true source, exact Posterior, [Posterior of each member])."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("boxed")):
        fig, axes = plt.subplots(
            1, len(cases), figsize=(4.5 * len(cases), 4.2), layout="constrained"
        )
        for ax, (title, m_true, exact, members) in zip(
            np.atleast_1d(axes), cases, strict=True
        ):
            if train_events is not None:
                ax.scatter(*np.asarray(train_events).T, s=2, color=_c("C6"),
                           alpha=0.25, lw=0)  # fmt: skip
            for k, member in enumerate(members):
                _contours(ax, member, _c("C1"), "each seed 68 / 95 %" if k == 0
                          else None, lw=1.0)  # fmt: skip
            _contours(ax, exact, _c("C2"), "exact posterior 68 / 95 %", lw=1.8)
            plot_stations(ax, xi_x, xi_y, s=30)
            ax.plot(*m_true, "*", color="k", ms=13, mec="w", zorder=6)
            map_axes(ax)
            ax.set_title(title)
            ax.legend(loc="lower left", fontsize=7.5)
    plt.close(fig)
    return fig
