"""Figures for Part 1b: the training pairs, the split, the scaling, the best possible
answer and the learning curves. Same style as ``plotting.py``: sources are
vermillion, the exact posterior is green, training is blue.

Networks and training stay in the notebook. The figures that only illustrate the
simulator (``dataset_figure``, ``split_figure``, ``posterior_figure``) draw their own
example events from the scene they are given."""

from __future__ import annotations

import numpy as np

from . import physics
from .density_plotting import _contours, _zoom
from .geometry import BOX
from .plotting import TOKENS, _c, _draw_prior, map_axes, plot_stations, style
from .posterior import grid_posterior
from .priors import sample_prior

NO_EVENTS = np.empty((0, 2))


def _stars(ax, m, colour, label=None):
    """Sources as numbered stars on a map."""
    for k, (x, y) in enumerate(np.asarray(m), start=1):
        ax.plot(x, y, "*", color=colour, ms=14, mec="white", mew=0.8, zorder=6)
        ax.annotate(
            str(k), (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10,
            color=TOKENS["ink"], zorder=7,
        )  # fmt: skip


def dataset_figure(scene, prior="uniform", seed=3, n_events=1000):
    """Training pairs from ``prior`` at the stations of ``scene``, shown as the numbers
    a program would print: the map with all ``n_events`` sources (three of
    them starred), and next to it the rows of the two arrays the network learns from,
    the sources ``m`` (km) and the centred picks ``d~`` (s) the stations record."""
    import matplotlib.pyplot as plt
    import torch

    torch.manual_seed(seed)
    m = sample_prior(prior, 3)
    d = physics.observe(scene.F, m, scene.sigma, scene.live)
    others = sample_prior(prior, max(n_events - 3, 0))  # the rest of the set
    xi_x, xi_y, n_stations = scene.xi_x, scene.xi_y, scene.n_stations
    mono = {"family": "DejaVu Sans Mono", "size": 11, "va": "center"}
    ink, ink2, red = TOKENS["ink"], TOKENS["ink2"], _c("C1")

    def row(values, spec):  # a printed list of numbers
        return "[" + " ".join(f"{v:{spec}}" for v in values) + "]"

    with plt.style.context(style("boxed")):
        fig, (ax, out) = plt.subplots(
            1, 2, figsize=(9, 3.6), layout="constrained", width_ratios=[1, 1.75]
        )
        _draw_prior(ax, prior, NO_EVENTS, red, xi_x, xi_y)
        ax.scatter(*others.numpy().T, s=3, color=red, lw=0, zorder=4)
        _stars(ax, m.numpy(), red)
        ax.set_title(f"{n_events} sources from the {prior} prior")

        out.set_axis_off()
        out.set_xlim(0, 1)
        out.set_ylim(0, 1)
        col = lambda c: c / 55.5  # noqa: E731  # monospace: one character is 1/55.5 of the axes
        x_event, x_m, x_d = col(0), col(7), col(25)
        out.text(x_event, 0.84, "event", color=ink, **mono)
        out.text(x_m, 0.84, "m (km)", color=red, **mono)
        out.text(
            x_d,
            0.84,
            rf"$\tilde{{d}}$ (s), at the {n_stations} stations",
            color=ink,
            **mono,
        )
        for k in range(len(m)):
            y = 0.68 - 0.15 * k
            out.text(col(2), y, str(k + 1), color=ink, **mono)
            out.text(x_m, y, row(m[k], "5.1f"), color=red, **mono)
            out.text(x_d, y, row(d[k], "5.2f"), color=ink, **mono)
        y = 0.68 - 0.15 * len(m)
        for x in (col(2), x_m + col(5), x_d + col(15)):
            out.text(
                x, y, "\u22ee", color=ink2, family="DejaVu Sans", size=12, va="center"
            )
        out.text(x_event, 0.10, "shape", color=ink2, **mono)
        out.text(x_m, 0.10, f"({n_events}, 2)", color=red, **mono)
        out.text(x_d, 0.10, f"({n_events}, {n_stations})", color=ink, **mono)
    plt.close(fig)
    return fig


def split_figure(scene, sizes, prior="uniform", seed=0):
    """The events of the train, validation and test set (``sizes`` of each, drawn
    from ``prior``), each on a map of its own."""
    import matplotlib.pyplot as plt
    import torch

    torch.manual_seed(seed)
    sets = (("train", "C0"), ("validation", "C1"), ("test", "C4"))
    with plt.style.context(style("boxed")):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout="constrained")
        for ax, (name, colour), n in zip(axes, sets, sizes, strict=True):
            events = sample_prior(prior, n)
            _draw_prior(ax, prior, events, _c(colour), scene.xi_x, scene.xi_y)
            ax.set_title(f"{name}: {n} events")
            ax.set_xlabel("x (km)")
        axes[1].set_ylabel("")
        axes[2].set_ylabel("")
    plt.close(fig)
    return fig


def scaling_figure(d_raw, d_scaled, m_raw, m_scaled):
    """Histograms of the network's inputs (all centred picks) and outputs (both
    source coordinates), before and after the scaling."""
    import matplotlib.pyplot as plt

    d_raw, d_scaled = np.ravel(d_raw), np.ravel(d_scaled)
    m_raw, m_scaled = np.asarray(m_raw), np.asarray(m_scaled)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(2, 2, figsize=(9, 4.1), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.12, w_pad=0.12, hspace=0.1, wspace=0.08)
        rows = [
            ("centred picks", (d_raw,), (d_scaled,), "s", TOKENS["ink2"]),
            ("locations", tuple(m_raw.T), tuple(m_scaled.T), "km", _c("C1")),
        ]
        for row, (name, raw, scaled, unit, colour) in enumerate(rows):
            for col, (values, tail) in enumerate(((raw, unit), (scaled, "scaled"))):
                ax = axes[row, col]
                for i, v in enumerate(values):
                    ax.hist(
                        v, bins=40, histtype="step", color=colour, lw=1.6,
                        ls="-" if i == 0 else "--",
                    )  # fmt: skip
                ax.set_yticks([])
                ax.tick_params(labelsize=11)
                if row:  # headroom for the legend
                    ax.set_ylim(0, 1.45 * ax.get_ylim()[1])
                spread = np.concatenate(values).std()
                ax.set_xlabel(f"{name} ({tail}): std {spread:.2f}", fontsize=12)
        axes[1, 0].plot([], [], color=_c("C1"), lw=1.6, label="$m_x$")
        axes[1, 0].plot([], [], color=_c("C1"), lw=1.6, ls="--", label="$m_y$")
        axes[1, 0].legend(loc="upper center", fontsize=12, ncols=2, frameon=False)
    plt.close(fig)
    return fig


def posterior_figure(scene, prior, sigma=1.0, seed=5):
    """Prior, likelihood and posterior of one event drawn from ``prior``, picked
    with pick noise ``sigma`` (larger than the scene's, so that the prior shows).
    The likelihood and the posterior are zoomed to the same square box, marked on
    the prior, with the same ticks on both axes. The diamond is the posterior mean,
    the best single answer under a squared error."""
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MultipleLocator

    torch.manual_seed(seed)
    m_true = sample_prior(prior, 1)
    d = physics.observe(scene.F, m_true, sigma, scene.live)[0]
    likelihood = grid_posterior(scene.F, d, sigma, None, scene.live)  # flat prior
    posterior = grid_posterior(scene.F, d, sigma, prior, scene.live)
    m_true, xi_x, xi_y = m_true[0], scene.xi_x, scene.xi_y

    nodes = np.linspace(0, BOX, likelihood.density.shape[0])
    inside = (likelihood.density >= likelihood.level_95).numpy()
    ys, xs = np.nonzero(inside)
    box = np.array(
        [[nodes[xs.min()], nodes[ys.min()]], [nodes[xs.max()], nodes[ys.max()]]]
    )
    star = {
        "marker": "*",
        "mfc": "white",
        "mec": "k",
        "ms": 15,
        "mew": 1.2,
        "zorder": 6,
    }
    with plt.style.context(style("boxed")):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout="constrained")
        _draw_prior(axes[0], prior, NO_EVENTS, _c("C1"), xi_x, xi_y)
        axes[0].set_title(f"prior ({prior})")
        axes[0].plot(*m_true, ls="", **star)
        axes[0].add_patch(Rectangle(
            box[0] - 4, *(box[1] - box[0] + 8), fill=False, ec=TOKENS["ink"], lw=1.0,
            zorder=8,
        ))  # fmt: skip
        for ax, post, title, colour, fill in (
            (axes[1], likelihood, "likelihood of the picks", "C2", "Greens"),
            (axes[2], posterior, "posterior = prior × likelihood", "C1", "Reds"),
        ):
            _contours(ax, post, _c(colour), fill=fill)
            plot_stations(ax, xi_x, xi_y, s=60)
            ax.plot(*m_true, ls="", **star)
            ax.set_title(title)
        axes[2].plot(
            *posterior.mean, "D", color="k", ms=6, mec="white", zorder=7,
            label="posterior mean",
        )  # fmt: skip
        axes[2].plot([], [], ls="", label="true source", **star)
        axes[2].legend(loc="lower left", fontsize=8)
        for ax in axes:
            map_axes(ax)
            ax.set_xlabel("x (km)")
        for ax in axes[1:]:
            _zoom(ax, box, pad=6.0)
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(MultipleLocator(10))  # same ticks on x and y
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.set_ylabel("")
    plt.close(fig)
    return fig


def learning_curves_figure(runs, floor):
    """Training and validation error (RMS, km) against the epoch of several runs,
    side by side, with the best possible error dashed. ``runs`` is a list of
    ``(title, rms_train, rms_val, best_epoch)``."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            1, len(runs), figsize=(9, 3.4), layout="constrained", sharey=True
        )
        for ax, (title, rms_train, rms_val, best) in zip(axes, runs, strict=True):
            ax.plot(rms_train, color=_c("C0"), label="training")
            ax.plot(rms_val, color=_c("C1"), label="validation")
            ax.axhline(floor, color=_c("C2"), ls="--", lw=1.2, label="best possible")
            ax.plot(best, rms_val[best], "v", color=_c("C1"), ms=7, zorder=5)
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()  # plain tick labels, without widening the view
            ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
            ax.set_ylim(lo, hi)
            ax.minorticks_off()
            ax.set_title(title)
            ax.set_xlabel("epoch")
        axes[0].set_ylabel("RMS location error (km)")
        axes[0].legend(loc="upper right", fontsize=7.5)
    plt.close(fig)
    return fig


def cost_figure(n=100, ks=(4, 8), dims=range(2, 11)):
    """Numbers needed to hold the answer against the number of unknowns ``d``: a grid
    with ``n`` nodes per axis (``n ** d``, the exact posterior) and Gaussian mixtures
    with ``K`` components, each K logits, K means and K Cholesky factors,
    ``K (1 + d + d (d + 1) / 2)`` numbers (6 K for d = 2)."""
    import matplotlib.pyplot as plt

    d = np.array(list(dims))
    with plt.style.context(style("open")):
        grid_colour, vermillion = TOKENS["ink"], _c("C1")
        fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
        ax.plot(
            d, float(n) ** d, "o-", color=grid_colour, label=f"grid, {n} nodes per axis"
        )
        for k, dash in zip(ks, ("-", "--"), strict=False):
            ax.plot(
                d, k * (1 + d + d * (d + 1) / 2), "o" + dash, color=vermillion,
                label=f"Gaussian mixture, K = {k}",
            )  # fmt: skip
        ax.set_yscale("log")
        ax.set_yticks([1e2, 1e6, 1e10, 1e14, 1e18])
        ax.minorticks_off()
        ax.set_xticks(
            [2, 4, 10],
            ["2\nepicentre", "4\nhypocentre,\norigin time", "10\n+ moment\ntensor"],
        )
        ax.set_xlim(1.7, 10.3)
        ax.set_xlabel("unknowns d")
        ax.set_ylabel("numbers to describe the answer")
        ax.annotate(
            "this lecture", (2, float(n) ** 2), xytext=(2.4, 1e9),
            arrowprops={"arrowstyle": "-", "color": TOKENS["ink2"], "lw": 0.8},
            color=TOKENS["ink"],
        )  # fmt: skip
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=7.5)
    plt.close(fig)
    return fig


__all__ = [
    "cost_figure",
    "dataset_figure",
    "learning_curves_figure",
    "posterior_figure",
    "scaling_figure",
    "split_figure",
]
