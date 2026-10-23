"""Course figures in the shared style (see ``styles/README.md``)."""

from dataclasses import dataclass
from importlib.resources import files

from .geometry import BOX

_STYLES = files("ml_seis") / "styles"

# Colour tokens of the course style (light; see styles/README.md).
TOKENS = {
    "bg": "#ffffff",
    "ink": "#111111",
    "ink2": "#5c6370",
    "grid": "#d9dde4",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
}

STATUS_MESSAGES = {
    "diverged": "The run diverged: reduce the step size.",
    "left_map": (
        "The path left the map (open circle: where it crossed the edge; arrow: "
        "direction of the iterate). Reduce the step size."
    ),
    "not_converged": "The run has not reached the best grid basin.",
}


@dataclass(frozen=True)
class RunSummary:
    """Landscape reference and end-of-run numbers for the playground readout."""

    S_grid: object
    m_grid: object
    S_grid_min: float
    S_final: float
    true_error: float  # km from the true source (nan if diverged)
    grid_error: float  # km from the grid minimum (nan if diverged)
    status: str  # "ok", "diverged", "left_map" or "not_converged"


def style(look: str = "boxed", dark: bool = False) -> list[str]:
    """Style files in the required order: base, optional dark, one axes look."""
    paths = [str(_STYLES / "ml_seis.mplstyle")]
    if dark:
        paths.append(str(_STYLES / "ml_seis_dark.mplstyle"))
    paths.append(str(_STYLES / f"ml_seis_{look}.mplstyle"))
    return paths


def _c(colour):
    """Resolve a ``"C0"``-style colour now, inside the active style context.

    Matplotlib resolves such strings at *draw* time, which happens after the style
    context has closed and would silently use the default colour cycle.
    """
    from matplotlib.colors import to_hex

    return to_hex(colour)


def _count(n, noun):
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _rgb(colour):
    from matplotlib.colors import to_rgb

    return to_rgb(colour)


def map_axes(ax, box: float = BOX):
    """Square km map: equal aspect, fixed limits, labelled axes."""
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def plot_stations(ax, xi_x, xi_y, **extra):
    """Stations as triangles."""
    kwargs = {
        "marker": "^",
        "color": _c("C6"),
        "s": 22,
        "zorder": 3,
        "edgecolors": "none",
    }
    kwargs |= extra
    return ax.scatter(xi_x, xi_y, **kwargs)


def plot_misfit(ax, grid_x, grid_y, S, levels: int = 14):
    """Misfit map: filled log-spaced contours of S on a (ny, nx) grid."""
    import numpy as np

    logS = np.log10(np.maximum(S, 1e-6))
    lo, hi = float(logS.min()), float(np.percentile(logS, 98))
    return ax.contourf(
        grid_x, grid_y, logS, levels=np.linspace(lo, hi, levels), cmap="Blues_r",
        extend="max",
    )  # fmt: skip


def plot_misfit_map(ax, grid_x, grid_y, S, xi_x, xi_y, m_true=None, path=None):
    """Misfit map with stations, the grid minimum, the true source and an optimiser
    path (rows of (m_x, m_y)). ``S`` has shape ``grid_x.shape``."""
    import numpy as np

    plot_misfit(ax, grid_x, grid_y, S)
    plot_stations(ax, xi_x, xi_y, s=10, alpha=0.6)
    k = int(np.argmin(S))
    ax.plot(
        grid_x.flatten()[k], grid_y.flatten()[k], "x", color=_c("C2"), ms=8, mew=2,
        zorder=5, label="grid minimum",
    )  # fmt: skip
    if m_true is not None:
        ax.plot(*m_true, "*", color="k", ms=11, zorder=5, label="true source")
    if path is not None:
        path = np.asarray(path)
        ax.plot(*path.T, "-o", color=_c("C1"), ms=2.5, lw=1.2, zorder=4, label="path")
        ax.plot(*path[0], "o", color=_c("C1"), mfc="w", ms=7, zorder=6, label="start")
        ax.plot(*path[-1], "o", color=_c("C1"), ms=7, zorder=6)
    map_axes(ax)


def _draw_prior(ax, prior, events, color, xi_x, xi_y):
    import torch

    from .priors import log_prior

    nodes = torch.linspace(0, BOX, 201, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(nodes, nodes, indexing="xy")
    density = log_prior(prior, grid_x, grid_y).exp()
    # square root keeps the weak priors visible; one scale for both priors
    field = density.sqrt()
    if prior == "uniform":  # flat: mid-range blue, so the events stand out on it
        field = torch.where(density > 0, 0.4 * 0.055, 0.0)
    ax.contourf(
        grid_x, grid_y, field, levels=torch.linspace(0, 0.055, 12),
        cmap="Blues", extend="max",
    )  # fmt: skip
    plot_stations(ax, xi_x, xi_y, s=70)
    # a thin white rim keeps events visible on the dark parts of the prior
    ax.scatter(*events.T, s=8, color=color, alpha=0.9, edgecolors="white", lw=0.3)
    map_axes(ax)


def prior_figure(train_prior, test_prior, m_train, m_test, xi_x, xi_y, titles=None):
    """Two maps: the training prior with the training events, and the test prior
    with the test events. Events are rows of (m_x, m_y); darker blue = more likely.
    ``titles`` replaces the two default titles (train / test)."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("boxed")):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
        _draw_prior(ax_a, train_prior, m_train, _c("C0"), xi_x, xi_y)
        ax_a.set_title(
            titles[0]
            if titles
            else f"train: {len(m_train)} events from the {train_prior} prior"
        )
        _draw_prior(ax_b, test_prior, m_test, _c("C1"), xi_x, xi_y)
        ax_b.set_title(
            titles[1]
            if titles
            else f"test: {len(m_test)} events from the {test_prior} prior"
        )
    plt.close(fig)
    return fig


def mlp_structure_figure(
    net,
    in_labels=None,
    out_labels=None,
    activation=None,
    max_nodes=None,
    trained=True,
    caption=None,
):
    """An MLP as a graph plus its weight matrices: one static figure, no training.

    ``net`` is a ``nn.Sequential`` of ``Linear`` layers with the activation
    modules between them; ``in_labels`` and ``out_labels`` name the input and
    output values (matplotlib mathtext). Same look as the Lecture 1
    ``MLPPlayground``: neurons are tiles (hidden ones sketch the activation),
    edges and matrix cells share the weight colours (blue positive, vermillion
    negative, strength the magnitude), and each matrix block sits under the
    band of edges it holds. Numbering runs top-down in graph and matrices alike:
    edge i -> j is cell (j, i).

    ``max_nodes`` draws at most that many neurons per layer (the last one is kept,
    a "\u22ee" stands for the rest) and weight matrices as images, so any width
    stays fast; edges are drawn between the neurons shown. ``trained=False``
    greys the weights (initial values). ``caption`` is a line of text under the
    figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch, Rectangle

    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    if not linears:
        raise ValueError("mlp_structure_figure needs at least one Linear layer")
    dims = [linears[0].in_features] + [lin.out_features for lin in linears]
    if in_labels is not None and len(in_labels) != dims[0]:
        raise ValueError(f"expected {dims[0]} in_labels, got {len(in_labels)}")
    if out_labels is not None and len(out_labels) != dims[-1]:
        raise ValueError(f"expected {dims[-1]} out_labels, got {len(out_labels)}")
    in_labels = list(in_labels or [f"$x_{{{i + 1}}}$" for i in range(dims[0])])
    out_labels = list(out_labels or [f"$y_{{{i + 1}}}$" for i in range(dims[-1])])
    if activation is None:
        acts = {
            torch.nn.ReLU: "ReLU",
            torch.nn.Tanh: "tanh",
            torch.nn.Sigmoid: "sigmoid",
            torch.nn.GELU: "gelu",
        }
        activation = next(
            (v for m in net for k, v in acts.items() if isinstance(m, k)), "linear"
        )
    fn = {
        "ReLU": torch.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "gelu": torch.nn.functional.gelu,
        "linear": torch.nn.Identity(),
    }[activation]
    sign = {True: TOKENS["blue"], False: TOKENS["vermillion"]}
    ink, ink2, grid = TOKENS["ink"], TOKENS["ink2"], TOKENS["grid"]
    n_cols = len(dims)

    def shown(n):  # the neurons drawn in a layer of n, and where the gap is
        if max_nodes is None or n <= max_nodes:
            return list(range(n)), -1
        return [*range(max_nodes - 1), n - 1], max_nodes - 1

    drawn = [shown(n) for n in dims]
    slots = [len(idx) + (gap >= 0) for idx, gap in drawn]  # vertical positions
    fade = 1.0
    if not trained:  # initial weights: grey and light
        sign, fade = {True: ink2, False: ink2}, 0.45

    # Layout in figure units, 100 wide (aspect 1). Every neuron is a tile of one
    # size at one pitch: a wide layer makes the graph, and the figure, taller
    # (by ``grow``) instead of shrinking the tiles. The matrices row stays put.
    tile, pitch = 2.2, 3.0
    xs = np.linspace(14, 86, n_cols)
    span = max(24.0, (max(slots) - 1) * pitch)
    grow = span - 24.0
    height = 51.0 + grow + (3.0 if caption else 0.0)
    low = 3.0 if caption else 0.0  # room for the caption under the matrices
    cy, bracket_y, top = 20.6 + low + span / 2, 18.8 + low, 15.6 + low

    def node_y(c, k):  # slot k of column c, slot 0 at the top
        return cy + ((slots[c] - 1) / 2 - k) * pitch

    def slot_of(c, i):  # the slot of neuron i, if it is drawn
        idx, gap = drawn[c]
        k = idx.index(i)
        return k + (gap >= 0 and k >= gap)

    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(9.0, 4.6 * height / 51), layout="constrained")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # ---- edges: colour the sign, width and strength |w| ------------------
        for c, lin in enumerate(linears):
            w = torch.nan_to_num(lin.weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            k = float(w.abs().max()) or 1.0
            segs, cols, lws = [], [], []
            for i in drawn[c][0]:
                for j in drawn[c + 1][0]:
                    a = float(w[j, i].abs()) / k
                    segs.append(
                        [
                            (xs[c], node_y(c, slot_of(c, i))),
                            (xs[c + 1], node_y(c + 1, slot_of(c + 1, j))),
                        ]
                    )
                    cols.append(
                        (*_rgb(sign[bool(w[j, i] >= 0)]), fade * (0.15 + 0.7 * a))
                    )
                    lws.append(0.3 + 1.9 * a)
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws))

        # ---- tiles: hidden ones sketch the activation over z in [-3, 3] ------
        # (a few collections, not an artist per tile: this is redrawn on every change)
        z = torch.linspace(-3, 3, 41).numpy()
        act = fn(torch.from_numpy(z)).numpy()
        act = act / max(float(np.abs(act).max()), 1e-6)
        boxes, above, below, curves = [], [], [], []
        for c in range(n_cols):
            idx, gap = drawn[c]
            if gap >= 0:
                ax.text(xs[c], node_y(c, gap), "\u22ee", ha="center", va="center",
                        color=ink2, fontsize=11)  # fmt: skip
            for i in idx:
                x, y = xs[c], node_y(c, slot_of(c, i))
                boxes.append(
                    FancyBboxPatch(
                        (x - tile / 2, y - tile / 2), tile, tile,
                        boxstyle=f"round,pad=0,rounding_size={0.12 * tile}",
                    )
                )  # fmt: skip
                if 0 < c < n_cols - 1:
                    px = x - tile / 2 + 0.1 * tile + 0.8 * tile * (z + 3) / 6
                    for part, polys in (
                        (np.maximum(act, 0), above),
                        (np.minimum(act, 0), below),
                    ):
                        py = y + part * 0.38 * tile
                        polys.append(
                            np.c_[np.r_[px, px[::-1]], np.r_[py, np.full_like(py, y)]]
                        )
                    curves.append(np.c_[px, y + act * 0.38 * tile])
        tiles_style = {"facecolor": TOKENS["bg"], "edgecolor": ink2, "linewidth": 0.7}
        ax.add_collection(PatchCollection(boxes, zorder=3, **tiles_style))
        for polys, colour in ((above, "blue"), (below, "vermillion")):
            fill = {"facecolor": TOKENS[colour], "alpha": 0.4, "linewidth": 0}
            ax.add_collection(PolyCollection(polys, zorder=4, **fill))
        ax.add_collection(LineCollection(curves, colors=ink, linewidths=0.7, zorder=5))
        for i in drawn[0][0]:
            ax.text(xs[0] - tile / 2 - 1.0, node_y(0, slot_of(0, i)), in_labels[i],
                    ha="right", va="center")  # fmt: skip
        for i in drawn[-1][0]:
            ax.text(xs[-1] + tile / 2 + 1.0, node_y(n_cols - 1, slot_of(n_cols - 1, i)),
                    out_labels[i], ha="left", va="center")  # fmt: skip

        # ---- column heads ------------------------------------------------------
        for c in range(n_cols):
            if c == 0:
                head, sub = "INPUT", _count(dims[c], "value")
            elif c == n_cols - 1:
                head, sub = "OUTPUT", _count(dims[c], "value")
            else:
                head = "HIDDEN" if n_cols == 3 else f"HIDDEN {c}"
                sub = f"{_count(dims[c], 'neuron')}, {activation}"
            ax.text(xs[c], height - 0.8, head, ha="center", va="top", color=ink2,
                    fontsize=8)  # fmt: skip
            ax.text(xs[c], height - 3.1, sub, ha="center", va="top", color=ink2,
                    fontsize=8)  # fmt: skip

        # ---- the same network as matrices, each under its band of edges --------
        for li, lin in enumerate(linears):
            a, b = xs[li] + 1.6, xs[li + 1] - 1.6
            ax.plot([a, a, b, b], [bracket_y + 0.9, bracket_y, bracket_y,
                    bracket_y + 0.9], color=grid, lw=1.0)  # fmt: skip
            n_out, n_in = lin.out_features, lin.in_features
            cell = min(1.6, 12.4 / n_out, (b - a - 3) / (n_in + 1.6))
            width = n_in * cell + 0.6 * cell + cell
            cx = (xs[li] + xs[li + 1]) / 2
            x0 = cx - width / 2
            bx = x0 + n_in * cell + 0.6 * cell
            w = torch.nan_to_num(lin.weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            bias = torch.nan_to_num(lin.bias.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            k_w = float(w.abs().max()) or 1.0
            k_b = float(bias.abs().max()) or 1.0
            for block, k, x_left, n_cols_b in (
                (w, k_w, x0, n_in),
                (bias[:, None], k_b, bx, 1),
            ):  # one image per block instead of a patch per cell
                rgba = np.zeros((*block.shape, 4))
                for pos, colour in ((block >= 0, sign[True]), (block < 0, sign[False])):
                    rgba[pos.numpy(), :3] = _rgb(colour)
                strength = block.abs().numpy() / k
                rgba[..., 3] = fade * np.minimum(0.1 + 0.9 * strength, 1.0)
                ax.imshow(
                    rgba,
                    extent=(x_left, x_left + n_cols_b * cell, top - n_out * cell, top),
                    interpolation="nearest", aspect="auto", zorder=2,
                )  # fmt: skip
                if n_out * n_in <= 600:  # white lines between the cells
                    ys = top - cell * np.arange(n_out + 1)
                    xg = x_left + cell * np.arange(n_cols_b + 1)
                    lines = [[(xg[0], v), (xg[-1], v)] for v in ys]
                    lines += [[(v, ys[0]), (v, ys[-1])] for v in xg]
                    ax.add_collection(
                        LineCollection(lines, colors="white", linewidths=0.4, zorder=3)
                    )  # fmt: skip
            ax.text(cx, top + 0.5,
                    rf"$\mathbf{{W}}^{{({li + 1})}}\;\;\mathbf{{b}}^{{({li + 1})}}$",
                    ha="center", va="bottom")  # fmt: skip
            ax.text(cx, top - n_out * cell - 0.8,
                    f"{n_out}\u00d7{n_in} + {n_out} = {n_out * n_in + n_out}",
                    ha="center", va="top", color=ink2, fontsize=8)  # fmt: skip

        # ---- weight key (left) and parameter count (right) ---------------------
        key = LinearSegmentedColormap.from_list(
            "w", [TOKENS["vermillion"], "#ffffff", TOKENS["blue"]]
        )
        ax.text(0, 12.2, "WEIGHT", color=ink2, fontsize=8, va="bottom")
        ax.imshow(np.linspace(0, 1, 64)[None, :], cmap=key, aspect="auto",
                  extent=(0, 12, 9.8, 11.0))  # fmt: skip
        ax.add_patch(Rectangle((0, 9.8), 12, 1.2, fill=False, ec=grid, lw=0.6))
        for x, t in ((0, "\u2212"), (6, "0"), (12, "+")):
            ax.text(x, 9.2, t, ha="center", va="top", fontsize=9)
        total = sum(lin.weight.numel() + lin.bias.numel() for lin in linears)
        ax.text(100, 12.2, "PARAMETERS", color=ink2, fontsize=8, ha="right",
                va="bottom")  # fmt: skip
        ax.text(100, 11.0, f"{total:,}", fontsize=10, ha="right", va="top")
        if caption:
            ax.text(50, 0.4, caption, ha="center", va="bottom", color=ink2,
                    fontsize=9)  # fmt: skip
        ax.set_xlim(0, 100)
        ax.set_ylim(0, height)
    plt.close(fig)
    return fig


def evaluate_run(problem, path, full_loss, loss_fn, grid_n: int = 101) -> RunSummary:
    """Evaluate the full misfit landscape and summarise one optimiser run.

    ``loss_fn`` is the loss defined in the notebook. This helper only evaluates it on
    a grid; it never defines the objective or performs an optimisation step. The
    figure itself (maps, path, loss curves) is the ``MapEditor`` widget.
    """
    import torch

    path = torch.as_tensor(path).detach()
    full_loss = torch.as_tensor(full_loss).detach()

    with torch.no_grad():
        nodes = torch.linspace(0, problem.box, grid_n, dtype=torch.float64)
        grid_x, grid_y = torch.meshgrid(nodes, nodes, indexing="xy")
        models = torch.stack([grid_x, grid_y], dim=-1)
        S_grid = loss_fn(models)

    k = int(torch.argmin(S_grid))
    m_grid = torch.stack([grid_x.flatten()[k], grid_y.flatten()[k]])
    m_end = path[-1]

    finite = bool(torch.isfinite(path).all()) and bool(torch.isfinite(full_loss).all())
    if not finite:
        status, true_error, grid_error = "diverged", float("nan"), float("nan")
    else:
        true_error = torch.dist(m_end, problem.m_true).item()
        grid_error = torch.dist(m_end, m_grid).item()
        if bool(((path < 0) | (path > problem.box)).any()):
            status = "left_map"
        elif full_loss[-1] > 3 * S_grid.min() + 0.1:
            status = "not_converged"
        else:
            status = "ok"

    return RunSummary(
        S_grid=S_grid,
        m_grid=m_grid,
        S_grid_min=float(S_grid.min()),
        S_final=float(full_loss[-1]),
        true_error=true_error,
        grid_error=grid_error,
        status=status,
    )
