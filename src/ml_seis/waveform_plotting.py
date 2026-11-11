"""Every matplotlib figure of Part 2 (shared course style, see ``styles/``)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_hex

from .plotting import style

CLASS_COLOURS = {"LP": "C0", "TR": "C2", "VT": "C1", "TC": "C6"}
COMPONENT_COLOURS = ("C4", "C5", "C0")
PHASE_COLOURS = {"P": "C1", "S": "C0", "neither": "C6"}


def _c(colour):
    """Resolve a ``"C0"``-style colour now, inside the active style context.

    Matplotlib resolves such strings at *draw* time, which happens after the style
    context has closed and would silently use the default colour cycle.
    """
    return to_hex(colour)


def _numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _done(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)
    return fig


def _colour(name, classes=None):
    return _c(CLASS_COLOURS.get(name, "C3"))


# ------------------------------------------------------------------- Llaima data


def plot_class_balance(data):
    """Records per class, split into train, validation and test."""
    import matplotlib.pyplot as plt

    names = data.classes
    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        bottom = np.zeros(len(names))
        for k, split in enumerate(("train", "validation", "test")):
            counts = _numpy(data.counts(split))
            ax.bar(
                names,
                counts,
                bottom=bottom,
                color=_c(f"C{[0, 3, 1][k]}"),
                label=split,
                alpha=[1.0, 0.85, 0.85][k],
            )
            bottom += counts
        for k, total in enumerate(bottom):
            ax.text(k, total + 15, f"{int(total)}", ha="center", fontsize=8)
        ax.set_ylabel("records")
        ax.set_ylim(0, bottom.max() * 1.12)
        ax.legend(loc="upper center", ncol=3)
    return _done(fig)


def plot_sigmoid_softmax():
    """One logit and the sigmoid next to two logits and the softmax.

    The probability of class 1 depends on the two logits only through ``a_1 - a_2``;
    the sigmoid is the cut of the softmax along ``a_2 = 0``."""
    import matplotlib.pyplot as plt

    a = np.linspace(-6, 6, 241)
    grid = np.linspace(-4, 4, 161)
    a_1, a_2 = np.meshgrid(grid, grid)
    p_1 = 1 / (1 + np.exp(a_2 - a_1))
    with plt.style.context(style("open")):
        cut = _c("C1")
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(7.5, 3.0), gridspec_kw={"width_ratios": [1.2, 1]}
        )
        left.plot(a, 1 / (1 + np.exp(-a)), color=cut)
        left.plot([0], [0.5], "o", color=cut)
        left.set_yticks([0, 0.5, 1])
        left.set_xlabel("logit $a$")
        left.set_ylabel("probability $p$")
        left.set_title("one logit: sigmoid  $p = 1/(1 + e^{-a})$")
        shades = right.contourf(
            a_1, a_2, p_1, levels=np.linspace(0, 1, 11), cmap="Blues"
        )
        right.contour(
            a_1, a_2, p_1, levels=[0.5], colors="k", linewidths=0.8, linestyles="--"
        )
        right.axhline(0, color=cut, lw=1.6)
        right.text(
            -3.8, 0.25, "$a_2 = 0$", color=cut, ha="left", va="bottom", fontsize=9
        )
        right.set_aspect("equal")
        right.set_xlabel("logit $a_1$")
        right.set_ylabel("logit $a_2$")
        right.set_title("two logits: softmax  $p_1$")
        fig.colorbar(shades, ax=right, ticks=[0, 0.5, 1], shrink=0.8)
    return _done(fig)


def plot_augmentations(
    u,
    sampling_rate: float,
    max_shift: int = 50,
    seconds: float = 8.0,
    name: str = "",
    seed: int = 0,
):
    """The record ``u`` (T,) as the network sees it: unchanged, shifted, flipped.

    Three rows: the original; the window moved by a random number of samples between
    0 and ``max_shift`` (drawn from ``seed``, uniform like the training), so the signal
    arrives earlier in it; the polarity reversed. The altered rows show the original
    faintly behind them. ``name`` (the class) goes in the title of the first row."""
    import matplotlib.pyplot as plt

    u = _numpy(u)
    n = int(seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    original = u[:n]
    shift = int(np.random.default_rng(seed).integers(0, max_shift + 1))
    later = f"shift: the window starts {shift / sampling_rate:.2f} s later (random)"
    rows = [
        (original, f"original {name}".strip()),
        (u[shift : shift + n], later),
        (-original, "flip: the polarity is reversed"),
    ]
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 3.8), sharex=True, sharey=True)
        for k, (ax, (curve, title)) in enumerate(zip(axes, rows, strict=True)):
            if k:  # the original, faint, behind the altered record
                ax.plot(t, original, color=_c("C6"), lw=1.0, alpha=0.35)
            ax.plot(t, curve, color=_c("C6" if k == 0 else "C1"), lw=1.0)
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_yticks([])
        axes[-1].set_xlabel("time in the window (s)")
    return _done(fig)


# -------------------------------------------------------------- CNN explanations


def plot_filter_demo(t, x, kernel, response):
    """A record, one filter, and the feature map that sliding the filter produces."""
    import matplotlib.pyplot as plt

    t, x, kernel, response = map(_numpy, (t, x, kernel, response))
    with plt.style.context(style("open")):
        fig = plt.figure(figsize=(9.0, 3.8))
        grid = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.35)
        ax_x = fig.add_subplot(grid[0, 0])
        ax_y = fig.add_subplot(grid[1, 0], sharex=ax_x)
        ax_k = fig.add_subplot(grid[:, 1])
        ax_x.plot(t, x, color=_c("C6"), lw=0.8)
        ax_x.set_ylabel("input")
        ax_y.plot(t[: len(response)], response, color=_c("C1"), lw=0.8)
        ax_y.set_ylabel("feature map")
        ax_y.set_xlabel("time (s)")
        ax_k.stem(np.arange(len(kernel)), kernel, basefmt=" ")
        ax_k.set_title("filter weights")
        ax_k.set_xlabel("sample")
    return _done(fig)


def plot_filter_bank(t, u, maps, names):
    """One record above the feature maps of several filters, one row per filter.

    ``maps`` has shape ``(filters, samples)``: the filters are the channels of the
    next layer."""
    import matplotlib.pyplot as plt

    t, u, maps = map(_numpy, (t, u, maps))
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            1 + len(maps),
            1,
            figsize=(9.0, 0.7 + 0.75 * (1 + len(maps))),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        axes[0].plot(t, u, color=_c("C6"), lw=0.8)
        axes[0].set_ylabel("record")
        for ax, h, name in zip(axes[1:], maps, names, strict=True):
            ax.plot(t[: len(h)], h, color=_c("C1"), lw=0.8)
            ax.set_ylabel(name)
        for ax in axes:
            ax.set_yticks([])
        axes[-1].set_xlabel("time (s)")
    return _done(fig)


def plot_arc():
    """The route of Part 2: one record, one network family, three kinds of answer."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(9.0, 3.4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 40)
        ax.axis("off")
        ink2 = _c("C6")

        def box(x, y, w, h, text, colour):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                    fc="none", ec=colour, lw=1.2,
                )
            )  # fmt: skip
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

        def arrow(x0, y0, x1, y1):
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                    color=ink2, lw=1.0, shrinkA=0, shrinkB=0,
                )
            )  # fmt: skip

        box(1, 15, 17, 10, "seismogram\n$\\mathbf{u}$", _c("C6"))
        box(26, 15, 20, 10, "convolutional\nnetwork", _c("C0"))
        arrow(18, 20, 26, 20)
        answers = [
            (31, "one class per record\n(2b)", _c("C2")),
            (17.5, "a code, no labels\n(2c)", _c("C4")),
            (4, "P and S at every sample\n(2d)", _c("C1")),
        ]
        for y, text, colour in answers:
            box(53, y - 1.5, 24, 9, text, colour)
            arrow(46, 20, 53, y + 3)
        box(83, 2.5, 16, 9, "picks, then\n$\\mathbf{m}$ (Part 1)", _c("C1"))
        arrow(77, 7.5, 83, 7)
    return _done(fig)


def plot_activations(x, activations, names, sampling_rate: float, title: str = ""):
    """The input record above the feature maps after successive layers.

    Every channel of a feature map gets the same height, so a panel is as tall as the
    block has filters. Faint lines every 2 s run through all panels, faint ones
    between the channels of a map, to compare when the blocks react. ``title`` is
    written above the record."""
    import matplotlib.pyplot as plt

    x = _numpy(x)
    maps = [_numpy(a) for a in activations]
    channels = [m.shape[0] for m in maps]
    seconds = len(x) / sampling_rate
    ticks = np.arange(0, seconds + 1e-9, 2.0)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            1 + len(maps),
            1,
            figsize=(12.5, 1.4 + 0.0235 * sum(channels)),
            gridspec_kw={"height_ratios": [40] + channels},
        )
        axes[0].plot(np.arange(len(x)) / sampling_rate, x, color=_c("C6"), lw=0.7)
        axes[0].set_ylabel("input", rotation=0, ha="right", va="center")
        axes[0].grid(True, axis="x", lw=0.5)
        if title:
            axes[0].set_title(title, loc="left")
        for ax, m, name in zip(axes[1:], maps, names, strict=True):
            ax.imshow(
                m,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                extent=(0, seconds, m.shape[0], 0),
            )
            ax.hlines(
                np.arange(1, m.shape[0]), 0, seconds, color="w", lw=0.3, alpha=0.2
            )
            ax.vlines(ticks, 0, m.shape[0], color="w", lw=0.5, alpha=0.3)
            ax.set_ylabel(
                f"{name}\n{m.shape[0]} filters", rotation=0, ha="right", va="center"
            )
            ax.set_yticks([])
        for ax in axes:
            ax.set_xlim(0, seconds)
            ax.set_xticks(ticks)
        axes[-1].set_xlabel("time (s)")
        for ax in axes[:-1]:
            ax.set_xticklabels([])
    return _done(fig)


def plot_filters(weights, sampling_rate: float, max_filters: int = 16):
    """First-layer convolution kernels, one small panel each."""
    import matplotlib.pyplot as plt

    w = _numpy(weights)[:max_filters, 0]
    columns = min(8, len(w))
    rows = int(np.ceil(len(w) / columns))
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            rows, columns, figsize=(9.0, 1.1 * rows + 0.4), squeeze=False, sharey=True
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, k in zip(axes.flat, w, strict=False):
            ax.axis("on")
            ax.plot(np.arange(len(k)) / sampling_rate, k, color=_c("C1"))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("first-layer filters", fontsize=10)
    return _done(fig)


# ----------------------------------------------------------------- training views


def plot_history(history: dict, ylabel: str = "loss", logy: bool = False):
    """Training curves; one line per key of ``history``."""
    import matplotlib.pyplot as plt

    colours = {"train": _c("C0"), "validation": _c("C1")}
    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for name, values in history.items():
            ax.plot(
                np.arange(1, len(values) + 1),
                values,
                color=colours.get(name),
                marker="o",
                ms=3,
                label=name,
            )
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
    return _done(fig)


def plot_live_length(n_live, label, classes, sampling_rate: float, window: int):
    """How long the signal lasts in each class: the share of the records whose signal
    (seconds before the flat padding) is shorter than the abscissa. The window the
    network gets is marked."""
    import matplotlib.pyplot as plt

    n_live, label = _numpy(n_live), _numpy(label)
    seconds = n_live / sampling_rate
    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for k, name in enumerate(classes):
            s = np.sort(seconds[label == k])
            share = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, share, where="post", lw=1.6, color=_colour(name), label=name)
        ax.axvline(window / sampling_rate, color="k", lw=0.8, ls="--")
        ax.text(window / sampling_rate + 0.8, 0.55, "window", fontsize=8)
        ax.set_xlabel("signal before the flat padding (s)")
        ax.set_ylabel("share of records shorter")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left")
    return _done(fig)


def plot_history_comparison(histories: dict):
    """Training (dashed) and validation (solid) loss of several runs, one colour per
    run, ``histories`` mapping a run name to its ``{"train", "validation"}`` curves."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        for k, (name, history) in enumerate(histories.items()):
            colour = _c(f"C{k}")
            epochs = np.arange(1, len(history["validation"]) + 1)
            ax.plot(
                epochs, history["validation"], color=colour, label=f"{name}, validation"
            )
            ax.plot(
                epochs,
                history["train"],
                color=colour,
                ls="--",
                label=f"{name}, training",
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
    return _done(fig)


def plot_recall_comparison(names, recalls: dict):
    """Recall per class for several versions of the test records, as grouped bars."""
    import matplotlib.pyplot as plt

    n = len(recalls)
    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        positions = np.arange(len(names))
        for k, (label, recall) in enumerate(recalls.items()):
            ax.bar(
                positions + (k - (n - 1) / 2) * 0.8 / n,
                recall,
                0.8 / n,
                color=_c(f"C{[0, 1, 2][k]}"),
                label=label,
            )
        ax.set_xticks(positions, names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("recall")
        ax.legend(loc="lower right")
    return _done(fig)


def plot_confusion(cm, names, report, highlight: int | None = None):
    """Row-normalised confusion matrix next to per-class recall and precision.

    ``highlight`` is a class index: its row (recall) and its column (precision) are
    outlined in the colours of the bars. The legend sits above the bars, not in them."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cm = np.asarray(cm)
    shares = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(names)
    with plt.style.context(style("open")):
        fig, (ax, bars) = plt.subplots(
            1, 2, figsize=(8.0, 3.2), gridspec_kw={"width_ratios": [1, 1.1]}
        )
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if shares[i, j] > 0.5 else "black",
                )
        if highlight is not None:
            k = highlight
            for xy, w, h, colour in [
                ((-0.5, k - 0.5), n, 1, _c("C0")),  # the row: recall
                ((k - 0.5, -0.5), 1, n, _c("C1")),  # the column: precision
            ]:
                ax.add_patch(
                    Rectangle(xy, w, h, fill=False, ec=colour, lw=2.2, clip_on=False)
                )
        ax.set_xticks(range(n), names)
        ax.set_yticks(range(n), names)
        ax.set_xlabel("prediction")
        ax.set_ylabel("label")
        ax.grid(False)
        positions = np.arange(n)
        bars.bar(positions - 0.2, report["recall"], 0.4, color=_c("C0"), label="recall")
        bars.bar(
            positions + 0.2, report["precision"], 0.4, color=_c("C1"), label="precision"
        )
        bars.axhline(
            report["accuracy"],
            color="k",
            lw=0.8,
            ls="--",
            label=f"accuracy {report['accuracy']:.2f}",
        )
        bars.set_xticks(positions, names)
        bars.set_ylim(0, 1.02)
        bars.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.2
        )
    return _done(fig)


def plot_probability_gallery(x, probability, label, names, indices, sampling_rate):
    """Waveform and predicted class probabilities for the chosen test records."""
    import matplotlib.pyplot as plt

    x, probability, label = _numpy(x), _numpy(probability), _numpy(label)
    if len(indices) == 0:  # no mistakes: show the first records instead
        indices = range(min(4, len(x)))
    k = len(indices)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            2,
            k,
            figsize=(9.0, 3.6),
            gridspec_kw={"height_ratios": [1.2, 1]},
            squeeze=False,
        )
        for col, i in enumerate(indices):
            guess = int(probability[i].argmax())
            axes[0, col].plot(
                np.arange(x.shape[-1]) / sampling_rate, x[i, 0], color=_c("C6"), lw=0.6
            )
            axes[0, col].set_ylim(-1.1, 1.1)
            axes[0, col].set_yticks([])
            axes[0, col].set_title(f"label: {names[label[i]]}", fontsize=9)
            colours = [
                _c("C2") if c == label[i] else _c("C6") for c in range(len(names))
            ]
            axes[1, col].bar(names, probability[i], color=colours)
            axes[1, col].set_ylim(0, 1)
            axes[1, col].set_xlabel(
                f"model: {names[guess]} ({probability[i].max():.2f})", fontsize=8
            )
            if col:
                axes[1, col].set_yticklabels([])
        axes[1, 0].set_ylabel("probability")
        axes[0, 0].set_xlabel("")
    return _done(fig)


def plot_embedding(coords, label, classes, title="embedding, first two principal axes"):
    """Scatter of 2-D coordinates coloured by the label."""
    import matplotlib.pyplot as plt

    coords, label = _numpy(coords), _numpy(label)
    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for k, name in enumerate(classes):
            m = label == k
            ax.scatter(
                *coords[m].T,
                s=8,
                alpha=0.7,
                color=_colour(name),
                label=name,
                edgecolors="none",
            )
        ax.set_xlabel("axis 1")
        ax.set_ylabel("axis 2")
        ax.set_title(title)
        ax.legend(markerscale=2)
    return _done(fig)


def plot_cluster_embedding(coords, recovered, label, classes, centres):
    """Same 2-D map twice: classes recovered by clustering, then analyst classes.

    ``coords`` are the first two principal coordinates of the code; ``centres`` are
    the cluster centres in the same two coordinates, drawn as their index.
    """
    import matplotlib.pyplot as plt

    coords, centres = _numpy(coords), _numpy(centres)
    recovered, label = _numpy(recovered), _numpy(label)
    panels = (
        (recovered, "recovered by clustering the code"),
        (label, "analyst label (only for checking)"),
    )
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True)
        for ax, (colour_by, title) in zip(axes, panels, strict=True):
            for k, name in enumerate(classes):
                m = colour_by == k
                ax.scatter(
                    *coords[m].T,
                    s=8,
                    alpha=0.7,
                    color=_colour(name),
                    label=name,
                    edgecolors="none",
                )
            ax.set_title(title)
            ax.set_xlabel("axis 1")
        for j, (cx, cy) in enumerate(centres):
            axes[0].text(cx, cy, str(j), ha="center", va="center", fontsize=7)
        axes[0].set_ylabel("axis 2")
        axes[1].legend(markerscale=2)
    return _done(fig)


# --------------------------------------------------------- autoencoder / U-Net


def _spectrogram_panel(ax, spec, hop, sampling_rate):
    """Draw one log-spectrogram panel with time and frequency axes."""
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(0, spec.shape[-1] * hop / sampling_rate, 0, sampling_rate / 2),
    )


def plot_spectrogram_row(spec, classes, indices, hop, sampling_rate):
    """One spectrogram per class in a row: what the autoencoder sees and rebuilds."""
    import matplotlib.pyplot as plt

    spec = _numpy(spec)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            1, len(indices), figsize=(9.0, 2.6), sharex=True, sharey=True
        )
        for ax, name, i in zip(np.atleast_1d(axes), classes, indices, strict=True):
            _spectrogram_panel(ax, spec[i], hop, sampling_rate)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("time (s)")
        np.atleast_1d(axes)[0].set_ylabel("frequency (Hz)")
    return _done(fig)


def plot_reconstruction_grid(spec, spec_hat, classes, indices, hop, sampling_rate):
    """Original spectrograms (top row) and what the network rebuilds (bottom row)."""
    import matplotlib.pyplot as plt

    spec, spec_hat = _numpy(spec), _numpy(spec_hat)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            2, len(indices), figsize=(9.0, 3.8), sharex=True, sharey=True, squeeze=False
        )
        for col, (name, i) in enumerate(zip(classes, indices, strict=True)):
            _spectrogram_panel(axes[0, col], spec[i], hop, sampling_rate)
            _spectrogram_panel(axes[1, col], spec_hat[i], hop, sampling_rate)
            axes[0, col].set_title(name, fontsize=9)
            axes[1, col].set_xlabel("time (s)")
        axes[0, 0].set_ylabel("original\nfrequency (Hz)")
        axes[1, 0].set_ylabel("rebuilt from z\nfrequency (Hz)")
    return _done(fig)


def plot_interpolation(specs, titles, hop, sampling_rate, columns: int = 4):
    """Spectrograms in a grid, row by row: two records and what lies between them."""
    import matplotlib.pyplot as plt

    specs = _numpy(specs)
    rows = -(-len(specs) // columns)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            rows, columns, figsize=(9.0, 1.9 * rows), sharex=True, sharey=True,
            squeeze=False,
        )  # fmt: skip
        for ax in axes.flat:
            ax.axis("off")
        for ax, spec, title in zip(axes.flat, specs, titles, strict=False):
            ax.axis("on")
            _spectrogram_panel(ax, spec, hop, sampling_rate)
            ax.set_title(title, fontsize=9)
        for row in range(rows):
            axes[row, 0].set_ylabel("frequency (Hz)")
        for col in range(columns):
            axes[-1, col].set_xlabel("time (s)")
    return _done(fig)


def plot_bars(labels, values, xlabel: str, baseline: float | None = None):
    """Horizontal bars of shares in [0, 1]; ``baseline`` draws a dashed line."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(6.0, 0.9 + 0.45 * len(values)))
        ax.barh(np.arange(len(values)), values, color=_c("C0"), height=0.6)
        ax.set_yticks(np.arange(len(values)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        if baseline is not None:
            ax.axvline(baseline, color=_c("C6"), ls="--", lw=1.0)
    return _done(fig)


def plot_autoencoder(input_numbers: int, code_numbers: int):
    """Encoder, code and decoder as a funnel; the loss compares input and output."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

    with plt.style.context(style("open")):
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 36)
        ax.axis("off")
        blue, ink2 = _c("C0"), _c("C6")

        def label(x, y, text, size=10):
            ax.text(x, y, text, ha="center", va="center", fontsize=size)

        ax.add_patch(Rectangle((2, 8), 12, 20, fc="none", ec=ink2, lw=1.2))
        label(8, 18, "spectrogram\n$\\mathbf{s}$")
        label(8, 31, f"{input_numbers:,} numbers", 9)
        label(92, 31, f"{input_numbers:,} numbers", 9)
        ax.add_patch(
            Polygon([(18, 6), (18, 30), (44, 21), (44, 15)], fc="none", ec=blue)
        )
        label(31, 18, "encoder")
        ax.add_patch(Rectangle((46, 14), 8, 8, fc="none", ec=_c("C1"), lw=1.6))
        label(50, 18, "$\\mathbf{z}$")
        label(50, 9.5, f"code: {code_numbers} numbers", 9)
        ax.add_patch(
            Polygon([(56, 15), (56, 21), (82, 30), (82, 6)], fc="none", ec=blue)
        )
        label(69, 18, "decoder")
        ax.add_patch(Rectangle((86, 8), 12, 20, fc="none", ec=ink2, lw=1.2))
        label(92, 18, "rebuilt\n$\\hat{\\mathbf{s}}$")
        for x0, x1 in [(14, 18), (44, 46), (54, 56), (82, 86)]:
            ax.add_patch(
                FancyArrowPatch(
                    (x0, 18), (x1, 18), arrowstyle="-|>", mutation_scale=10,
                    color=ink2, lw=1.0, shrinkA=0, shrinkB=0,
                )
            )  # fmt: skip
        ax.annotate(
            "", xy=(8, 3), xytext=(92, 3),
            arrowprops=dict(arrowstyle="<->", color=_c("C1"), lw=1.0),
        )  # fmt: skip
        label(50, 5.2, "loss: squared difference of the two", 9)
    return _done(fig)


def plot_class_prototypes(decoded, names, sizes, classes, hop, sampling_rate):
    """Decoded cluster centres averaged per recovered class name."""
    import matplotlib.pyplot as plt

    decoded, names, sizes = _numpy(decoded), _numpy(names), _numpy(sizes)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            1, len(classes), figsize=(9.0, 2.6), sharex=True, sharey=True
        )
        for ax, (index, name) in zip(
            np.atleast_1d(axes), enumerate(classes), strict=True
        ):
            members = names == index
            weight = sizes[members][:, None, None]
            prototype = (
                (decoded[members] * weight).sum(0) / weight.sum()
                if weight.sum() > 0
                else np.zeros_like(decoded[0])
            )
            _spectrogram_panel(ax, prototype, hop, sampling_rate)
            ax.set_title(f"{name} (n={int(sizes[members].sum())})", fontsize=9)
        np.atleast_1d(axes)[0].set_ylabel("frequency (Hz)")
        np.atleast_1d(axes)[0].set_yticks([0, sampling_rate / 4, sampling_rate / 2])
        np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    return _done(fig)


# ------------------------------------------------------------------- CEED / picks


def plot_ceed_overview(data):
    """Where the retained events and stations are, and how S-P time maps to distance."""
    import matplotlib.pyplot as plt

    split = _numpy(data.split)
    s_minus_p = _numpy(data.s - data.p) / 100.0
    with plt.style.context(style("boxed")):
        fig, (ax_map, ax_sp) = plt.subplots(
            1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.1, 1]}
        )
        stations = np.unique(data.station_lonlat, axis=0)
        ax_map.scatter(
            *stations.T, marker="^", s=18, color=_c("C6"), label="stations", zorder=2
        )
        for k, name in enumerate(("train", "validation", "test")):
            m = split == k
            ax_map.scatter(
                *data.event_lonlat[m].T,
                s=5,
                alpha=0.5,
                color=_c(f"C{[0, 3, 1][k]}"),
                label=f"{name} events",
                zorder=3,
                edgecolors="none",
            )
        ax_map.set_xlabel("longitude (deg)")
        ax_map.set_ylabel("latitude (deg)")
        ax_map.set_aspect(1 / np.cos(np.deg2rad(34.5)))
        ax_map.legend(markerscale=1.5, fontsize=7, loc="upper right")
        ax_sp.scatter(
            data.distance_km,
            s_minus_p,
            s=5,
            alpha=0.4,
            color=_c("C0"),
            edgecolors="none",
        )
        ax_sp.set_xlabel("epicentral distance (km)")
        ax_sp.set_ylabel("S - P time (s)")
        ax_sp.set_title("S - P time grows with distance")
    return _done(fig)


def plot_ceed_examples(data, indices):
    """Three-component records with the analyst P and S picks (relative to P)."""
    import matplotlib.pyplot as plt

    x = _numpy(data.u)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            3, len(indices), figsize=(9.0, 3.9), sharey="col", squeeze=False
        )
        t = (np.arange(x.shape[-1]) - 0) / 100.0
        for col, i in enumerate(indices):
            p, s = int(data.p[i]), int(data.s[i])
            lo, hi = max(p - 600, 0), min(s + 800, x.shape[-1])
            for row, name in enumerate("ENZ"):
                ax = axes[row, col]
                ax.plot(
                    t[lo:hi] - t[p],
                    x[i, row, lo:hi],
                    color=_c(COMPONENT_COLOURS[row]),
                    lw=0.6,
                )
                ax.axvline(0.0, color=_c(PHASE_COLOURS["P"]), lw=1.0)
                ax.axvline((s - p) / 100.0, color=_c(PHASE_COLOURS["S"]), lw=1.0)
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(name, rotation=0, ha="right", va="center")
                if row < 2:
                    ax.set_xticklabels([])
            axes[0, col].set_title(
                f"{data.station[i]}  {data.distance_km[i]:.0f} km"
                f"  M{data.magnitude[i]:.1f}",
                fontsize=8,
            )
            axes[2, col].set_xlabel("time from P (s)")
        for name, label in (("P", "analyst P"), ("S", "analyst S")):
            axes[0, 0].plot([], [], color=_c(PHASE_COLOURS[name]), label=label)
        axes[0, 0].legend(loc="upper left", fontsize=7)
    return _done(fig)


def plot_targets(x, target, p, s, sampling_rate: float = 100.0):
    """One cropped record with the soft P / S / neither target curves beneath it."""
    import matplotlib.pyplot as plt

    x, target = _numpy(x), _numpy(target)
    t = np.arange(x.shape[-1]) / sampling_rate
    with plt.style.context(style("open")):
        fig, (ax_x, ax_t) = plt.subplots(
            2,
            1,
            figsize=(9.0, 3.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1]},
        )
        for row in range(3):
            ax_x.plot(
                t,
                x[row] + 2.2 * (2 - row) * np.abs(x).max() / 3,
                color=_c(COMPONENT_COLOURS[row]),
                lw=0.6,
            )
        ax_x.set_yticks([])
        ax_x.set_ylabel("E, N, Z")
        for ax in (ax_x, ax_t):
            ax.axvline(p / sampling_rate, color=_c(PHASE_COLOURS["P"]), lw=0.8, ls=":")
            ax.axvline(s / sampling_rate, color=_c(PHASE_COLOURS["S"]), lw=0.8, ls=":")
        for row, name in enumerate(("P", "S", "neither")):
            ax_t.plot(t, target[row], color=_c(PHASE_COLOURS[name]), label=name)
        ax_t.set_ylim(-0.05, 1.1)
        ax_t.set_xlabel("time (s)")
        ax_t.set_ylabel("target")
        ax_t.legend(ncol=3, loc="center right", fontsize=8)
    return _done(fig)


def plot_picker_examples(x, probability, p, s, picks, sampling_rate: float = 100.0):
    """Vertical-component records with predicted P/S curves and the picks taken."""
    import matplotlib.pyplot as plt

    x, probability = _numpy(x), _numpy(probability)
    k = len(x)
    t = np.arange(x.shape[-1]) / sampling_rate
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            2,
            k,
            figsize=(9.0, 3.8),
            sharex="col",
            squeeze=False,
            gridspec_kw={"height_ratios": [1.3, 1]},
        )
        for col in range(k):
            ax_x, ax_p = axes[0, col], axes[1, col]
            ax_x.plot(t, x[col, 2], color=_c(COMPONENT_COLOURS[2]), lw=0.6)
            ax_x.set_yticks([])
            for ax in (ax_x, ax_p):
                ax.axvline(
                    float(p[col]) / sampling_rate,
                    color=_c(PHASE_COLOURS["P"]),
                    lw=0.9,
                    ls=":",
                )
                ax.axvline(
                    float(s[col]) / sampling_rate,
                    color=_c(PHASE_COLOURS["S"]),
                    lw=0.9,
                    ls=":",
                )
            for phase, name in enumerate(("P", "S")):
                if picks[col][phase] >= 0:
                    ax_x.axvline(
                        picks[col][phase] / sampling_rate,
                        color=_c(PHASE_COLOURS[name]),
                        lw=0.9,
                        ls="--",
                    )
            ax_p.plot(
                t, probability[col, 0], color=_c(PHASE_COLOURS["P"]), lw=1.0, label="P"
            )
            ax_p.plot(
                t, probability[col, 1], color=_c(PHASE_COLOURS["S"]), lw=1.0, label="S"
            )
            ax_p.set_ylim(0, 1.05)
            ax_p.set_xlabel("time (s)")
            if col:
                ax_p.set_yticklabels([])
        axes[1, 0].set_ylabel("probability")
        axes[1, 0].legend(fontsize=8, loc="upper left")
        axes[0, 0].set_title("dotted: analyst, dashed: model", fontsize=8, loc="left")
    return _done(fig)


def plot_pick_residuals(
    residual_p, residual_s, summary_p, summary_s, limit: float = 1.0
):
    """Histograms of predicted minus analyst pick time for P and S."""
    import matplotlib.pyplot as plt

    bins = np.linspace(-limit, limit, 41)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
        for ax, name, residual, summary in zip(
            axes,
            ("P", "S"),
            (residual_p, residual_s),
            (summary_p, summary_s),
            strict=True,
        ):
            residual = np.asarray(residual)
            ax.hist(
                np.clip(residual, -limit, limit),
                bins=bins,
                color=_c(PHASE_COLOURS[name]),
            )
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"{name} pick: predicted - analyst (s, clipped at {limit:g})")
            ax.set_title(
                f"median |error| {summary['median_abs']:.2f} s,"
                f" bias {summary['bias']:+.2f} s,"
                f" miss {100 * summary['miss_rate']:.0f}%",
                fontsize=8,
            )
        axes[0].set_ylabel("traces")
    return _done(fig)


def plot_residual_trends(snr, distance_km, residual_p, residual_s):
    """Absolute pick error against signal-to-noise ratio and source-station distance."""
    import matplotlib.pyplot as plt

    with plt.style.context(style("boxed")):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
        for ax, x, label in zip(
            axes,
            (snr, distance_km),
            ("signal-to-noise ratio", "distance (km)"),
            strict=True,
        ):
            for name, residual in (("P", residual_p), ("S", residual_s)):
                ok = np.isfinite(residual)
                ax.scatter(
                    np.asarray(x)[ok],
                    np.abs(residual[ok]) + 1e-3,
                    s=6,
                    alpha=0.5,
                    color=_c(PHASE_COLOURS[name]),
                    label=name,
                    edgecolors="none",
                )
            ax.set_yscale("log")
            ax.set_xlabel(label)
        axes[0].set_xscale("log")
        axes[0].set_ylabel("|pick error| (s)")
        axes[0].legend(markerscale=2)
    return _done(fig)


def plot_handoff(scene, m_true, sigmas, seed: int = 0):
    """One earthquake located from simulated P picks of different quality.

    One map per pick error in ``sigmas`` (s): the grid posterior of Part 1c (flat
    prior) for the scene's stations, with one noise draw scaled to each ``sigma``."""
    import matplotlib.pyplot as plt
    import torch

    from . import physics
    from .density_plotting import _zoom
    from .geometry import BOX
    from .plotting import map_axes, plot_stations
    from .posterior import grid_posterior

    m_true = torch.as_tensor(m_true, dtype=torch.float64)
    with plt.style.context(style("boxed")):
        fig, axes = plt.subplots(
            1, len(sigmas), figsize=(9.0, 3.6), layout="constrained", squeeze=False
        )
        for ax, sigma in zip(axes[0], sigmas, strict=True):
            torch.manual_seed(seed)
            d = physics.observe(scene.F, m_true[None], sigma, scene.live)[0]
            post = grid_posterior(scene.F, d, sigma, None, scene.live)
            density = post.density.numpy()
            nodes = np.linspace(0, BOX, density.shape[0])
            ax.contourf(
                nodes, nodes, density / density.max(), levels=np.linspace(0.02, 1, 12),
                cmap="Reds", alpha=0.9,
            )  # fmt: skip
            if post.level_68 < density.max():  # 1 km grid: a sharp posterior is a dot
                ax.contour(
                    nodes, nodes, density, levels=[post.level_68],
                    colors=[_c("C1")], linewidths=1.4,
                )  # fmt: skip
            plot_stations(ax, scene.xi_x, scene.xi_y, s=40)
            ax.plot(*m_true, "*", mfc="white", mec="k", ms=13, mew=1.1, zorder=6)
            map_axes(ax)
            ax.set_title(f"pick error {sigma:.2f} s")
            area = float((post.density >= post.level_68).sum())  # 1 km cells
            ax.text(
                0.03, 0.04, f"68% region: {area:.0f} km$^2$", transform=ax.transAxes,
                fontsize=8, zorder=9,
            )  # fmt: skip
            _zoom(ax, m_true[None].numpy(), pad=22.0)
            ax.set_ylabel("")
        axes[0, 0].set_ylabel("y (km)")
    return _done(fig)
