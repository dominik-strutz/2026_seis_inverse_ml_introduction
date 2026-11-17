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
            (17.5, "latent vector, no labels\n(2c)", _c("C4")),
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
                epochs,
                history["validation"],
                color=colour,
                lw=2.2,
                label=f"{name}, validation",
            )
            ax.plot(
                epochs,
                history["train"],
                color=colour,
                lw=2.2,
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


def plot_confusion(cm, names, report, highlight: int | None = None, figsize=(8.0, 3.2)):
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
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.1]}
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


# --------------------------------------------------------- autoencoder / U-Net


def _spectrogram_panel(
    ax, spec, hop, sampling_rate, window=None, fmax=None, fmin=None, **kwargs
):
    """Draw one spectrogram panel with time and frequency axes, as ObsPy does.

    ``spec`` is (frequency, time bin). Its rows are centred on frequencies from
    ``fmin`` to ``fmax`` (Hz), evenly spaced; by default ``fmin`` is the spacing
    itself, so the rows are ``df``, ``2 df``, ... ``fmax`` (the zero frequency is
    dropped, as in ObsPy; ``fmax`` is the Nyquist frequency by default). A window of
    ``window`` samples that starts ``hop`` samples after the last gives one time bin,
    drawn at its centre, ``window / 2`` samples after its start; without ``window``
    the time bins are drawn from time zero."""
    window = hop if window is None else window
    n_rows = spec.shape[-2]
    top = sampling_rate / 2 if fmax is None else fmax
    bottom = top / n_rows if fmin is None else fmin
    half_row = (top - bottom) / (n_rows - 1) / 2 if n_rows > 1 else top / 2
    start = (window - hop) / 2 / sampling_rate  # the left edge of the first time bin
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(
            start,
            start + spec.shape[-1] * hop / sampling_rate,
            bottom - half_row,
            top + half_row,
        ),
        **kwargs,
    )
    ax.set_xlim(0, start + spec.shape[-1] * hop / sampling_rate)
    ax.set_ylim(0, top if fmin is None else top + half_row)


def plot_spectrogram_row(
    spec,
    classes,
    indices,
    hop,
    sampling_rate,
    window=None,
    fmax=None,
    waves=None,
    fmin=None,
):
    """One record per class in a row: what the autoencoder sees and rebuilds.

    Every spectrogram has its own colour scale, from its smallest to its largest
    value, as in ObsPy. ``window`` is the length of a window in samples, ``fmax`` the
    frequency (Hz) of the top row. With ``waves`` (one record per index, in samples)
    each waveform is drawn above its spectrogram."""
    import matplotlib.pyplot as plt

    spec = _numpy(spec)
    rows = 1 if waves is None else 2
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            rows, len(indices), figsize=(9.0, 2.6 if waves is None else 3.9),
            sharex=True, squeeze=False,
            gridspec_kw={"height_ratios": [1] if waves is None else [1, 1.6]},
        )  # fmt: skip
        for col, (name, i) in enumerate(zip(classes, indices, strict=True)):
            if waves is not None:
                wave = _numpy(waves[col])
                t = np.arange(len(wave)) / sampling_rate
                axes[0, col].plot(t, wave, color=_colour(name), lw=0.7)
                axes[0, col].set_yticks([])
            axes[0, col].set_title(name, fontsize=9)
            ax = axes[-1, col]
            _spectrogram_panel(ax, spec[i], hop, sampling_rate, window, fmax, fmin)
            ax.set_xlabel("time (s)")
            if col:
                ax.set_yticks([])
        axes[-1, 0].set_ylabel("frequency (Hz)")
        if waves is not None:
            axes[0, 0].set_ylabel("waveform")
    return _done(fig)


def plot_reconstruction(
    waves,
    specs,
    specs_hat,
    names,
    latents,
    hop,
    sampling_rate,
    window=None,
    fmax=None,
    fmin=None,
):
    """One record per class, its spectrogram and what the decoder rebuilds.

    One row per record: ``waves`` (record, sample), ``specs`` and ``specs_hat``
    (record, frequency, time bin), ``names`` the records' classes and ``latents``
    (record, number) their numbers z. In a row the two spectrograms share one colour
    scale; ``window`` is the length of a window in samples and ``fmax`` the frequency
    (Hz) of their top row."""
    import matplotlib.pyplot as plt

    waves, specs, specs_hat = _numpy(waves), _numpy(specs), _numpy(specs_hat)
    latents = _numpy(latents)
    rows = len(waves)
    length = waves.shape[-1] / sampling_rate
    t = np.arange(waves.shape[-1]) / sampling_rate
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            rows, 3, figsize=(11.0, 1.55 * rows + 0.5), sharex=True, squeeze=False
        )
        for row in range(rows):
            low, high = float(specs[row].min()), float(specs[row].max())
            axes[row, 0].plot(t, waves[row], color=_colour(names[row]), lw=0.7)
            axes[row, 0].set_yticks([])
            numbers = ", ".join(f"{v:.1f}" for v in latents[row])
            axes[row, 0].set_ylabel(f"{names[row]}\nz = ({numbers})", fontsize=8)
            for ax, panel in zip(
                axes[row, 1:], (specs[row], specs_hat[row]), strict=True
            ):
                _spectrogram_panel(
                    ax, panel, hop, sampling_rate, window, fmax, fmin,
                    vmin=low, vmax=high,
                )  # fmt: skip
                if ax is axes[row, 2]:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("frequency (Hz)", fontsize=8)
        for col, title in enumerate(("waveform", "spectrogram", "rebuilt")):
            axes[0, col].set_title(title, fontsize=9)
            axes[-1, col].set_xlabel("time (s)")
        axes[0, 0].set_xlim(0, length)
    return _done(fig)


def plot_cluster_members(
    waves,
    specs,
    label,
    classes,
    hop,
    sampling_rate,
    window=None,
    fmax=None,
    fmin=None,
    figsize=(9.0, 3.4),
):
    """Records of one cluster, one column each: waveform above, spectrogram below.

    Each column is titled with the record's true label, and the waveform is drawn in
    that class's colour."""
    import matplotlib.pyplot as plt

    waves, specs, label = _numpy(waves), _numpy(specs), _numpy(label)
    n = len(waves)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(
            2, n, figsize=figsize, sharex=True, squeeze=False,
            gridspec_kw={"height_ratios": [1, 1.3]},
        )  # fmt: skip
        for col in range(n):
            name = classes[int(label[col])]
            t = np.arange(waves.shape[-1]) / sampling_rate
            axes[0, col].plot(t, waves[col], color=_colour(name), lw=0.7)
            axes[0, col].set_title(name, fontsize=9)
            axes[0, col].set_yticks([])
            _spectrogram_panel(
                axes[1, col], specs[col], hop, sampling_rate, window, fmax, fmin
            )
            axes[1, col].set_xlabel("time (s)")
            if col:
                axes[1, col].set_yticks([])
        axes[0, 0].set_ylabel("waveform")
        axes[1, 0].set_ylabel("frequency (Hz)")
    return _done(fig)


def _cluster_colours(k, saturation=0.45, value=0.93):
    """k well separated hex colours for clusters; they mean nothing. Pale by default
    so that they stay off the saturated class colours."""
    from colorsys import hsv_to_rgb

    return [
        to_hex(hsv_to_rgb((0.08 + 0.618 * j) % 1.0, saturation, value))
        for j in range(k)
    ]


def plot_corner(latent, group=None, classes=None, centres=None, title="", size=5.2):
    """Corner plot of a latent vector: every pair of numbers, no marginals.

    ``latent`` is (records, d); the grid is (d - 1) by (d - 1), the panels below
    the diagonal only, so that it can be as large as possible. ``group`` colours
    the records by class index into ``classes``; without it they are grey, as an
    unlabelled latent vector is. ``centres`` (k, d) are drawn as their index in
    the panels. The figure is ``size`` inches square."""
    import matplotlib.pyplot as plt

    latent = _numpy(latent)
    d = latent.shape[1]
    n = d - 1
    low, high = np.percentile(latent, 0.5, axis=0), np.percentile(latent, 99.5, axis=0)
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(n, n, figsize=(size, size), squeeze=False)
        sets = (
            [(np.ones(len(latent), bool), _c("C6"), None)]
            if group is None
            else [
                (_numpy(group) == k, _colour(name), name)
                for k, name in enumerate(classes)
            ]
        )
        for r in range(n):
            for j in range(n):
                ax = axes[r, j]
                i = r + 1  # the number on the vertical axis
                if j > r:
                    ax.axis("off")
                    continue
                for mask, colour, _ in sets:
                    ax.scatter(
                        latent[mask, j], latent[mask, i], s=9, alpha=0.6,
                        color=colour, edgecolors="none",
                    )  # fmt: skip
                ax.set_xlim(low[j], high[j])
                ax.set_ylim(low[i], high[i])
                if r < n - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(f"$z_{j + 1}$", fontsize=14)
                    ax.tick_params(labelsize=10)
                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(f"$z_{i + 1}$", fontsize=14)
                    ax.tick_params(labelsize=10)
        if centres is not None:
            centres = _numpy(centres)
            for r in range(n):
                for j in range(r + 1):
                    for index, c in enumerate(centres):
                        axes[r, j].text(
                            c[j], c[r + 1], str(index), ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black",
                            bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                                      ec="black", lw=1.0, alpha=0.95),
                        )  # fmt: skip
        if group is not None:
            handles = [
                plt.Line2D([], [], marker="o", ls="", color=colour, label=name)
                for _, colour, name in sets
            ]
            fig.legend(
                handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96),
                frameon=False, fontsize=13, markerscale=1.6,
            )  # fmt: skip
        if title:
            fig.text(0.98, 0.5 if n > 1 else 0.9, title, ha="right", va="center",
                     fontsize=13, wrap=True)  # fmt: skip
    return _done(fig)


# ------------------------------------------------------------------- CEED / picks


def plot_targets(x, target, p, s, sampling_rate: float = 100.0, margin: float = 1.5):
    """One cropped record with the soft P / S / neither target curves beneath it.

    The time axis is zoomed to ``margin`` seconds before the P pick and after the S
    pick, so the width of the target peaks is visible; ``margin=None`` shows it all.
    """
    import matplotlib.pyplot as plt

    x, target = _numpy(x), _numpy(target)
    t = np.arange(x.shape[-1]) / sampling_rate
    with plt.style.context(style("open")):
        fig, (ax_x, ax_t) = plt.subplots(
            2,
            1,
            figsize=(11.5, 4.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.8, 1]},
        )
        # each component on its own scale, one unit apart, so a weak P stays visible
        for row in range(3):
            ax_x.plot(
                t,
                0.42 * x[row] / np.abs(x[row]).max() + (2 - row),
                color=_c(COMPONENT_COLOURS[row]),
                lw=1.0,
            )
        ax_x.set_yticks([0, 1, 2], ["Z", "N", "E"])
        ax_x.set_ylim(-0.6, 2.6)
        for ax in (ax_x, ax_t):
            ax.axvline(p / sampling_rate, color=_c(PHASE_COLOURS["P"]), lw=1.4, ls="--")
            ax.axvline(s / sampling_rate, color=_c(PHASE_COLOURS["S"]), lw=1.4, ls="--")
        for row, name in enumerate(("P", "S", "neither")):
            ax_t.plot(t, target[row], color=_c(PHASE_COLOURS[name]), lw=2.4, label=name)
        ax_t.set_ylim(-0.05, 1.1)
        ax_t.set_xlabel("time (s)")
        ax_t.set_ylabel("target")
        ax_t.legend(ncol=3, loc="center", fontsize=8)
        if margin is not None:
            ax_t.set_xlim(
                max(0.0, p / sampling_rate - margin),
                min(t[-1], s / sampling_rate + margin),
            )
    return _done(fig)


def plot_pick_residuals(residuals: dict, limit: float = 1.0):
    """Histograms of predicted minus analyst pick time (P, S), log counts.

    ``residuals`` maps a model name to its ``(residual_p, residual_s)``; the first
    model is drawn filled, the others as outlines. Errors beyond ``limit`` pile up in
    the end bins, so the tail stays visible instead of being cut off.
    """
    import matplotlib.pyplot as plt

    bins = np.linspace(-limit, limit, 41)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
        for column, (ax, name) in enumerate(zip(axes, ("P", "S"), strict=True)):
            for k, (model, pair) in enumerate(residuals.items()):
                values = np.clip(np.asarray(pair[column], dtype=float), -limit, limit)
                if k == 0:
                    ax.hist(
                        values,
                        bins=bins,
                        color=_c(PHASE_COLOURS[name]),
                        label=model,
                    )
                else:
                    ax.hist(
                        values,
                        bins=bins,
                        histtype="step",
                        color=_c("C6"),
                        lw=2.2,
                        label=model,
                    )
            ax.axvline(0, color=_c("k"), lw=1.4)
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.7)
            ax.set_xlabel(
                f"{name} pick: predicted - analyst (s, clipped at \u00b1{limit:g})"
            )
        axes[0].set_ylabel("traces (log scale)")
        if len(residuals) > 1:
            axes[1].legend(fontsize=8, loc="upper right")
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
                    s=12,
                    alpha=0.7,
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


SPLIT_COLOURS = ("C0", "C3", "C1")  # train, validation, test
SPLIT_NAMES = ("train", "validation", "test")


def plot_split_by_event(data, n_events: int = 14, seed: int = 0):
    """The same traces dealt to train / validation / test two ways, one row per event.

    Left: whole traces dealt at random (the split we do not use): most events end up
    in several sets. Right: whole events (the split we use): every row is one colour.
    ``data`` is a ``PickerData``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from . import waveform_metrics as metrics

    event, split = np.asarray(data.event_id), _numpy(data.split)
    draw = metrics.random_trace_split(len(event), seed=seed)
    leak = metrics.trace_split_leak(event, draw)
    ids, counts = np.unique(event, return_counts=True)
    rng = np.random.default_rng(seed)
    pool = ids[counts >= 5]
    chosen = rng.choice(pool, size=min(n_events, len(pool)), replace=False)
    size = dict(zip(ids, counts, strict=True))
    chosen = sorted(chosen, key=lambda e: -size[e])
    width = max(size[e] for e in chosen)
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
        leak_note = f"{100 * leak:.0f}% of test traces have a sibling in training"
        for ax, assignment, title in (
            (axes[0], draw, f"split by trace\n{leak_note}"),
            (axes[1], split, "split by earthquake\nno test trace does"),
        ):
            for row, e in enumerate(chosen):
                members = np.flatnonzero(event == e)
                ax.scatter(
                    np.arange(len(members)),
                    np.full(len(members), row),
                    s=90,
                    marker="s",
                    c=[_c(SPLIT_COLOURS[k]) for k in assignment[members]],
                    edgecolors="none",
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("traces of the earthquake")
            ax.set_xticks([])
            ax.set_xlim(-0.7, width - 0.3)
            ax.set_ylim(len(chosen) - 0.4, -0.8)
        axes[0].set_ylabel("earthquakes, one row each")
        axes[0].set_yticks([])
        axes[1].legend(
            handles=[
                Patch(color=_c(c), label=n)
                for c, n in zip(SPLIT_COLOURS, SPLIT_NAMES, strict=True)
            ],
            loc="lower right",
            fontsize=9,
            frameon=False,
        )
    return _done(fig)


def plot_pick_scatter(picks, p, s, sampling_rate: float = 100.0):
    """Model pick time against analyst pick time: P, S and the S - P time.

    ``picks`` is ``(B, 2)`` sample indices (-1: missed); ``p`` and ``s`` are the analyst
    samples. Points on the diagonal agree; the S - P panel is what a locator needs.
    """
    import matplotlib.pyplot as plt

    picks = np.asarray(picks, dtype=float)
    truth = np.stack([_numpy(p), _numpy(s)], axis=1) / sampling_rate
    found = picks >= 0
    model = np.where(found, picks / sampling_rate, np.nan)
    both = found.all(axis=1)
    panels = (
        ("P pick", truth[:, 0], model[:, 0], PHASE_COLOURS["P"], found[:, 0]),
        ("S pick", truth[:, 1], model[:, 1], PHASE_COLOURS["S"], found[:, 1]),
        (
            "S - P time",
            truth[:, 1] - truth[:, 0],
            model[:, 1] - model[:, 0],
            "C6",
            both,
        ),
    )
    with plt.style.context(style("open")):
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3))
        for ax, (name, x, y, colour, ok) in zip(axes, panels, strict=True):
            lo, hi = np.nanmin(x[ok]), np.nanmax(x[ok])
            ax.plot([lo, hi], [lo, hi], color=_c("k"), lw=1.4)
            ax.scatter(
                x[ok], y[ok], s=12, alpha=0.7, color=_c(colour), edgecolors="none"
            )
            ax.set_xlabel(f"analyst {name} (s)")
            ax.set_title(f"{int((~ok).sum())} missed", fontsize=8, loc="right")
            ax.set_aspect("equal", adjustable="datalim")
        axes[0].set_ylabel("model (s)")
    return _done(fig)
