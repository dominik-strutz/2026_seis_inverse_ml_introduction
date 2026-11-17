"""The Part 2c cluster playground (anywidget): name clusters with few labels.

Left, three panels with the same records in the same two numbers of the autoencoder's
latent vector: the cluster k-means found (viridis colours that mean nothing, the
cluster centres as numbered discs), the clusters named by the labels of five of their
records (one hue per class, a lighter or darker tone of it for the clusters that share
a name), and the true labels, which the method never uses. The records whose label was
looked up are ringed. Below, one row per cluster: the true labels of its labelled
records, the label the majority gives the cluster, and how the cluster does on the
test records.

Everything is computed in the notebook for every number of clusters and for several
draws of the labelled records, and sent once, so the browser switches between them at
once. The number of clusters ``k`` and the draw ``sample`` are the only traits that
travel back to Python (the notebook's later cells use them); the pair of latent numbers,
the records shown and what is pointed at stay in the browser.
"""

from __future__ import annotations

import anywidget
import numpy as np
import traitlets

from .waveform_plotting import _colour, _numpy
from .widgetkit import esm

# Tones of a class colour, darker to lighter (negative: towards black, positive: towards
# white). Clusters that share a name take them in the order TONE_ORDER, the largest
# cluster keeping the class colour itself. The range is narrow on purpose: TC is a grey.
TONES = (-0.2, 0.0, 0.22, 0.4)
TONE_ORDER = (1, 2, 0, 3)


def _rounded(a, digits: int) -> list:
    return np.round(np.asarray(_numpy(a), dtype=np.float64), digits).tolist()


def _ints(a) -> list:
    return np.asarray(_numpy(a)).astype(int).tolist()


def _class_colours(classes) -> list[str]:
    import matplotlib.pyplot as plt

    from .plotting import style

    with plt.style.context(style("open")):
        return [_colour(name) for name in classes]


def _tone(colour: str, t: float) -> str:
    """``colour`` (``#rrggbb``) made darker (``t`` < 0) or lighter (``t`` > 0)."""
    rgb = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    if t < 0:
        rgb = [c * (1 + t) for c in rgb]
    else:
        rgb = [c + (255 - c) * t for c in rgb]
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def cluster_colours(k: int) -> tuple[list[str], list[str]]:
    """Viridis colours of ``k`` clusters, spread over the map from dark purple to
    yellow-green: a quiet scale instead of ``k`` unrelated hues. Returns the colour of
    every cluster (a little lightened, for the records) and a deeper one (for the
    cluster in focus)."""
    import matplotlib

    cmap = matplotlib.colormaps["viridis"]
    deep = [matplotlib.colors.to_hex(cmap(0.95 * j / max(k - 1, 1))) for j in range(k)]
    return [_tone(c, 0.12) for c in deep], deep


def cluster_tones(names, asked, assign, k: int) -> list[int]:
    """The index into ``TONES`` of every cluster, for one draw of the labelled records.

    ``names`` (k,) is the class index of every cluster, ``asked`` the training records
    whose label was looked up and ``assign`` (records,) the cluster of every training
    record. Clusters that received the same name take different tones, the largest
    cluster first; a cluster none of whose records was looked up has no name: -1."""
    names, asked, assign = (np.asarray(_numpy(a)) for a in (names, asked, assign))
    sizes = np.bincount(assign, minlength=k)
    named = np.zeros(k, dtype=bool)
    named[assign[asked]] = True
    tones = [-1] * k
    for name in sorted({int(names[j]) for j in range(k) if named[j]}):
        sharing = [j for j in range(k) if named[j] and int(names[j]) == name]
        sharing.sort(key=lambda j: -sizes[j])  # stable: the lower number first
        for rank, j in enumerate(sharing):
            tones[j] = TONE_ORDER[rank % len(TONE_ORDER)]
    return tones


class ClusterNaming(anywidget.AnyWidget):
    """Clusters of a latent vector and the labels five records give them.

    ``k`` is the number of clusters shown and ``sample`` (counted from 0) the draw of
    the labelled records, both set in the browser.
    """

    _esm = esm("clusterwidget.js")

    classes = traitlets.List().tag(sync=True)
    colours = traitlets.List().tag(sync=True)  # one per class
    shades = traitlets.List().tag(sync=True)  # per class, the tones of its colour
    train = traitlets.Dict().tag(sync=True)  # latent, label, index
    test = traitlets.Dict().tag(sync=True)  # the same for the test records
    fits = traitlets.List().tag(sync=True)  # one entry per number of clusters
    samples = traitlets.Int(1).tag(sync=True)  # how many draws of labelled records
    k = traitlets.Int(4).tag(sync=True)
    sample = traitlets.Int(0).tag(sync=True)
    explore = traitlets.List().tag(sync=True)  # (concept, question) pairs

    def __init__(
        self,
        classes,
        latent_train,
        label_train,
        latent_test,
        label_test,
        fits,
        index_train=None,
        index_test=None,
        k: int = 4,
        explore=(),
        **kwargs,
    ):
        """``code_*`` are (records, d) and ``label_*`` (records,) class indices of the
        training and test records; ``index_*`` (optional) their numbers in the data set,
        which the widget shows. ``fits`` maps a number of clusters ``k`` to an object
        with

        * ``centres`` (k, d) and ``assign_train``, ``assign_test`` (records,): the
          cluster of every record,
        * ``names`` and ``asked``: one entry per draw, ``names[s]`` (k,) the class
          index each cluster receives and ``asked[s]`` the training records (indices
          into ``latent_train``) whose labels were looked up. A cluster none of whose
          records was looked up has no name, whatever ``names[s]`` holds there.

        ``explore`` (optional) is a list of ``(concept, question)`` pairs: the
        playground's "things to look for", set in the free corner under the panels so
        that they are on screen with it, however tall the table grows."""
        if k not in fits:
            raise ValueError(f"no fit for k = {k}; have {sorted(fits)}")
        draws = {len(fit.names) for fit in fits.values()}
        draws |= {len(fit.asked) for fit in fits.values()}
        if len(draws) != 1:
            raise ValueError("every fit needs the same number of draws")
        latent_train, latent_test = _numpy(latent_train), _numpy(latent_test)
        label_train, label_test = _numpy(label_train), _numpy(label_test)
        if index_train is None:
            index_train = np.arange(len(latent_train))
        if index_test is None:
            index_test = np.arange(len(latent_test))
        super().__init__(**kwargs)
        self.classes = list(classes)
        self.colours = _class_colours(classes)
        self.shades = [[_tone(c, t) for t in TONES] for c in self.colours]
        self.train = {
            "latent": _rounded(latent_train, 3),
            "label": _ints(label_train),
            "index": _ints(index_train),
        }
        self.test = {
            "latent": _rounded(latent_test, 3),
            "label": _ints(label_test),
            "index": _ints(index_test),
        }
        self.fits = [
            {
                "k": int(n),
                "pale": cluster_colours(int(n))[0],  # viridis, spread over this k
                "deep": cluster_colours(int(n))[1],  # the same, for the focus
                "centres": _rounded(fit.centres, 3),
                "assign": _ints(fit.assign_train),
                "assign_test": _ints(fit.assign_test),
                "names": [_ints(names) for names in fit.names],
                "asked": [_ints(asked) for asked in fit.asked],
                "tones": [
                    cluster_tones(names, asked, fit.assign_train, int(n))
                    for names, asked in zip(fit.names, fit.asked, strict=True)
                ],
            }
            for n, fit in sorted(fits.items())
        ]
        self.samples = draws.pop()
        self.k = int(k)
        self.explore = [[str(a), str(b)] for a, b in explore]
