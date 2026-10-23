"""The Lecture 1 locator playground (anywidget).

The student edits a scene on two maps (stations, P-speed model, pick noise, the
priors and a draggable test event), chooses a network and its training, and presses
**train**. The press captures the scene and every setting (``trained``); the notebook
trains on that (plain PyTorch, in the notebook) and the widget plays the run back.

Everything after the press is evaluated with the last trained network on the
*current* scene, so moving or muting a station, switching the P-speed model or
raising the noise shows what a network does with data it was not trained for. The
classical posterior of the test event uses what the analyst assumes: the current
stations with the trained P-speed model and noise.

Python side: ``train_config()`` and ``current_scene(config)`` read the widget;
``Recorder`` keeps learning curves and weight snapshots during training;
``show(run, floor, locate, observe, posterior_of)`` replays the snapshots on the
current scene and sends everything to the figure; ``network_figure(net)`` draws the
chosen network. Networks, losses, scaling and the training loop are not here, they
are the lesson.

With ``show_output=True`` the playground also offers the network's OUTPUT: one
point (MSE) or a Gaussian mixture of K components. For a mixture,
``posterior_of(d)`` returns the network's posterior q(m | d~) in km as a torch
distribution, and the maps can show it ("network") next to the exact posterior.
"""

from __future__ import annotations

import copy
import io
import json
import math
import time
import weakref
from functools import cache

import anywidget
import numpy as np
import torch
import traitlets
from torch import nn

from .density_plotting import mixture_components
from .geometry import BOX, DEFAULT_SOURCE, MAX_STATIONS, make_stations
from .posterior import (
    GRID,
    grid_log_prior,
    grid_posterior,
    grid_sources,
    posterior_from_log,
)
from .priors import sample_prior
from .scene import Scene, centred_arrivals, make_scene, ray_paths
from .velocity import SEED, Velocity
from .widgetkit import Progress, colour_layer, esm

N_POST = 201  # posterior grid nodes per side (0.5 km): narrow posteriors stay resolved
N_GRID = 25  # true sources per side for the error map and the drag lookup
GRID_NODES = torch.linspace(2.0, BOX - 2.0, N_GRID, dtype=torch.float64)
N_SNAPSHOTS = 30  # weight snapshots kept for the playback
N_DEFAULT_STATIONS = 5  # stations the playground opens with
N_TEST = 200  # test events from the test prior
N_DRAWS = 64  # noise draws of the dragged test event
RAY_POINTS = 4  # every this many points of a traced ray are sent

# Everything the training depends on; the train button copies these.
TRAIN_KEYS = (
    "stations",
    "muted",
    "velocity",
    "field",
    "sigma",
    "train_prior",
    "n_train",
    "widths",
    "activation",
    "optimiser",
    "lr",
    "batch_size",
    "epochs",
    "normalise",
    "output",
    "n_components",
)
DEFAULTS = {
    "muted": [],
    "velocity": "homogeneous",
    "field": SEED,
    "sigma": 0.3,
    "train_prior": "uniform",
    "test_prior": "uniform",
    "n_train": 1000,
    "widths": [16, 16],
    "activation": "tanh",
    "optimiser": "Adam",
    "lr": 1e-2,
    "batch_size": 64,
    "epochs": 60,
    "normalise": True,
    "output": "point",
    "n_components": 2,
}


def station_list(n: int) -> list[list[float]]:
    """The first ``n`` stations of the course network as [[x, y], ...] in km."""
    xi_x, xi_y = make_stations(n)
    return [
        [round(x, 2), round(y, 2)]
        for x, y in zip(xi_x.tolist(), xi_y.tolist(), strict=True)
    ]


# ---- training record -----------------------------------------------------------------


class Recorder:
    """Learning curves (RMS location error, km, one per epoch) and about
    ``N_SNAPSHOTS`` copies of the weights for the playback. While the playground
    waits for a press of train, ``log`` also reports the progress to its bar."""

    def __init__(self, net, n_epochs: int):
        self.net, self.n_epochs = net, n_epochs
        self.progress = LocatorPlayground.progress_of_press()
        self.every = max(n_epochs // N_SNAPSHOTS, 1)
        self.rms_train, self.rms_val = [], []
        self.snapshots, self.snapshot_epochs = [], []
        self.best_epoch = 0  # the epoch with the lowest validation error
        self.diverged = False
        self._t0 = time.perf_counter()
        self.seconds = 0.0

    def log(self, epoch: int, rms_train: float, rms_val: float):
        self.rms_train.append(rms_train)
        self.rms_val.append(rms_val)
        if not (math.isfinite(rms_train) and math.isfinite(rms_val)):
            self.diverged = True
        elif rms_val < self.rms_val[self.best_epoch]:
            self.best_epoch = epoch
        last = epoch == self.n_epochs or self.diverged
        if epoch % self.every == 0 or last:
            self.snapshots.append(copy.deepcopy(self.net.state_dict()))
            self.snapshot_epochs.append(epoch)
        self.seconds = time.perf_counter() - self._t0
        if self.progress:
            self.progress(epoch / self.n_epochs)


# ---- the payload ---------------------------------------------------------------------


def _r(values, digits=2):
    """Rounded nested lists of the same shape, for the JSON payload."""
    values = torch.as_tensor(values).detach().double().numpy()
    out = np.round(values, digits).astype(object)
    out[~np.isfinite(values)] = None  # a diverged run: JSON has no NaN
    return out.tolist()


@cache
def _prior_layer(prior: str):
    density = grid_log_prior(prior).exp().reshape(GRID, GRID).numpy()  # [y, x]
    rel = np.sqrt(density / density.max())  # square root keeps weak areas visible
    support = density[density > 0]  # the uniform prior is zero outside the map
    if support.max() < 1.01 * support.min():  # flat: mid-scale, so the events show
        return colour_layer(rel, 0, 2, "Blues", "√(prior / max)", ticks=[0, 1, 2])
    return colour_layer(rel, 0, 1, "Blues", "√(prior / max)", ticks=[0, 0.5, 1])


def _speed_layer(velocity: Velocity):
    speed = velocity.speed_grid().numpy()
    half = float(max(np.abs(speed - velocity.v_ref).max(), 0.05 * velocity.v_ref))
    layer = colour_layer(
        speed,
        velocity.v_ref - half,
        velocity.v_ref + half,
        "coolwarm_r",
        "P speed (km/s)",
    )
    layer["constant"] = bool(velocity.constant)  # the map is one colour
    layer["v_ref"] = velocity.v_ref
    return layer


def _posterior_layer(post, label, cmap="Greens"):
    """A grid posterior as a colour layer with its 68 % and 95 % contours and mean."""
    import contourpy

    # contours on a finer grid, so they stay smooth in the zoomed inset: bicubic in
    # log density, which is close to quadratic near the peak (a narrow posterior
    # would be cut into steps by interpolating the density itself)
    log_rel = (post.density / post.density.max()).clamp_min(1e-300).log().clamp_min(-30)
    fine = nn.functional.interpolate(
        log_rel[None, None], scale_factor=3, mode="bicubic", align_corners=True
    )[0, 0].exp().numpy() * float(post.density.max())
    nodes = np.linspace(0, BOX, fine.shape[0])
    lines = contourpy.contour_generator(nodes, nodes, fine).lines
    layer = colour_layer(fine / fine.max(), 0, 1, cmap, label, ticks=[0, 0.5, 1])
    layer["contours"] = {
        name: [
            np.round(np.asarray(seg)[:: 1 + len(seg) // 120], 2).tolist()
            for seg in lines(level)
        ]
        for name, level in (("68", post.level_68), ("95", post.level_95))
    }
    layer["mean"] = [round(float(v), 2) for v in post.mean]
    return layer


def _rays(scene: Scene, source) -> list:
    """Ray paths from the source to each station as [[x, y], ...] (``None`` for a
    muted station), thinned out for the payload."""
    paths = ray_paths(scene, source)
    keep = [*range(0, paths.shape[1] - 1, RAY_POINTS), paths.shape[1] - 1]
    return [
        _r(path[keep], 2) if scene.live[i] else None for i, path in enumerate(paths)
    ]


def _replay(run, locate, d):
    """The network's answer for centred picks ``d`` at every snapshot of the run,
    shape (snapshots, n, 2); the network ends with its final weights again."""
    final = copy.deepcopy(run.net.state_dict())
    out = []
    with torch.no_grad():
        for state in run.recorder.snapshots:
            run.net.load_state_dict(state)
            out.append(locate(d))
    run.net.load_state_dict(final)
    return torch.stack(out)


def _components(q):
    """The components of a Gaussian mixture behind ``q`` for the payload."""
    parts = mixture_components(q)
    if parts is None:
        return None
    return [
        {"w": round(w, 3), "mean": _r(mu), "cov": _r(cov, 3)} for w, mu, cov in parts
    ]


def _learned_layer(posterior_of, d):
    """The network's posterior for one event's picks ``d`` (1, N): its density on
    the grid with 68 % and 95 % regions and mean, and the components of a mixture."""
    m = grid_sources(N_POST)
    with torch.no_grad():
        log_q = posterior_of(d.expand(len(m), -1)).log_prob(m.float())
        q = posterior_of(d)
    layer = _posterior_layer(
        posterior_from_log(log_q, N_POST), "network / max", "Oranges"
    )
    layer["components"] = _components(q)
    return layer


# ---- the widget ----------------------------------------------------------------------


class LocatorPlayground(anywidget.AnyWidget):
    """Scene on two maps, network and training settings, a train button; plays the
    run back and shows the network on a draggable test event."""

    _esm = esm("locatorwidget.js")

    box = traitlets.Float(BOX).tag(sync=True)
    min_stations = traitlets.Int(3).tag(sync=True)
    max_stations = traitlets.Int(MAX_STATIONS).tag(sync=True)
    n_grid = traitlets.Int(N_GRID).tag(sync=True)
    auto_retrain = traitlets.Bool(True).tag(sync=True)  # RETRAIN: on change / on button
    # where the "+" next to the station count may put a station (the one farthest
    # from the current stations is taken)
    reference_stations = traitlets.List(station_list(MAX_STATIONS)).tag(sync=True)
    # the scene (JS -> Python)
    stations = traitlets.List(station_list(N_DEFAULT_STATIONS)).tag(sync=True)
    muted = traitlets.List(DEFAULTS["muted"]).tag(sync=True)  # station indices
    test_event = traitlets.List(list(DEFAULT_SOURCE)).tag(sync=True)
    velocity = traitlets.Unicode(DEFAULTS["velocity"]).tag(sync=True)
    field = traitlets.Int(DEFAULTS["field"]).tag(sync=True)
    sigma = traitlets.Float(DEFAULTS["sigma"]).tag(sync=True)  # pick noise (s)
    train_prior = traitlets.Unicode(DEFAULTS["train_prior"]).tag(sync=True)
    test_prior = traitlets.Unicode(DEFAULTS["test_prior"]).tag(sync=True)
    n_train = traitlets.Int(DEFAULTS["n_train"]).tag(sync=True)
    # the network and its training (JS -> Python)
    widths = traitlets.List(DEFAULTS["widths"]).tag(sync=True)
    activation = traitlets.Unicode(DEFAULTS["activation"]).tag(sync=True)
    optimiser = traitlets.Unicode(DEFAULTS["optimiser"]).tag(sync=True)
    lr = traitlets.Float(DEFAULTS["lr"]).tag(sync=True)
    batch_size = traitlets.Int(DEFAULTS["batch_size"]).tag(sync=True)  # 0: all
    epochs = traitlets.Int(DEFAULTS["epochs"]).tag(sync=True)
    normalise = traitlets.Bool(DEFAULTS["normalise"]).tag(sync=True)
    output = traitlets.Unicode(DEFAULTS["output"]).tag(sync=True)  # point/mixture
    n_components = traitlets.Int(DEFAULTS["n_components"]).tag(sync=True)  # K
    # layout (Python -> JS, set once): the OUTPUT controls and the maps' backgrounds
    show_output = traitlets.Bool(False).tag(sync=True)
    backgrounds = traitlets.List(["P speed", "posterior"]).tag(sync=True)
    # the train button: the settings it captured, and a press counter
    trained = traitlets.Dict().tag(sync=True)
    train_id = traitlets.Int(1).tag(sync=True)
    # display only (Python -> JS)
    progress = traitlets.Float(0.0).tag(sync=True)  # share of the epochs trained
    run = traitlets.Dict().tag(sync=True)  # learning curves, snapshots' epochs
    scene_view = traitlets.Dict().tag(sync=True)  # test events and the error map
    event_view = traitlets.Dict().tag(sync=True)  # the dragged test event

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.trained:  # the page opens with one run of the defaults
            self.trained = {k: copy.deepcopy(getattr(self, k)) for k in TRAIN_KEYS}
        self._report = Progress(self)
        self._keys = {}  # what was last sent, so unchanged parts are not recomputed
        self._figures = {}  # network figures already drawn

    # ---- reading the widget -------------------------------------------------------
    def train_config(self) -> dict:
        """The scene and settings captured by the last press of train."""
        LocatorPlayground._serving = weakref.ref(self)
        config = dict(self.trained)
        config["widths"] = [int(w) for w in config["widths"]]
        config["muted"] = sorted(int(i) for i in config["muted"])
        return config

    _serving = None  # the playground whose train_config() was read last

    @classmethod
    def progress_of_press(cls):
        """The progress reporter of the playground whose press is being trained, or
        ``None`` (a cached run, or a loop that is no press, such as an ensemble)."""
        widget = cls._serving() if cls._serving else None
        if widget is None or widget.train_id == widget.run.get("train_id"):
            return None
        return widget._report

    def current_scene(self, config: dict) -> tuple[Scene, Scene]:
        """``now``: the scene as it is on the map (data come from it);
        ``assumed``: the same stations with the P-speed model and noise of the
        training (what the analyst believes)."""
        now = make_scene(
            self.stations, self.velocity, self.field, self.sigma, self.muted
        )
        assumed = make_scene(
            self.stations,
            config["velocity"],
            config["field"],
            config["sigma"],
            self.muted,
        )
        return now, assumed

    # ---- sending ------------------------------------------------------------------
    def show(self, run, floor, locate, observe, posterior_of=None):
        """Replay the run on the current scene and send it to the figure.

        ``run`` has ``net``, ``recorder``, ``config`` and ``m_train``; ``floor`` is
        the RMS error of the posterior mean (km); ``locate(d)`` maps centred picks
        (n, N) to sources (n, 2) with the network's current weights; and
        ``observe(scene, m, generator)`` simulates centred noisy picks. For a
        mixture output, ``posterior_of(d)`` returns the network's posterior in km
        (a torch distribution with batch shape (n,)). Only the parts whose inputs
        changed are recomputed and sent.
        """
        config = run.config
        now, assumed = self.current_scene(config)
        fits = now.n_stations == len(config["stations"])
        scene_key = [
            config,
            self.train_id,
            self.stations,
            self.muted,
            self.velocity,
            self.field,
            self.sigma,
            self.test_prior,
        ]
        if self._changed("run", [config, self.train_id]):
            self._send_run(run, floor)
        if self._changed("scene", scene_key):
            self.scene_view = self._scene_view(run, locate, observe, now, fits)
        if self._changed("event", [scene_key, self.test_event]):
            self.event_view = self._event_view(
                run, locate, observe, now, assumed, fits, posterior_of
            )

    def _changed(self, name, key) -> bool:
        key = json.dumps(key, sort_keys=True, default=str)
        if self._keys.get(name) == key:
            return False
        self._keys[name] = key
        return True

    def _send_run(self, run, floor):
        rec = run.recorder
        every = max(len(run.m_train) // 500, 1)  # a sample of the training events
        self.run = {
            "train_id": self.train_id,
            "config": run.config,
            "rms_train": _r(rec.rms_train, 4),
            "rms_val": _r(rec.rms_val, 4),
            "floor": round(float(floor), 4),
            "best_epoch": rec.best_epoch,
            "diverged": rec.diverged,
            "snapshot_epochs": rec.snapshot_epochs,
            "seconds": round(rec.seconds, 2),
            "train_events": _r(run.m_train[::every], 1),
        }

    def _scene_view(self, run, locate, observe, now, fits):
        """Backgrounds, and the network on test events and on a grid of sources."""
        view = {
            "ok": fits,
            "layers": {
                "prior": _prior_layer(run.config["train_prior"]),
                "speed": _speed_layer(now.velocity),
            },
        }
        if not fits:
            n_in = len(run.config["stations"])
            view["message"] = (
                f"The network takes {n_in} picks and the map has "
                f"{now.n_stations} stations: press train."
            )
            return view
        gen = torch.Generator().manual_seed(1)
        m_test = sample_prior(self.test_prior, N_TEST, gen)
        d_test = observe(now, m_test, gen)
        m_grid = torch.cartesian_prod(GRID_NODES, GRID_NODES).flip(1)  # x runs fastest
        tic = time.perf_counter()
        locate(d_test)
        seconds = time.perf_counter() - tic
        test_pred = _replay(run, locate, d_test)
        rms = ((test_pred - m_test) ** 2).sum(-1).mean(-1).sqrt()
        grid_pred = _replay(run, locate, centred_arrivals(now, m_grid))
        return view | {
            "test_true": _r(m_test, 1),
            "test_pred": _r(test_pred, 1),
            "grid_pred": [_r(p.flatten(), 1) for p in grid_pred],
            "rms_test": _r(rms, 3),
            "us_per_event": 1e6 * seconds / N_TEST,
        }

    def _event_view(self, run, locate, observe, now, assumed, fits, posterior_of):
        """The dragged test event: rays, posterior, likelihood and the network's
        answers for noise draws of its picks (the first draw is the observed one)."""
        config = run.config
        m_event = torch.tensor(self.test_event, dtype=torch.float64)
        d_event = observe(
            now, m_event.expand(N_DRAWS, 2), torch.Generator().manual_seed(2)
        )
        args = (assumed.F, d_event[0], assumed.sigma)
        posterior = grid_posterior(*args, config["train_prior"], assumed.live, n=N_POST)
        likelihood = grid_posterior(*args, None, assumed.live, n=N_POST)
        wrong_model = (self.velocity, self.field) != (
            config["velocity"],
            config["field"],
        )
        view = {
            "ok": fits,
            "event": _r(m_event),
            "rays": _rays(now, m_event),
            "rays_assumed": _rays(assumed, m_event) if wrong_model else None,
            "posterior": _posterior_layer(posterior, "posterior / max"),
            "likelihood": _posterior_layer(likelihood, "likelihood / max"),
        }
        if fits:
            view["cloud"] = _r(
                _replay(run, locate, d_event), 2
            )  # (snapshots, draws, 2)
            if posterior_of is not None and config.get("output", "point") != "point":
                view["learned"] = _learned_layer(posterior_of, d_event[:1])
        return view

    # ---- the network, drawn -------------------------------------------------------
    def network_figure(self, trained_net):
        """``mlp_structure_figure`` of the network chosen here: the stations, hidden
        layers, widths and activation set in the playground. It shows ``trained_net``
        (the last training) if that has this shape, otherwise the initial weights of
        a new network, greyed. The picture is rendered once and kept: marimo runs
        this on every change of the playground, and drawing is slow."""
        import marimo as mo

        from .mlpwidget import ACTIVATIONS

        widths = [int(w) for w in self.widths]
        n_out = 6 * self.n_components if self.output == "mixture" else 2
        dims = [len(self.stations), *widths, n_out]
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(n_in, n_out), ACTIVATIONS[self.activation]()]
        net = nn.Sequential(*layers[:-1])
        trained = _shape(net) == _shape(trained_net)
        weights = sum(
            float(p.detach().double().sum()) for p in trained_net.parameters()
        )
        key = (*dims, self.activation, self.n_train, trained, repr(weights))
        if key not in self._figures:
            if trained:
                net = trained_net
            else:  # the initial weights the next training starts from
                torch.manual_seed(0)  # as in the notebook's training
                net = nn.Sequential(*(_fresh(m) for m in net))
            png = _png_bytes(
                _structure_figure(net, trained, self.n_train, self.n_components)
            )
            self._figures = {**dict(list(self._figures.items())[-7:]), key: png}
        return mo.image(self._figures[key])


def _shape(net):
    """The layer types and parameter shapes of a network, to tell if two match."""
    return [type(m) for m in net], [p.shape for p in net.parameters()]


def _fresh(module):
    """A new initialisation of a ``Linear`` layer; other modules are kept."""
    if isinstance(module, nn.Linear):
        return nn.Linear(module.in_features, module.out_features)
    return module


def _png_bytes(figure, dpi=110) -> bytes:
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def _output_labels(n_out, n_components):
    """Names of the network's outputs: (m_x, m_y), or the 6 K numbers of a mixture
    (K logits, K means, K Cholesky factors)."""
    if n_out == 2:
        return [r"$m_x$", r"$m_y$"]
    K = n_components
    return (
        [rf"$a_{{{k}}}$" for k in range(1, K + 1)]
        + [rf"$\mu_{{{k}{c}}}$" for k in range(1, K + 1) for c in "xy"]
        + [rf"$L_{{{k}{c}}}$" for k in range(1, K + 1) for c in "abc"]
    )


def _structure_figure(net, trained, n_train, n_components=2):
    from .plotting import mlp_structure_figure

    parameters = list(net.parameters())
    label = "trained weights" if trained else "initial weights: press train"
    if not all(bool(p.isfinite().all()) for p in parameters):
        label = "the training diverged, lower the step size"
    per_event = sum(p.numel() for p in parameters) / n_train
    n_stations = parameters[0].shape[1]
    return mlp_structure_figure(
        net,
        in_labels=[rf"$\tilde{{d}}_{{{i + 1}}}$" for i in range(n_stations)],
        out_labels=_output_labels(parameters[-1].shape[0], n_components),
        max_nodes=16,
        trained=trained,
        caption=f"{label} · {per_event:.1f} parameters per training event",
    )
