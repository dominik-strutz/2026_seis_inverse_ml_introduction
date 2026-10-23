"""The Lecture 1 MLP playground (anywidget).

The student draws a function y(x) on the output plot (or loads an example), sets
the network shape with +/- buttons and the training settings in the header bar.
There are two sets of points: **train** (filled) is what the network is fitted
to, **validation** (hollow) is only looked at, to show how well the fit carries over.
A toggle picks which set a drag draws or (with shift) erases; an example has as
many points of each as the two number fields say. Python reads all of that
back, trains the network on the train points (in the notebook, plain PyTorch)
and sends the run to show: snapshots of the network output and of its weights
from epoch 0 onwards, and the train and validation loss of every epoch.

The figure reads left to right, as in TensorFlow Playground: DATA (examples),
the network (one tile per neuron showing its output h(x), edges coloured by
weight), OUTPUT (train and validation loss, and the fit). Under the graph the same
network as weight matrices, each block under the edges it holds. The header sets the
training (step size, activation, epochs typed in freely), says whether to retrain at
every change or only on the train button, and plays the run back
in a few seconds: play/pause, one step, and reset, which re-initialises the
weights (a new ``seed``) and trains again.

The bookkeeping around the notebook's training loop lives here, so the loop
stays in view: ``training_setup()`` hands over the two sets and the settings,
``recorder(net, settings)`` keeps the snapshots the playback needs, and
``show_run(recorder)`` sends them.
"""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import anywidget
import torch
import traitlets
from torch import nn

from .widgetkit import Progress, esm

X_RANGE = (-1.0, 1.0)  # the drawing area, in both coordinates
Y_RANGE = (-1.0, 1.0)
MAX_WIDTH = 16  # neurons per hidden layer; every one is drawn
MAX_LAYERS = 3  # hidden layers
GOLDEN = 0.6180339887498949  # x positions of the example points
NOISE_RMS = 0.1  # rms of the example noise (a product of two sines)
TRAIN_KEYS = ["points", "val_points", "widths", "activation", "lr", "epochs", "seed"]
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
}


def example_data(
    n_train: int = 25, n_val: int = 0
) -> tuple[list[list[float]], list[list[float]]]:
    """A noisy sine, train and validation points: what the widget starts with.

    The recipe is the one of the widget's example buttons (``sample`` in
    mlpwidget.js): x on a golden-ratio sequence, so that the first n points of
    a set are also the first n of a longer one, and a fixed pseudo-random noise
    of rms 0.1. The two sets use different x and different noise.
    """
    lo, hi = X_RANGE

    def sample(n: int, which: int) -> list[list[float]]:
        pts = []
        for i in range(n):
            x = lo + (hi - lo) * ((GOLDEN * i + 0.5 * which + 0.1) % 1.0)
            noise = 2 * NOISE_RMS * math.sin(97.0 * i + 13.0 * which)
            noise *= math.cos(31.0 * i + 7.0 * which)
            pts.append([round(x, 4), round(0.8 * math.sin(3.2 * x) + noise, 4)])
        return pts

    return sample(n_train, 0), sample(n_val, 1)


class MLPPlayground(anywidget.AnyWidget):
    """Draw points; set shape and training; shows the training run as it plays."""

    _esm = esm("mlpwidget.js")

    x_range = traitlets.List(list(X_RANGE)).tag(sync=True)
    y_range = traitlets.List(list(Y_RANGE)).tag(sync=True)
    # student input (JS -> Python): the two sets, the network and training
    points = traitlets.List(example_data()[0]).tag(sync=True)  # train points
    val_points = traitlets.List(example_data()[1]).tag(sync=True)
    example = traitlets.Unicode("sine").tag(sync=True)  # "" once drawn by hand
    n_train = traitlets.Int(25).tag(sync=True)  # points of an example
    n_val = traitlets.Int(0).tag(sync=True)
    noise_train = traitlets.Float(NOISE_RMS).tag(sync=True)  # rms noise of an example
    noise_val = traitlets.Float(NOISE_RMS).tag(sync=True)
    noise_seed = traitlets.Int(0).tag(sync=True)  # the "new noise" button bumps it
    widths = traitlets.List([4]).tag(sync=True)  # one width per hidden layer
    activation = traitlets.Unicode("ReLU").tag(sync=True)  # name of phi
    lr = traitlets.Float(0.01).tag(sync=True)  # step size eta
    epochs = traitlets.Int(1000).tag(sync=True)
    seed = traitlets.Int(1).tag(sync=True)  # weight initialisation; reset bumps it
    auto_retrain = traitlets.Bool(True).tag(sync=True)  # RETRAIN: on change / on button
    # the train button: the settings it captured, and a press counter
    trained = traitlets.Dict().tag(sync=True)
    train_id = traitlets.Int(1).tag(sync=True)
    # display only (Python -> JS)
    frames = traitlets.List().tag(sync=True)  # network output on a grid, per snapshot
    frame_epochs = traitlets.List().tag(sync=True)  # epoch of each snapshot
    weights = traitlets.List().tag(sync=True)  # per snapshot, per layer [W, b]
    loss = traitlets.List().tag(sync=True)  # train loss, one value per epoch
    val_loss = traitlets.List().tag(sync=True)  # validation loss (empty: none drawn)
    fit_activation = traitlets.Unicode("ReLU").tag(sync=True)  # phi of this run
    run_id = traitlets.Int(0).tag(sync=True)  # bumped last: a complete new run
    progress = traitlets.Float(0.0).tag(sync=True)  # share of the epochs trained

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.trained:  # the page opens with one run of the defaults
            self.trained = {k: copy.deepcopy(getattr(self, k)) for k in TRAIN_KEYS}
        self._shown = None  # (recorder, press) last sent, so a rerun does not replay

    def training_setup(self):
        """The two sets as pairs and the settings captured by the last press of train.

        Returns ``x_train``, ``y_train``, ``x_val``, ``y_val`` (shape (n, 1)
        each; the validation set may be empty) and ``settings`` with ``widths`` (one
        per hidden layer), ``activation`` (an ``nn`` module class), ``lr``,
        ``epochs``, ``seed`` and ``key`` (a string that is the same for the same
        settings, to cache runs by). Stops the cell while fewer than three train
        points are drawn (the widget says so).
        """
        import marimo as mo

        t = self.trained
        train = torch.tensor(t["points"], dtype=torch.float32).reshape(-1, 2)
        val = torch.tensor(t["val_points"], dtype=torch.float32).reshape(-1, 2)
        mo.stop(len(train) < 3)
        settings = SimpleNamespace(
            widths=[int(w) for w in t["widths"]],
            activation=ACTIVATIONS[t["activation"]],
            activation_name=t["activation"],
            lr=float(t["lr"]),
            epochs=int(t["epochs"]),
            seed=int(t["seed"]),
            key=json.dumps(t, sort_keys=True),  # same key: same run, no retraining
        )
        return train[:, :1], train[:, 1:], val[:, :1], val[:, 1:], settings

    def recorder(self, net, settings, n_snapshots: int = 150):
        """A ``TrainingRecorder`` for ``net``; call its ``log`` once per epoch (the
        widget shows a progress bar while the loop runs)."""
        return TrainingRecorder(net, settings, n_snapshots, Progress(self))

    def show_run(self, recorder):
        """Send a recorded run to the widget, which plays it back (once per press)."""
        if self._shown == (id(recorder), self.train_id):
            return
        self._shown = (id(recorder), self.train_id)
        self.show(
            recorder.frames,
            recorder.frame_epochs,
            recorder.weights,
            recorder.loss,
            recorder.activation,
            recorder.val_loss,
        )

    def show(self, frames, frame_epochs, weights, loss, activation, val_loss=()):
        """Send one training run to the widget, which plays it back.

        ``frames`` is (n_snapshots, n_grid): the model output on an even grid
        over the x range, starting at epoch 0. ``weights`` is, per snapshot, one
        ``[W, b]`` pair per ``Linear`` layer, ``W`` flattened row-major; the
        widget reads the layer shapes from the payload itself. ``loss`` and
        ``val_loss`` hold one value per epoch (``val_loss`` is left out, or
        not finite, if there are no validation points).
        """
        self.frames = [[round(float(v), 4) for v in f] for f in frames]
        self.frame_epochs = [int(e) for e in frame_epochs]
        self.weights = [
            [
                [
                    [round(float(v), 4) for v in w_flat],
                    [round(float(v), 4) for v in b],
                ]
                for w_flat, b in snapshot
            ]
            for snapshot in weights
        ]
        self.loss = [float(v) for v in loss]
        val = [float(v) for v in val_loss]
        self.val_loss = val if val and all(math.isfinite(v) for v in val) else []
        self.fit_activation = str(activation)
        self.run_id += 1


def _number(value) -> float:
    """A float from a tensor (also one that needs grad) or a number."""
    return float(value.detach() if hasattr(value, "detach") else value)


class TrainingRecorder:
    """What the playground plays back: the train and validation loss of every epoch and
    about ``n_snapshots`` snapshots (epoch 0 included) of the network output on a
    grid over the drawing and of every ``Linear`` layer's ``[W flat, b]``."""

    def __init__(self, net, settings, n_snapshots: int = 150, progress=None):
        self.net = net
        self.progress = progress  # called with the share of the epochs done
        self.n_epochs = settings.epochs
        self.activation = settings.activation_name
        self.every = max(settings.epochs // n_snapshots, 1)
        self.x_grid = torch.linspace(*X_RANGE, 200)[:, None]
        self.loss, self.val_loss = [], []
        self.frames, self.frame_epochs, self.weights = [], [], []
        self._snapshot(0)

    def log(self, epoch: int, loss, val_loss=float("nan")):
        """Record the losses of this epoch; snapshot the network every few epochs."""
        self.loss.append(_number(loss))
        self.val_loss.append(_number(val_loss))
        if self.progress:
            self.progress(epoch / self.n_epochs)
        if epoch % self.every == 0 or epoch == self.n_epochs:
            self._snapshot(epoch)

    def _snapshot(self, epoch: int):
        with torch.no_grad():
            self.frames.append(self.net(self.x_grid).flatten().tolist())
            self.frame_epochs.append(epoch)
            self.weights.append(
                [
                    [m.weight.flatten().tolist(), m.bias.tolist()]
                    for m in self.net.modules()
                    if isinstance(m, nn.Linear)
                ]
            )
