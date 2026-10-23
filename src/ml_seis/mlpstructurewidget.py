"""A small MLP as an interactive graph plus its weight matrices (anywidget).

The static ``mlp_structure_figure`` made hoverable, with the look and the linking
of the ``MLPPlayground``: edges and matrix cells share the weight colours, hovering
an edge or a cell marks the other one with its value, a bias cell marks all edges
into its neuron, and a neuron marks its incoming edges and matrix row. No training:
the weights are the ones the network has when the widget is made.
"""

from __future__ import annotations

import anywidget
import traitlets
from torch import nn

from .widgetkit import esm

ACTIVATION_NAMES = {
    nn.ReLU: "ReLU",
    nn.Tanh: "tanh",
    nn.Sigmoid: "sigmoid",
    nn.Identity: "linear",
}


class MLPStructure(anywidget.AnyWidget):
    """Graph and weight matrices of one ``nn.Sequential`` of ``Linear`` layers."""

    _esm = esm("mlpstructurewidget.js")

    layers = traitlets.List().tag(sync=True)  # per Linear layer [W rows, b]
    activation = traitlets.Unicode("tanh").tag(sync=True)
    in_labels = traitlets.List().tag(sync=True)  # "d_1": base, "_", subscript
    out_labels = traitlets.List().tag(sync=True)
    editable = traitlets.Bool(False).tag(sync=True)  # +/- buttons resize the network

    @classmethod
    def from_net(cls, net, in_labels=None, out_labels=None, editable=False):
        """Widget for ``net``; labels are plain strings, ``_`` starts a subscript.

        With ``editable`` the buttons add or remove inputs, hidden layers and neurons
        in the browser; weights that still fit are kept, new ones are drawn at random.
        """
        linears = [m for m in net if isinstance(m, nn.Linear)]
        activation = next(
            (ACTIVATION_NAMES[type(m)] for m in net if type(m) in ACTIVATION_NAMES),
            "linear",
        )
        layers = [
            [
                [[round(float(w), 3) for w in row] for row in lin.weight.tolist()],
                [round(float(b), 3) for b in lin.bias.tolist()],
            ]
            for lin in linears
        ]
        return cls(
            layers=layers,
            activation=activation,
            in_labels=list(in_labels or []),
            out_labels=list(out_labels or []),
            editable=editable,
        )
