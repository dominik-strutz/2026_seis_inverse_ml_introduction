"""The Lecture 1 optimiser playground figure (anywidget).

A header bar holds playback (reset, play/pause, step) and all settings: the scene
(P speed, random field, pick noise) and the optimiser (method, step size,
momentum, steps). Two maps (P speed and misfit) share one scene; below them the
misfit curve of the run and the results at the replayed step. The student moves
the true source (star), the optimiser start (circle) and the stations (triangles)
with the mouse; + and - in the header change the number of stations. Python
reads the scene and the settings back, runs the optimiser (in
the notebook) and sends rays, the optimiser path and the loss curves, which the
widget plays back.
"""

from __future__ import annotations

import anywidget
import numpy as np
import traitlets

from .geometry import BOX
from .widgetkit import colour_layer, esm


class MapEditor(anywidget.AnyWidget):
    """Draggable stations, true source and start on the speed map (km)."""

    _esm = esm("mapwidget.js")

    box = traitlets.Float(BOX).tag(sync=True)
    min_stations = traitlets.Int(3).tag(sync=True)
    max_stations = traitlets.Int(60).tag(sync=True)
    # student input (JS -> Python)
    stations = traitlets.List().tag(sync=True)
    source = traitlets.List().tag(sync=True)
    start = traitlets.List().tag(sync=True)
    velocity = traitlets.Unicode("homogeneous").tag(sync=True)  # or "heterogeneous"
    field = traitlets.Int(5).tag(sync=True)  # which random P-speed field
    sigma = traitlets.Float(0.3).tag(sync=True)  # pick noise (s)
    method = traitlets.Unicode("SGD").tag(sync=True)  # or "Adam"
    momentum = traitlets.Float(0.0).tag(sync=True)  # SGD only
    lr = traitlets.Float(3.0).tag(sync=True)  # step size eta
    steps = traitlets.Int(10).tag(sync=True)
    # display only (Python -> JS)
    layers = traitlets.Dict().tag(sync=True)  # name -> {png, bar, lo, hi, label}
    grid_min = traitlets.List().tag(sync=True)
    rays = traitlets.List().tag(sync=True)
    path = traitlets.List().tag(sync=True)
    full_loss = traitlets.List().tag(sync=True)  # S on all stations, one per path point
    batch_loss = traitlets.List().tag(sync=True)  # S on the batch; empty if full batch
    loss_min = traitlets.Float(0.0).tag(sync=True)
    run_id = traitlets.Int(0).tag(sync=True)  # bumped last: a complete new run

    def show(self, velocity, rays, path, S_grid, m_grid, full_loss, batch_loss=()):
        """Update the picture: speed map, ray paths (steps, N, 2), optimiser path,
        the misfit landscape ``S_grid`` (grid, grid) indexed [y, x], and the loss
        curves (``full_loss``: one value per path point; ``batch_loss``: one fewer)."""
        speed = velocity.speed_grid().numpy()
        # symmetric about the reference speed, so white means "no anomaly"
        v_half = float(max(np.abs(speed - velocity.v_ref).max(), 0.05 * velocity.v_ref))
        v_lo, v_hi = velocity.v_ref - v_half, velocity.v_ref + v_half
        logS = np.log10(np.maximum(S_grid.numpy(), 1e-6))
        s_lo, s_hi = float(logS.min()), float(np.percentile(logS, 98))
        self.layers = {
            "velocity": colour_layer(speed, v_lo, v_hi, "coolwarm_r", "P speed (km/s)"),
            "misfit": colour_layer(logS, s_lo, s_hi, "Blues_r", "log10 S (km²)"),
        }
        self.grid_min = np.round(np.asarray(m_grid), 2).tolist()
        self.rays = np.round(rays.numpy()[::2].transpose(1, 0, 2), 2).tolist()
        self.loss_min = float(S_grid.min())
        self.full_loss = [float(v) for v in full_loss]
        self.batch_loss = [float(v) for v in batch_loss]
        self.path = np.round(np.asarray(path), 3).tolist()
        self.run_id += 1
